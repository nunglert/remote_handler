"""Runtime patch for the paramiko ssh-agent *certificate* bug.

paramiko 3.5.1 crashes with ``AttributeError: public_blob`` during publickey
auth when the ssh-agent offers an OpenSSH *certificate* (e.g. one loaded by
``step ssh login``) rather than a plain key. paramiko cannot parse the cert
key-type, so ``AgentKey.inner_key`` is ``None``; the auth handler then reads
``key.public_blob``, which ``AgentKey.__getattr__`` cannot proxy, and it
raises instead of returning ``None``.

Upstream issue (unfixed as of the pinned 3.5.1):
    https://github.com/paramiko/paramiko/issues/2462
    https://github.com/paramiko/paramiko/issues/2437

This module fixes it **without modifying the installed paramiko**: it adds a
``public_blob`` property to ``AgentKey`` that returns ``None`` when there is no
inner key. The auth handler then falls back to signing with the cert itself
(``AgentKey.asbytes()`` already returns the full certificate blob and
``sign_ssh_data`` signs via the agent), which is correct certificate auth.

The patch is:
  * idempotent -- safe to call ``apply()`` repeatedly;
  * capability-gated -- it self-detects whether the running paramiko is
    actually buggy and does nothing on a fixed version, so it disappears the
    moment paramiko ships a fix or the pin is bumped.
"""

import logging

from paramiko.agent import AgentKey
from paramiko.message import Message

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_remote_handler_cert_patch"


def _make_cert_agentkey() -> AgentKey:
    """Build an AgentKey whose type is an unparseable cert (inner_key=None).

    Mirrors the state a real ``step``-issued agent certificate lands in,
    without needing an agent, a cert or a network.
    """
    msg = Message()
    msg.add_string("ssh-ed25519-cert-v01@openssh.com")
    msg.add_string(b"\x00\x00\x00\x00unparseable-cert-body")
    return AgentKey(agent=None, blob=msg.asbytes())


def is_buggy() -> bool:
    """True if this paramiko raises on ``AgentKey.public_blob`` for certs."""
    try:
        _ = _make_cert_agentkey().public_blob
        return False
    except AttributeError:
        return True


def _public_blob(self):
    """Proxy to ``inner_key.public_blob``; ``None`` when there is no inner key.

    For cert types paramiko cannot parse, ``inner_key`` is ``None`` and there
    is no separate pubkey blob -- returning ``None`` makes the auth handler
    sign with the certificate itself.
    """
    inner = self.inner_key
    return inner.public_blob if inner is not None else None


def apply() -> bool:
    """Install the patch if needed. Returns True if it was applied now.

    No-op (returns False) when already applied or when the running paramiko is
    not affected by the bug.
    """
    if getattr(AgentKey, _PATCH_FLAG, False):
        return False
    if not is_buggy():
        logger.debug("paramiko AgentKey cert bug absent; no patch applied")
        return False

    AgentKey.public_blob = property(_public_blob)
    setattr(AgentKey, _PATCH_FLAG, True)
    logger.info(
        "Applied paramiko AgentKey.public_blob patch for ssh-agent "
        "certificate support (paramiko cert bug #2462)."
    )
    return True

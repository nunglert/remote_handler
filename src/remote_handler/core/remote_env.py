"""Shared remote-environment concerns for the execution backends.

Single source of truth for the things *every* backend needs but that are not
about *how* a job is scheduled: the on-remote directory layout for array
shards, save-path validation, and conda activation. Keeping them here stops the
SLURM and direct executors from each carrying their own (drifting) copy.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from remote_handler.exceptions import ConfigurationError, FileSystemError

logger = logging.getLogger(__name__)

# Array tasks place shard <i> in "<rootpath>/calcdirs/<i>". The SLURM and direct
# array executors (and the task layer that allocates into these dirs) all
# reference this one name rather than hard-coding the string.
CALCDIRS_DIRNAME = "calcdirs"


def validate_savepath(savepath, remote_fs, *, remote_machine=None) -> None:
    """Reject a ``~`` save path and require it to exist on ``remote_fs``.

    SFTP never expands ``~``, so a tilde path can never resolve; catch it here
    rather than letting it surface deep in a transfer as a bare "No such file".
    """
    text = str(savepath)
    where = f"\nRemote machine: {remote_machine}" if remote_machine else ""
    if text.startswith("~"):
        raise FileSystemError(
            f"Save path '{text}' uses '~', which is not supported: "
            "SFTP does not expand '~'. Use an absolute remote path "
            f"(e.g. '/home/<user>/...') as the cluster config 'savepath'.{where}"
        )
    if not remote_fs.exists(savepath):
        raise FileSystemError(
            "Save path does not exist on the remote filesystem: "
            f"{savepath}\n"
            "Create it on the remote, or set an existing absolute path "
            f"as the cluster config 'savepath'.{where}"
        )


@dataclass(frozen=True)
class CondaEnv:
    """How to activate a conda env, resolved from a ``conda_source`` string."""

    kind: str  # "miniconda" | "micromamba" | "activate"
    activate_source: str  # path to ``source``, or the micromamba eval hook
    env_home: Path  # base dir; environments live in ``env_home/envs/<name>``


def resolve_conda(conda_source: str) -> CondaEnv:
    """Classify a ``conda_source`` into how to activate + where envs live.

    Raises :class:`ConfigurationError` on an unrecognised format. Callers that
    want a best-effort, non-raising variant use :func:`conda_activation_script`.
    """
    source = conda_source.strip()
    if source.endswith("miniconda3"):
        return CondaEnv(
            "miniconda", str(Path(source) / "bin/activate"), Path(source)
        )
    if "micromamba" in source:
        return CondaEnv(
            "micromamba",
            'eval "$(micromamba shell hook --shell bash)"',
            Path(source),
        )
    if "bin/activate" in source:
        return CondaEnv("activate", str(Path(source)), Path(source).parents[1])
    raise ConfigurationError(
        f"Invalid conda_source format: {conda_source}\n"
        "Expected format: /home/user/miniconda3/ or /path/to/bin/activate\n"
        "Note: Only miniconda3 is currently supported"
    )


def validate_conda_on_fs(
    conda_source, conda_env, remote_fs, *, remote_machine
):
    """Resolve ``conda_source`` and check the env exists on ``remote_fs``.

    Returns the resolved ``activate_source`` (what a SLURM template ``source``s).
    Mirrors the historical SLURM-side validation: micromamba is taken on trust
    (the hook needs no on-disk path), everything else must point at a real
    activation script and a real environment.
    """
    spec = resolve_conda(conda_source)
    if spec.kind == "micromamba":
        return spec.activate_source

    if not remote_fs.exists(spec.activate_source):
        raise FileSystemError(
            "Conda activation script not found at: "
            f"{spec.activate_source}. We currently do not "
            "support '~', write the full path!\n"
            f"Remote machine: {remote_machine}"
        )
    conda_env_path = spec.env_home / "envs" / conda_env
    dot_conda_env_path = (
        Path(os.path.expanduser("~")) / ".conda/envs" / conda_env
    )
    if not remote_fs.exists(conda_env_path) and not remote_fs.exists(
        dot_conda_env_path
    ):
        raise ConfigurationError(
            f"Conda environment '{conda_env}' not found on "
            f"{remote_machine}\n"
            f"Expected path: {conda_env_path} or "
            f"{dot_conda_env_path}"
        )
    return spec.activate_source


def conda_activation_script(conda_source, conda_env) -> str:
    """Best-effort shell lines to activate ``conda_env`` (for direct execution).

    Empty or unrecognised config yields ``""`` (no activation -> ambient
    ``python``), so a local or dummy config just runs whatever is on PATH.
    """
    source = (conda_source or "").strip()
    env = (conda_env or "").strip()
    if not source or not env:
        return ""
    try:
        spec = resolve_conda(source)
    except ConfigurationError:
        logger.warning(
            "Unrecognised conda_source '%s'; running without activation.",
            source,
        )
        return ""
    if spec.kind == "micromamba":
        return f"{spec.activate_source}\nmicromamba activate {env}\n"
    return f"source {spec.activate_source}\nconda activate {env}\n"

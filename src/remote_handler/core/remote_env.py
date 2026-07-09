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


def remote_home(remote_fs) -> str:
    """The home directory on the machine ``remote_fs`` points at.

    Used to expand a leading ``~`` for existence checks: the *shell* expands
    ``~`` when it later ``source``s the path, but SFTP existence checks are
    literal, so we resolve it here against the real remote home. Falls back to
    the local home if the remote cannot be queried.
    """
    try:
        result = remote_fs.run_command('printf %s "$HOME"')
        home = (result.stdout or "").strip()
        if home:
            return home
    except Exception:  # noqa: BLE001 - any failure -> local fallback below
        logger.debug("Could not query remote $HOME; using local home.")
    return os.path.expanduser("~")


def _expand_user(path, home: str) -> str:
    """Expand a leading ``~`` / ``~/`` in ``path`` against ``home``."""
    text = str(path)
    if text == "~":
        return home
    if text.startswith("~/"):
        return home + text[1:]
    return text


def resolve_remote_module_file(remote_fs, conda_source, conda_env, module):
    """Locate a Python module's file in the *remote* conda env, or ``None``.

    Runs ``python -c "import <module>; print(<module>.__file__)"`` on the
    machine ``remote_fs`` points at, inside the env named by
    ``conda_source``/``conda_env`` -- so it returns the path as it exists on
    the remote, not on the (dependency-light) host. Works uniformly for a real
    cluster (SSH) and a local ``direct`` run (subprocess). Returns ``None`` if
    the module is absent or the probe fails.
    """
    activation = conda_activation_script(conda_source, conda_env)
    cmd = (
        f"{activation}\n"
        f'python -c "import {module} as _m; print(_m.__file__)"'
    )
    try:
        result = remote_fs.run_command(cmd)
    except Exception:  # noqa: BLE001 - any transport failure -> not resolved
        logger.debug("Remote module probe failed for %s.", module)
        return None
    if result.returncode == 0:
        path = (result.stdout or "").strip().splitlines()
        if path and path[-1]:
            return path[-1]
    return None


def validate_conda_on_fs(
    conda_source, conda_env, remote_fs, *, remote_machine, conda_env_dir=None
):
    """Resolve ``conda_source`` and check the env exists on ``remote_fs``.

    Returns the resolved ``activate_source`` (what a SLURM template ``source``s).
    Mirrors the historical SLURM-side validation: micromamba is taken on trust
    (the hook needs no on-disk path), everything else must point at a real
    activation script and a real environment.

    The environment is looked for under the conda install's ``envs/`` and
    ``~/.conda/envs``. Some setups store their environments elsewhere and have
    neither of those directories; pass ``conda_env_dir`` (the cluster config
    field) to add that directory to the search.

    A leading ``~`` is supported: it is expanded against the remote home for
    the existence checks (the shell expands it at runtime when the rendered
    job script ``source``s the path), so ``~/miniconda3`` is a valid
    ``conda_source`` (and a valid ``conda_env_dir``).
    """
    spec = resolve_conda(conda_source)
    if spec.kind == "micromamba":
        return spec.activate_source

    home = remote_home(remote_fs)
    activate_source = _expand_user(spec.activate_source, home)
    env_home = _expand_user(spec.env_home, home)

    if not remote_fs.exists(activate_source):
        raise FileSystemError(
            f"Conda activation script not found at: {activate_source}\n"
            f"Remote machine: {remote_machine}"
        )

    # Standard locations, plus an explicit override for setups that keep their
    # environments outside both (so neither 'envs/' nor '~/.conda/envs' exists).
    candidates = [
        Path(env_home) / "envs" / conda_env,
        Path(home) / ".conda/envs" / conda_env,
    ]
    if conda_env_dir:
        candidates.insert(
            0, Path(_expand_user(conda_env_dir, home)) / conda_env
        )

    if not any(remote_fs.exists(path) for path in candidates):
        searched = "\n".join(f"  - {path}" for path in candidates)
        hint = (
            ""
            if conda_env_dir
            else (
                "\nIf your environments are stored outside the conda install's "
                "'envs/' and '~/.conda/envs' (e.g. a shared/project envs dir), "
                "set the cluster config 'conda_env_dir' to that directory."
            )
        )
        raise ConfigurationError(
            f"Conda environment '{conda_env}' not found on "
            f"{remote_machine}.\n"
            f"Searched:\n{searched}{hint}"
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

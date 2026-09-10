"""Persistent task handles: submit now, collect later (even after a crash).

A running :class:`~remote_handler.core.tasks.RemoteTask` normally keeps the
information needed to await and collect a job -- the remote job id, the tagged
remote directory, where to copy results -- only in memory. If the driver
process dies, that handle is lost even though the job keeps running on the
cluster.

A :class:`TaskManifest` is that handle made durable: a small JSON snapshot,
written to both the remote job directory and the local machine right after
submission. :class:`TaskHandle` reads one back, rebuilds just enough state
(filesystem + execution backend, re-bound to the recorded job id via
``Executor.attach``) to poll the job and collect its results -- in a
*different* process, from a *different* machine, at a *later* time.

This is what makes two things possible:

* **crash resume** -- reattach to an in-flight job and collect it, and
* **detach-on-submit** -- fire a job and return immediately, so submitting a
  remote job feels like a local, backgrounded command.

A manifest may also carry a **finalizer**: a dotted import path to a plain
``callable(local_path, **finalizer_kwargs)`` that a project-specific task
(e.g. a ``Sampler``/``ModelTrainer`` living outside this package) registers
before submission by setting ``self.finalizer`` / ``self.finalizer_kwargs``
on the ``RemoteTask``. ``TaskHandle.collect`` invokes it right after the raw
directory copy, so ``remote_handler collect`` reproduces whatever
domain-specific assembly the in-process code path would have done (e.g.
concatenating per-replica trajectories into one file, or picking model
artifacts out of an array job's subdirectories) even when the job is
resumed from a manifest in a fresh process. It is best-effort: a failing
finalizer never hides an otherwise-successful raw collect (see
``TaskHandle.collect``).

Scope: the manifest records everything needed to *await and collect* a job, not
to *resubmit* it. Re-launching a crashed job (which needs the script/input
paths and node-retry state) is a deliberate later extension.
"""

import importlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from remote_handler.core import executors, file_utils
from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.file_handler import (
    FileSystemHandlerBase,
    LocalFileSystem,
    RemoteFileSystem,
)
from remote_handler.exceptions import TaskError

logger = logging.getLogger(__name__)

#: Filename used for the manifest dropped inside the remote job directory.
MANIFEST_NAME = "task_manifest.json"

#: Bump when the on-disk schema changes incompatibly;
#: :meth:`TaskManifest.from_dict` rejects manifests it does not understand
#: rather than mis-resuming.
MANIFEST_VERSION = 1

TASK_SINGLE = "single"
TASK_ARRAY = "array"


def _resolve_finalizer(dotted: str) -> Callable:
    """Import a ``"pkg.module:callable_name"`` finalizer reference.

    Raises ``TaskError`` with a clear message on a bad path -- an unresolvable
    finalizer means whatever registered it (a ``RemoteTask`` subclass in a
    downstream project) shipped a manifest a fresh process/environment can't
    honor, e.g. because that project's package isn't installed here.
    """
    module_name, _, attr = dotted.partition(":")
    if not module_name or not attr:
        raise TaskError(
            f"Malformed finalizer {dotted!r}; expected 'pkg.module:callable'."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TaskError(
            f"Could not import finalizer module {module_name!r} "
            f"(from {dotted!r}): {exc}"
        ) from exc
    try:
        fn = getattr(module, attr)
    except AttributeError as exc:
        raise TaskError(
            f"Finalizer module {module_name!r} has no attribute {attr!r} "
            f"(from {dotted!r})."
        ) from exc
    if not callable(fn):
        raise TaskError(f"Finalizer {dotted!r} is not callable.")
    return fn


@dataclass
class TaskManifest:
    """A durable, self-contained snapshot of a submitted task.

    Holds the minimal state a fresh process needs to rebuild the execution
    backend, re-bind it to the running job, and collect the results:

    * ``cluster_config`` is embedded in full (not by name) so a resume does not
      depend on the original cluster JSON still living in the config dir;
    * ``remote_savepath`` is the *resolved, tagged* remote directory (the one
      the executor actually ran in), which is otherwise unrecoverable because
      the tag is generated at construction time;
    * ``job_id`` is the scheduler handle (SLURM job id); ``None`` for the
      ``direct`` backend, which detects completion via a sentinel file.
    * ``finalizer`` / ``finalizer_kwargs`` are the optional post-collect hook
      described in the module docstring; ``finalizer_kwargs`` must be
      JSON-plain since it round-trips through this same JSON snapshot.
    """

    task_type: str
    backend: str
    job_id: Optional[str]
    remote_savepath: str
    local_savepath: str
    jobname: str
    cluster_config: dict
    fs_type: str  # "local" | "remote"
    n_processors: int = 1
    loop_time: float = 30.0
    num_parallel: int = 10
    num_array_tasks: Optional[int] = None
    finalizer: Optional[str] = None
    finalizer_kwargs: dict = field(default_factory=dict)
    version: int = field(default=MANIFEST_VERSION)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskManifest":
        got = data.get("version")
        if got != MANIFEST_VERSION:
            raise TaskError(
                f"Unsupported task manifest version {got!r} "
                f"(this build understands {MANIFEST_VERSION})."
            )
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    # -- persistence ------------------------------------------------------

    def save_local(self, path: os.PathLike) -> Path:
        """Write the manifest to ``path`` on the local machine."""
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        LocalFileSystem().write_dict_to_json(path, self.to_dict())
        logger.info("Wrote task manifest to %s", path)
        return path

    def save_remote(self, remote_fs: FileSystemHandlerBase) -> Path:
        """Write the manifest into the remote job directory.

        A copy living next to the job survives loss of the local machine, so a
        resume can be driven from anywhere that can reach the cluster.
        """
        remote_path = Path(self.remote_savepath) / MANIFEST_NAME
        remote_fs.write_dict_to_json(remote_path, self.to_dict())
        logger.info("Wrote task manifest to %s (remote)", remote_path)
        return remote_path

    @classmethod
    def load(cls, path: os.PathLike) -> "TaskManifest":
        """Load a manifest from a local JSON file."""
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    # -- reconstruction ---------------------------------------------------

    def build_cluster_config(self) -> ClusterConfig:
        return ClusterConfig.from_dict(self.cluster_config)

    def build_fs(self) -> FileSystemHandlerBase:
        """Rebuild the filesystem handler this task ran against."""
        if self.fs_type == "local":
            return LocalFileSystem()
        cfg = self.build_cluster_config()
        return RemoteFileSystem(
            remote_alias=cfg.remote_machine,
            connection_policy=cfg.connection_policy,
        )


class TaskHandle:
    """A resumed, in-flight task rebuilt from a :class:`TaskManifest`.

    Provides the await/collect half of a task's lifecycle without its
    submit half: :meth:`is_finished` / :meth:`wait` poll the recorded job and
    :meth:`collect` copies the remote job directory back to the local machine.

    Construct via :meth:`from_manifest` or :meth:`from_file`.
    """

    def __init__(
        self,
        manifest: TaskManifest,
        remote_fs: Optional[FileSystemHandlerBase] = None,
    ):
        self.manifest = manifest
        self.remote_fs = remote_fs or manifest.build_fs()
        self.cluster_config = manifest.build_cluster_config()
        self._executor = None
        #: Result of the registered finalizer (see the module docstring),
        #: set by :meth:`collect`. ``None`` if no finalizer is registered, or
        #: if it hasn't run yet.
        self.finalized_output: Optional[Any] = None
        #: Exception raised by the finalizer, if it failed. A failing
        #: finalizer never hides an otherwise-successful raw collect, so
        #: check this rather than relying on ``collect`` to raise.
        self.finalizer_error: Optional[Exception] = None

    @property
    def executor(self):
        """The execution backend, built (and job-attached) on first use.

        Built lazily so :meth:`collect` -- which only needs the filesystem --
        never triggers the backend's construction-time validation (SLURM/conda
        probes). That keeps "the job already finished, just fetch the files"
        working even if the remote env has since drifted.
        """
        if self._executor is None:
            self._executor = self._build_executor()
        return self._executor

    @classmethod
    def from_manifest(cls, manifest: TaskManifest) -> "TaskHandle":
        return cls(manifest)

    @classmethod
    def from_file(cls, path: os.PathLike) -> "TaskHandle":
        """Load a local manifest JSON and rebuild the handle from it."""
        return cls(TaskManifest.load(path))

    def _build_executor(self):
        m = self.manifest
        common = dict(
            jobname=m.jobname,
            savepath=Path(m.remote_savepath),
            n_processors=m.n_processors,
            remote_fs=self.remote_fs,
            loop_time=m.loop_time,
            slurm_header_add="",
        )
        if m.task_type == TASK_ARRAY:
            executor = executors.make_array_executor(
                self.cluster_config, num_parallel=m.num_parallel, **common
            )
        elif m.task_type == TASK_SINGLE:
            executor = executors.make_single_executor(
                self.cluster_config, **common
            )
        else:
            raise TaskError(f"Unknown task_type {m.task_type!r} in manifest.")
        # Re-bind to the already-running job rather than submitting a new one.
        executor.attach(m.job_id)
        return executor

    def is_finished(self) -> bool:
        """Non-blocking: has the resumed job finished?"""
        return self.executor.is_finished()

    def wait(self) -> None:
        """Block until the resumed job finishes."""
        self.executor.wait_until_finished()

    def collect(self, move: bool = True) -> Path:
        """Copy the remote job directory to the manifest's local savepath.

        Mirrors ``ResultCollector.collect_directory`` so a resumed handle
        collects results identically to the original in-process task, then
        runs the registered finalizer (if any) -- see :meth:`run_finalizer`.
        """
        remote_path = Path(self.manifest.remote_savepath)
        local_path = Path(self.manifest.local_savepath)
        file_utils.secure_declare_path(
            local_path, move=move, is_directory=True
        )
        logger.info(
            "Collecting results from %s to %s", remote_path, local_path
        )
        self.remote_fs.get_dir(remote_path, local_path)
        logger.info("Results collected successfully")
        self.run_finalizer(local_path)
        return local_path

    def run_finalizer(self, local_path: Path) -> Optional[Any]:
        """Run the task-specific finalizer registered at submit time, if any.

        Resolves ``manifest.finalizer`` (a ``"pkg.module:callable"`` path)
        and calls it as ``callable(local_path, **manifest.finalizer_kwargs)``.
        This is how a project-specific task (e.g. a ``Sampler`` living
        outside this package) gets its own post-collect assembly step -- e.g.
        concatenating per-replica trajectories, or picking model artifacts
        out of an array job's subdirectories -- run even when the job is
        resumed from a manifest rather than collected in-process.

        Best-effort: a failing finalizer is logged and recorded on
        ``self.finalizer_error`` rather than raised, so a bug in
        project-specific post-processing never hides the raw files this
        already copied. Callers that care should check ``finalizer_error``
        after calling (``collect`` does not raise it either).
        """
        self.finalized_output = None
        self.finalizer_error = None
        if not self.manifest.finalizer:
            return None
        try:
            fn = _resolve_finalizer(self.manifest.finalizer)
            self.finalized_output = fn(
                local_path, **self.manifest.finalizer_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - never hide a good collect
            self.finalizer_error = exc
            logger.warning(
                "Finalizer %s failed: %s", self.manifest.finalizer, exc
            )
            return None
        logger.info(
            "Finalizer %s produced: %s",
            self.manifest.finalizer,
            self.finalized_output,
        )
        return self.finalized_output

    def wait_and_collect(self, move: bool = True) -> Path:
        """Convenience: block until done, then collect."""
        self.wait()
        return self.collect(move=move)

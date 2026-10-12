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

Scope: the manifest records everything needed to *await and collect* a job, not
to *resubmit* it. Re-launching a crashed job (which needs the script/input
paths and node-retry state) is a deliberate later extension.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

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
        return self.build_cluster_config().remote_fs()


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
        collects results identically to the original in-process task.
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
        return local_path

    def wait_and_collect(self, move: bool = True) -> Path:
        """Convenience: block until done, then collect."""
        self.wait()
        return self.collect(move=move)

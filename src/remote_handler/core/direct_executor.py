"""Scheduler-free execution backend.

Runs job scripts directly on the target -- a local subprocess, or a single
detached SSH exec -- instead of submitting them to SLURM. It honours the same
``RemoteTask``-facing contract as the SLURM executors (``write_jobfile`` /
``submit`` / ``wait_until_finished`` + ``calcdir``), so a task runs unchanged
on a machine with no scheduler, or locally, when
``ClusterConfig.executor == "direct"``.

Jobs are launched detached (``nohup ... &``) and completion is detected by
polling a sentinel file through the (resilience-aware) filesystem handler --
so the launching SSH call returns immediately and a connection that drops
mid-job is survivable, matching the package's connection policy.

This is the minimal, scheduler-free backend: shards run sequentially and conda
activation is best-effort. Bounded parallelism and stricter per-shard error
propagation are deliberate future extensions, not present here.
"""

import logging
import time
from pathlib import Path

from remote_handler.core import remote_env
from remote_handler.core.executors import ArrayExecutor, SingleExecutor

logger = logging.getLogger(__name__)

# Touched unconditionally once the launched work returns (success or not).
# Distinct from a job's own ``is_finished.info`` success marker: this only says
# "the process exited", which is what ``wait_until_finished`` blocks on.
SENTINEL = ".executor_finished"


def _conda_activation(cluster_config) -> str:
    """Best-effort conda activation lines for this config's launch script."""
    return remote_env.conda_activation_script(
        getattr(cluster_config, "conda_source", ""),
        getattr(cluster_config, "conda_env", ""),
    )


class _DirectExecutorBase:
    """Shared construction + launch + polling for direct executors."""

    def __init__(
        self,
        jobname,
        cluster_config,
        savepath,
        n_processors,
        remote_fs,
        loop_time=10.0,
        jobfile_name="run_direct.sh",
        **_ignored,  # tolerate SLURM-only kwargs forwarded by the factory
    ):
        self.jobname = jobname
        self.cluster_config = cluster_config
        self.savepath = Path(savepath)
        self.n_processors = n_processors
        self.remote_fs = remote_fs
        self.loop_time = loop_time
        self.jobfile_name = jobfile_name
        self.is_written = False
        self.is_submitted = False

        self._validate_savepath()

    def _validate_savepath(self):
        remote_env.validate_savepath(
            self.savepath,
            self.remote_fs,
            remote_machine=getattr(
                self.cluster_config, "remote_machine", None
            ),
        )

    def _launch(self, workdir: Path):
        """Launch ``jobfile_name`` detached in ``workdir`` and return at once.

        Runs over ``remote_fs`` -- the same connection (and resilience policy)
        used for file transfer -- so local and remote launches share one path
        and the remote leg needs no separate ``ssh`` subprocess.
        """
        cmd = (
            f"cd {workdir} && nohup bash {self.jobfile_name} "
            f"> {workdir}/.direct_launch.log 2>&1 &"
        )
        self.remote_fs.run_command(cmd)
        logger.info("Launched direct job '%s' in %s", self.jobname, workdir)
        self.is_submitted = True

    def wait_until_finished(self):
        """Block until the completion sentinel appears."""
        while True:
            time.sleep(self.loop_time)
            if self.remote_fs.exists(self._sentinel):
                break

    def is_finished(self) -> bool:
        """Non-blocking completion poll: has the sentinel been touched yet?

        Sentinel-based, so it works for a job launched by another process
        (a resumed handle) with no scheduler handle needed.
        """
        return bool(self.remote_fs.exists(self._sentinel))


class DirectSingleExecutor(_DirectExecutorBase, SingleExecutor):
    """Runs one script against one input file, no scheduler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calcdir = self.savepath
        self.script_path = self.calcdir / self.jobfile_name
        self._sentinel = self.calcdir / SENTINEL

    def write_jobfile(
        self, input_file, python_script_path, exclude_nodes=None
    ):
        # exclude_nodes is a scheduler concept; ignored (no nodes here).
        body = (
            "#!/bin/bash\n"
            f"{_conda_activation(self.cluster_config)}"
            f'python "{python_script_path}" "{input_file}"\n'
            f'touch "{self._sentinel}"\n'
        )
        self.remote_fs.write_str_to_file(body, self.script_path)
        self.is_written = True

    def submit(self):
        assert self.is_written, "Cannot submit before write_jobfile()."
        self._launch(self.calcdir)


class DirectArrayExecutor(_DirectExecutorBase, ArrayExecutor):
    """Runs a script across shards in ``<savepath>/calcdirs/<i>``, no scheduler."""

    def __init__(self, *args, map_indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_indices = map_indices
        self.rootpath = self.savepath
        # Must match SlurmArrayHandler so the task allocates shards here.
        self.calcdir = self.rootpath / remote_env.CALCDIRS_DIRNAME
        self.output_path = self.rootpath / "direct_output"
        self.remote_fs.makedirs(self.calcdir, exist_ok=True)
        self.remote_fs.makedirs(self.output_path, exist_ok=True)
        self.script_path = self.rootpath / self.jobfile_name
        self._sentinel = self.rootpath / SENTINEL

    def write_jobfile(self, python_script_path, num_array_tasks):
        # Honour map_indices (set by recompute_failed) like the SLURM array;
        # otherwise run shards 0..N-1.
        if self.map_indices is not None:
            indices = list(self.map_indices)
        else:
            indices = list(range(num_array_tasks))

        lines = [
            "#!/bin/bash",
            _conda_activation(self.cluster_config).rstrip(),
        ]
        for run_id in indices:
            shard = self.calcdir / str(run_id)
            lines.append(f'python "{python_script_path}" "{shard}"')
        lines.append(f'touch "{self._sentinel}"')
        body = "\n".join(line for line in lines if line) + "\n"

        self.remote_fs.write_str_to_file(body, self.script_path)
        self.is_written = True

    def submit(self):
        assert self.is_written, "Cannot submit before write_jobfile()."
        self._launch(self.rootpath)

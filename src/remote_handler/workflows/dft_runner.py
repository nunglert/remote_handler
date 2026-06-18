"""Single-point ground-truth handlers (DFT and other ASE calculators).

All single-point evaluation on a remote cluster follows the same shape: shard
the configurations one-per-job, ship each with a small JSON describing the
calculator, run a remote payload that attaches an ASE calculator and writes a
labelled ``dft_<DB>.extxyz`` + ``is_finished.info``, then collect. The only
thing that varies between VASP, ORCA, EMT, ... is *which* calculator the
payload builds and the settings it receives.

:class:`SinglePointHandler` owns that shared machinery; :class:`VASPHandler`
and :class:`ORCAHandler` are thin presets that only assemble their calculator
config and default remote script. Their public signatures are unchanged.
"""

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from remote_handler.core import file_handler, tasks
from remote_handler.core.atoms_handler import RemoteAtomsHandler
from remote_handler.core.cluster_handler import ClusterConfig

logger = logging.getLogger(__name__)

# Remote payloads shipped with the package live in ``scripts/`` next to this
# module's package root. Resolved at runtime so there is no hard-coded absolute
# path (the old per-user defaults only existed on one machine).
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# Default VASP INCAR settings (single-point r2SCAN). A module constant so the
# mutable dict is never shared as a function default argument.
_DEFAULT_INCAR_KWARGS = {
    "xc": "R2SCAN",
    "encut": 300,
    "prec": "Normal",
    "ismear": -1,
    "sigma": 0.025,
    "ibrion": -1,
    "ediff": 1e-5,
    "symprec": 1e-7,
    "lwave": False,
    "lcharg": False,
    "lreal": False,
    "kspacing": 0.3,
}

_DEFAULT_VASP_HEADER = (
    "#SBATCH --time=0-02:00:00\n"
    "module purge\n"
    "module load vasp/6.4.2_intel\n"
    "source /opt/intel/oneapi/setvars.sh --force\n"
    "export VASP_PP_PATH=/home/aserafini/cellar/vasp/paw"
)


class SinglePointHandler(RemoteAtomsHandler):
    """Run one ASE-calculator single-point per configuration as a job array.

    Calculator-agnostic: it ships ``calc_config`` (the per-job input JSON) and
    runs ``python_script_path`` on the remote. Concrete calculators are
    expressed as presets that fill in ``calc_config`` and the default script
    (see :class:`VASPHandler`, :class:`ORCAHandler`).

    Args:
        traj_files: Trajectory file(s) with the configurations to evaluate.
        cluster_config: Target cluster configuration.
        remote_fs: Filesystem handler for the machine running the evaluation.
        savepath: Remote root path for the computation.
        num_parallel: Number of array jobs to run in parallel.
        python_script_path: Remote payload that builds the calculator and
            writes ``dft_<DB>.extxyz`` + ``is_finished.info``.
        calc_config: Calculator settings written into each job's input JSON.
        loop_time: Minutes between polls for completion.
        jobname: SLURM job name.
        every: Take every ``every``-th frame from the trajectories.
        start: First frame index to take.
        end: Last frame index to take.
        local_savepath: Local path the results are collected to.
        logfile: Log file name.
        collect_only: If True, only collect results (no submission).
        tag: Tag prepended to collected trajectory filenames.
        n_processors: Processors requested per job.
        slurm_header_add: Extra SLURM header lines.
    """

    def __init__(
        self,
        traj_files: list[Union[str, os.PathLike]],
        cluster_config: ClusterConfig,
        remote_fs: file_handler.FileSystemHandlerBase,
        savepath: str,
        num_parallel: int,
        python_script_path: os.PathLike,
        calc_config: dict,
        loop_time: float = 30,
        jobname: str = "DFT_Eval",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "dft.log",
        collect_only: bool = False,
        tag: str = "dft",
        n_processors: int = 8,
        slurm_header_add: str = "",
    ):
        super().__init__(
            traj_files,
            cluster_config,
            remote_fs,
            1,  # shard_size: one structure per array task
            savepath,
            num_parallel,
            loop_time,
            jobname,
            every,
            start,
            end,
            local_savepath,
            logfile,
            collect_only,
            tag,
            n_processors,
            slurm_header_add=slurm_header_add,
        )

        self.logger = tasks.SimpleLogger(logfile)
        self.calc_config = dict(calc_config)
        # RemoteAtomsHandler does not set these for us.
        self.python_script_file = str(python_script_path)
        self.shared_ressources = []

        if not collect_only:
            self.prepare_individual_data()

    def prepare_individual_data(self):
        """Write one input JSON + one structure file per array task."""
        config_files = []
        atoms_batches = []
        for calc_idx in range(self.num_array_tasks):
            json_config = deepcopy(self.calc_config)
            # Filename the payload reads the structure from / writes back to.
            json_config["DATABASE"] = "dft.extxyz"

            config_files.append(
                tasks.DictJSON(json_config, "input_dft.json", self.remote_fs)
            )
            atoms_batches.append(
                tasks.AtomsBatch(
                    self.shards[calc_idx],
                    json_config["DATABASE"],
                    self.remote_fs,
                )
            )

        del self.shards
        # Shape (num_obj, num_tasks) expected by the array task.
        self.individual_ressources = [config_files, atoms_batches]

    def finish_check(self, required_fraction: float = 0.8):
        """Check the fraction of jobs that wrote ``is_finished.info``.

        Args:
            required_fraction: Minimum success fraction to return True.
        """
        assert self.all_calcpaths, "There are no calcpaths to check!"

        num_converged = 0
        for calcpath in self.all_calcpaths:
            if self.remote_fs.exists(Path(calcpath) / "is_finished.info"):
                num_converged += 1

        finished_fraction = num_converged / len(self.all_calcpaths)
        if finished_fraction < 0.5:
            logger.warning("More than 50% of the submitted runs failed..")
        return finished_fraction >= required_fraction

    def run(self):
        """Allocate inputs, submit the array, and collect results."""
        self.allocate_data()
        self.submit()
        self.collect_results()


class VASPHandler(SinglePointHandler):
    """VASP single-point preset of :class:`SinglePointHandler`.

    Builds the VASP ``calc_config`` (INCAR tags + launch command) and defaults
    to the shipped ``scripts/dft_singlepoint.py`` payload. Public signature is
    unchanged from the historical handler.

    Args:
        traj_files: Trajectory file(s) with configurations to evaluate.
        cluster_config: Target cluster configuration.
        remote_fs: Filesystem handler for the evaluation machine.
        savepath: Remote root path for the computation.
        num_parallel: Number of array jobs to run in parallel.
        loop_time: Minutes between polls for completion.
        jobname: SLURM job name.
        every: Take every ``every``-th frame from the trajectories.
        start: First frame index to take.
        end: Last frame index to take.
        local_savepath: Local path the results are collected to.
        logfile: Log file name.
        collect_only: If True, only collect results (no submission).
        tag: Tag prepended to collected trajectory filenames.
        python_script_path: Remote payload; defaults to the shipped
            ``dft_singlepoint.py``.
        stub: Accepted for backward compatibility; stubbing is handled by
            ``VASPEvaluator``, not the handler.
        launcher: MPI launch command prefix (e.g. ``"mpiexec -np"``).
        slurm_header_add: Extra SLURM header lines.
        vasp_binary: Name of the VASP executable.
        n_processors: Processors requested per job.
        incar_kwargs: VASP INCAR tags; defaults to a single-point r2SCAN set.
    """

    def __init__(
        self,
        traj_files: list[Union[str, os.PathLike]],
        cluster_config: ClusterConfig,
        remote_fs: file_handler.FileSystemHandlerBase,
        savepath: str,
        num_parallel: int,
        loop_time: float = 30,
        jobname: str = "DFT_Eval",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "dft.log",
        collect_only: bool = False,
        tag: str = "dft",
        python_script_path: Optional[str] = None,
        stub: bool = False,
        launcher: str = "mpiexec -np",
        slurm_header_add: str = _DEFAULT_VASP_HEADER,
        vasp_binary: str = "vasp_std",
        n_processors: int = 8,
        incar_kwargs: Optional[dict] = None,
    ):
        if incar_kwargs is None:
            incar_kwargs = dict(_DEFAULT_INCAR_KWARGS)
        if python_script_path is None:
            python_script_path = str(_SCRIPTS_DIR / "dft_singlepoint.py")

        calc_config = {
            "INCAR_kwargs": incar_kwargs,
            "launch_command": f"{launcher} {n_processors} {vasp_binary}",
        }

        super().__init__(
            traj_files=traj_files,
            cluster_config=cluster_config,
            remote_fs=remote_fs,
            savepath=savepath,
            num_parallel=num_parallel,
            python_script_path=python_script_path,
            calc_config=calc_config,
            loop_time=loop_time,
            jobname=jobname,
            every=every,
            start=start,
            end=end,
            local_savepath=local_savepath,
            logfile=logfile,
            collect_only=collect_only,
            tag=tag,
            n_processors=n_processors,
            slurm_header_add=slurm_header_add,
        )


class ORCAHandler(SinglePointHandler):
    """ORCA single-point preset of :class:`SinglePointHandler`.

    Builds the ORCA ``calc_config`` and defaults to the shipped
    ``scripts/orca_singlepoint.py`` payload. Keeps its historical
    ``(traj_files, remote_fs, cluster_config, ...)`` argument order
    (``remote_fs`` and ``cluster_config`` swapped relative to
    :class:`VASPHandler`) for backward compatibility.

    Args:
        traj_files: Trajectory file(s) with configurations to evaluate.
        remote_fs: Filesystem handler for the evaluation machine.
        cluster_config: Target cluster configuration.
        savepath: Remote root path for the computation.
        num_parallel: Number of array jobs to run in parallel.
        loop_time: Minutes between polls for completion.
        jobname: SLURM job name.
        every: Take every ``every``-th frame from the trajectories.
        start: First frame index to take.
        end: Last frame index to take.
        local_savepath: Local path the results are collected to.
        logfile: Log file name.
        collect_only: If True, only collect results (no submission).
        tag: Tag prepended to collected trajectory filenames.
        python_script_path: Remote payload; defaults to the shipped
            ``orca_singlepoint.py``.
        slurm_header_add: Extra SLURM header lines.
        orca_binary: Full path to the ORCA executable.
        n_processors: Processors requested per job.
        orcasimpleinput: ORCA simple-input line (functional, basis, method).
    """

    def __init__(
        self,
        traj_files: list[Union[str, os.PathLike]],
        remote_fs: file_handler.FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        savepath: str,
        num_parallel: int,
        loop_time: float = 30,
        jobname: str = "DFT_Eval",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "orca.log",
        collect_only: bool = False,
        tag: str = "orca",
        python_script_path: Optional[str] = None,
        slurm_header_add: str = "#SBATCH --time=0-02:00:00\nmodule",
        orca_binary: str = "/opt/ohpc/pub/dft/orca_6_1_0/orca",
        n_processors: int = 8,
        orcasimpleinput: str = "wB97X 6-31G(d) ENGRAD",
    ):
        if python_script_path is None:
            python_script_path = str(_SCRIPTS_DIR / "orca_singlepoint.py")

        calc_config = {
            "orcasimpleinput": orcasimpleinput,
            "orcablocks": f"%pal nprocs {n_processors} end",
            "orca_binary": orca_binary,
        }

        super().__init__(
            traj_files=traj_files,
            cluster_config=cluster_config,
            remote_fs=remote_fs,
            savepath=savepath,
            num_parallel=num_parallel,
            python_script_path=python_script_path,
            calc_config=calc_config,
            loop_time=loop_time,
            jobname=jobname,
            every=every,
            start=start,
            end=end,
            local_savepath=local_savepath,
            logfile=logfile,
            collect_only=collect_only,
            tag=tag,
            n_processors=n_processors,
            slurm_header_add=slurm_header_add,
        )

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
import warnings
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


def _resolve_environment_header(
    cluster_config, environment_key, slurm_header_add
):
    """Resolve the SLURM header lines for an application on a cluster.

    The cluster declares *how* an application is set up on it (modules, env
    scripts, paths) under ``ClusterConfig.environment[<key>]`` -- that is the
    supported home for this setup. ``slurm_header_add`` is the legacy per-call
    knob, kept for backward compatibility.

    The two are **alternatives, never combined** (so setup is never
    duplicated), with this precedence:

    * ``slurm_header_add`` given -> it wins, verbatim. Existing callers that
      pass a header keep their exact behaviour. If the cluster *also* declares
      ``environment[key]``, that entry is ignored and a ``DeprecationWarning``
      is emitted -- move the setup into ``environment`` and drop the header.
    * only ``environment[key]`` -> the cluster's setup (the preferred way).
    * neither -> ``""`` plus a warning that nothing is configured for this app.
    """
    if not environment_key:
        return slurm_header_add
    env_setup = (getattr(cluster_config, "environment", {}) or {}).get(
        environment_key, ""
    )
    machine = getattr(cluster_config, "remote_machine", "?")

    if slurm_header_add:
        if env_setup:
            warnings.warn(
                f"Both a slurm_header_add and "
                f"ClusterConfig.environment['{environment_key}'] are set for "
                f"cluster '{machine}'. slurm_header_add takes precedence and "
                "the environment entry is ignored (so the two are never "
                f"combined). Move the '{environment_key}' setup into the "
                "cluster's environment map and stop passing slurm_header_add; "
                "supplying both is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
        return slurm_header_add

    if env_setup:
        return env_setup

    logger.warning(
        "No '%s' environment configured for cluster '%s' (and no "
        "slurm_header_add given); the job may not find %s on the remote. "
        "Add an 'environment' entry to the cluster JSON, e.g. "
        '{"environment": {"%s": "module load ..."}}.',
        environment_key,
        machine,
        environment_key,
        environment_key,
    )
    return ""


def resolve_incar_kwargs(incar_kwargs=None, incar_config_file=None) -> dict:
    """Resolve the VASP INCAR settings from an explicit dict and/or a YAML file.

    The INCAR tags can be supplied as a dict (``incar_kwargs``) or read from a
    YAML file (``incar_config_file``):

    - ``incar_config_file`` only -> the YAML contents.
    - ``incar_kwargs`` only -> that dict, as-is (historical behaviour).
    - both -> the YAML as a base, with ``incar_kwargs`` overriding on top.
    - neither -> the built-in single-point r2SCAN defaults.
    """
    if incar_config_file is not None:
        import yaml

        with open(incar_config_file) as handle:
            base = yaml.safe_load(handle) or {}
        return {**base, **incar_kwargs} if incar_kwargs else base
    if incar_kwargs is not None:
        return incar_kwargs
    return dict(_DEFAULT_INCAR_KWARGS)


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
        slurm_header_add: Legacy per-call SLURM header lines. If given, it
            takes precedence over the cluster's ``environment[environment_key]``
            (a ``DeprecationWarning`` fires when both are set); the two are
            never combined.
        environment_key: Key into ``cluster_config.environment`` for this
            application's cluster setup (e.g. ``"vasp"``). ``None`` injects
            nothing -- the historical behaviour.
        time_limit: SLURM ``--time`` value (e.g. ``"0-02:00:00"``) to cap each
            job -- useful to kill hanging DFT runs. ``None`` (default) emits no
            ``--time`` directive, so the job runs to the partition maximum.
        memory: SLURM ``--mem`` value (e.g. ``"16G"``, ``"2000M"``, or MB as a
            bare number) requested per job. ``None`` (default) emits no
            ``--mem`` directive, so the job gets the partition default.
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
        environment_key: Optional[str] = None,
        time_limit: Optional[str] = None,
        memory: Optional[str] = None,
    ):
        slurm_header_add = _resolve_environment_header(
            cluster_config, environment_key, slurm_header_add
        )
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
            time_limit=time_limit,
            memory=memory,
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
        incar_config_file: Optional path to a YAML file of INCAR tags, used instead of
            (or as a base for) ``incar_kwargs`` -- see
            :func:`resolve_incar_kwargs`.
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
        slurm_header_add: str = "",
        vasp_binary: str = "vasp_std",
        n_processors: int = 8,
        incar_kwargs: Optional[dict] = None,
        incar_config_file: Optional[os.PathLike] = None,
        time_limit: Optional[str] = None,
        memory: Optional[str] = None,
    ):
        incar_kwargs = resolve_incar_kwargs(incar_kwargs, incar_config_file)
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
            environment_key="vasp",
            time_limit=time_limit,
            memory=memory,
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
        slurm_header_add: str = "",
        orca_binary: str = "/opt/ohpc/pub/dft/orca_6_1_0/orca",
        n_processors: int = 8,
        orcasimpleinput: str = "wB97X 6-31G(d) ENGRAD",
        time_limit: Optional[str] = None,
        memory: Optional[str] = None,
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
            environment_key="orca",
            time_limit=time_limit,
            memory=memory,
        )

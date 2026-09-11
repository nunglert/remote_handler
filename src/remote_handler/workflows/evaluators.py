"""Ground-truth evaluators for active-learning workflows.

An evaluator labels sampled configurations with ground truth (energy + forces)
by running a single-point calculation per configuration on a remote machine.
Every calculator follows the same shape, so the shared machinery lives in
:class:`AseCalculatorEvaluator`; :class:`VASPEvaluator` and
:class:`ORCAEvaluator` are thin presets that only assemble their calculator
config. Most evaluators support ``stub=True`` to dry-run the loop without real
compute.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
from ase import Atoms

from remote_handler.core.atoms_handler import RemoteAtomsHandler
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import DetachableTaskMixin
from remote_handler.core.utils import grep
from remote_handler.exceptions import WorkflowError

from .active_learning_base import GroundTruthEvaluator

logger = logging.getLogger(__name__)

# Remote payloads shipped with the package live in ``scripts/`` next to this
# module's package root.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# The shipped single-point payloads write the labelled structure to
# ``dft_<DATABASE>``; SinglePointHandler always sets DATABASE = "dft.extxyz".
_LABELLED_FILENAME = "dft_dft.extxyz"


class AseCalculatorEvaluator(DetachableTaskMixin, GroundTruthEvaluator):
    """Single-point ground-truth evaluation with any ASE calculator.

    Calculator-agnostic: it ships ``calc_config`` (the per-job input JSON) and
    runs ``singlepoint_script`` on the remote, which builds the calculator,
    computes energy + forces, and writes a labelled ``dft_<DB>.extxyz``.
    Concrete calculators are presets that assemble ``calc_config`` (see
    :class:`VASPEvaluator`, :class:`ORCAEvaluator`).

    Args:
        singlepoint_script: Remote payload that builds the calculator.
        remote_fs: Filesystem handler for the evaluation machine.
        cluster_config: Target cluster configuration.
        num_parallel: Number of parallel jobs to queue.
        calc_config: Calculator settings written into each job's input JSON.
        remote_savepath: Remote path for the calculation files.
        loop_time: Seconds between polls for completion.
        required_fraction: Fraction of jobs that must succeed to proceed.
        n_processors: CPU cores per job.
        stub: If True, produce mock outputs without running anything.
        slurm_header_add: Extra SLURM header lines (appended after any
            cluster-declared ``environment`` setup for this calculator).
        time_limit: SLURM ``--time`` value (e.g. ``"0-02:00:00"``) capping each
            job. ``None`` (default) emits no ``--time``, so the job runs to the
            partition maximum.
        collect_only: If True, ``update`` reconstructs the array submitted by
            an earlier, ``collect_only=False`` call (from the manifest
            written beside ``<calcdir>/dft/results``, via
            ``RemoteAtomsHandler.from_manifest``) instead of
            allocating/submitting a new one, and ``get_evaluated_configs``
            skips the (already-done) submission. ``evaluator_input`` is not
            read in this mode. Use this to collect a detached evaluation run
            from a fresh process.
    """

    # Key into ``ClusterConfig.environment`` for this calculator's cluster
    # setup (modules / paths). Presets override it (e.g. VASP -> "vasp"); the
    # generic base and pure-ASE presets (EMT) inject nothing.
    _environment_key: Optional[str] = None

    def __init__(
        self,
        singlepoint_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        num_parallel: int,
        calc_config: dict,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        required_fraction: float = 0.5,
        n_processors: int = 8,
        stub: bool = False,
        slurm_header_add: str = "",
        time_limit: Optional[str] = None,
        collect_only: bool = False,
    ):
        super().__init__()
        self.singlepoint_script = singlepoint_script
        self.calc_config = dict(calc_config)
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.num_parallel = num_parallel
        self.remote_savepath = (
            Path(remote_savepath) if remote_savepath else None
        )
        self.loop_time = loop_time
        self.required_fraction = required_fraction
        self.n_processors = n_processors
        self.stub = stub
        self.slurm_header_add = slurm_header_add
        self.time_limit = time_limit
        self.collect_only = collect_only
        self.handler = None

    def update(
        self,
        calcdir: Path,
        evaluator_input: Optional[Path | list[Path]],
        **kwargs,
    ) -> None:
        """Prepare the evaluator for a new iteration.

        Args:
            calcdir: Directory for this AL iteration.
            evaluator_input: Path(s) to unlabelled configs to evaluate. Not
                read when ``collect_only`` -- a second process reconstructing
                an already-submitted array from its manifest needs none of
                the original input, so ``None`` is expected there.
            **kwargs: Additional parameters.
        """
        self.dft_dir = Path(calcdir) / "dft"
        self.results_dir = self.dft_dir / "results"
        os.makedirs(self.dft_dir, exist_ok=True)

        if not self.collect_only:
            self.traj_files = (
                [str(evaluator_input)]
                if isinstance(evaluator_input, Path)
                else [str(p) for p in evaluator_input]
            )

        if self.stub:
            # No remote handler is needed; outputs are mocked in get_*.
            self.handler = None
            return

        def _build_handler():
            # Local import keeps the evaluator import light and avoids any
            # cycle -- only needed on this, non-collect_only path.
            from remote_handler.workflows.dft_runner import SinglePointHandler

            return SinglePointHandler(
                traj_files=self.traj_files,
                cluster_config=self.cluster_config,
                remote_fs=self.remote_fs,
                savepath=str(self.remote_savepath)
                if self.remote_savepath
                else "/tmp",
                num_parallel=self.num_parallel,
                python_script_path=self.singlepoint_script,
                calc_config=self.calc_config,
                loop_time=self.loop_time,
                jobname="dft",
                every=1,
                local_savepath=self.results_dir,
                logfile="dft.log",
                collect_only=False,
                tag="dft",
                n_processors=self.n_processors,
                slurm_header_add=self.slurm_header_add,
                environment_key=self._environment_key,
                time_limit=self.time_limit,
            )

        # collect_only=True reconstructs via RemoteAtomsHandler.from_manifest
        # (the base class -- see its docstring for why: it never re-reads/
        # -shards the original trajectories) rather than SinglePointHandler,
        # which is exactly why the collect-only path here needs neither
        # calc_config/python_script_path/environment_key nor the original
        # evaluator_input. The result still supports finish_check() and the
        # generic collect_results() -- everything get_evaluated_configs()
        # below uses.
        self.handler = self.build_or_resume_task(
            RemoteAtomsHandler, self.results_dir, _build_handler, tag="dft"
        )

    def get_evaluated_configs(self) -> Path:
        """Run the evaluation and collect the labelled configurations.

        Returns:
            Path to the ``.extxyz`` of evaluated configurations.

        Raises:
            WorkflowError: If fewer than ``required_fraction`` jobs succeed.
        """
        if self.stub:
            logger.info("Running in stub mode - creating mock outputs")
            all_ats = self._create_stub_outputs(num_configs=5)
        else:
            # Submit (unless collect_only -- already submitted in another
            # process), wait, then collect -- rather than the single
            # handler.run() convenience call, which unconditionally re-runs
            # allocate_data() and would be wrong for a collect_only handler.
            self.submit_if_needed(self.handler)
            if not self.handler.finish_check(self.required_fraction):
                raise WorkflowError(
                    f"Less than {self.required_fraction * 100}% ran "
                    "successfully. Something probably went wrong. Otherwise, "
                    "lower the required_fraction parameter."
                )
            self.handler.collect_results()
            all_ats = self._collect_results()

        os.makedirs(self.results_dir, exist_ok=True)
        output_path = self.results_dir / "evaluated_configs.extxyz"
        ase.io.write(output_path, all_ats)
        logger.info(
            f"Wrote {len(all_ats)} evaluated configurations to {output_path}"
        )
        return output_path

    def _collect_results(self) -> list[Atoms]:
        """Read each job's labelled extxyz and copy E/F into REF_* fields."""
        all_ats = []
        for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
            labelled = Path(calcpath) / _LABELLED_FILENAME
            try:
                at = ase.io.read(labelled)
            except Exception as e:  # noqa: BLE001 - report & skip a failed job
                logger.info(f"evaluation failed for {calcpath=}: {e}")
                continue
            at.info["REF_energy"] = at.get_potential_energy()
            at.arrays["REF_forces"] = at.get_forces(apply_constraint=False)
            at.calc = None
            all_ats.append(at)
            logger.info(f"evaluation successful for {calcpath=}")
        return all_ats

    def _create_stub_outputs(self, num_configs: int = 5) -> list[Atoms]:
        """Create mock labelled outputs (water molecules) for testing."""
        logger.info("Creating stub outputs for testing")

        mock_atoms = []
        for _ in range(num_configs):
            positions = np.array(
                [
                    [0.0, 0.0, 0.0],  # O
                    [0.96, 0.0, 0.0],  # H
                    [-0.24, 0.93, 0.0],  # H
                ]
            )
            positions += np.random.normal(0, 0.1, positions.shape)

            atoms = Atoms(
                symbols="OH2",
                positions=positions,
                cell=[10, 10, 10],
                pbc=True,
            )
            atoms.info["REF_energy"] = -14.5 + np.random.normal(0, 0.1)
            atoms.arrays["REF_forces"] = np.random.normal(0, 0.5, (3, 3))
            atoms.calc = None
            mock_atoms.append(atoms)

        return mock_atoms


class VASPEvaluator(AseCalculatorEvaluator):
    """VASP DFT preset of :class:`AseCalculatorEvaluator`.

    Args:
        remote_fs: Filesystem handler for the evaluation machine.
        cluster_config: Target cluster configuration.
        num_parallel: Number of parallel SLURM jobs to queue.
        singlepoint_script: Remote payload; defaults to the shipped
            ``dft_singlepoint.py``.
        incar_kwargs: VASP INCAR tags (optional if ``incar_config_file`` is given).
        remote_savepath: Remote path for the calculation files.
        launcher: VASP launch prefix (e.g. ``"mpirun -n"``).
        stub: If True, produce mock outputs without running VASP.
        loop_time: Seconds between polls for completion.
        required_fraction: Fraction of jobs that must succeed.
        n_processors: CPU cores per VASP job.
        slurm_header_add: Legacy per-call SLURM header lines. If given it takes
            precedence over the cluster's ``environment["vasp"]`` (and emits a
            DeprecationWarning when both are set). Prefer putting the VASP
            module load and ``VASP_PP_PATH`` in that cluster JSON entry.
        incar_config_file: Optional path to a YAML file of INCAR tags, used instead of
            (or as a base for) ``incar_kwargs``.
        collect_only: If True, ``update`` reconstructs the array submitted by
            an earlier, ``collect_only=False`` call instead of
            allocating/submitting a new one. See
            ``AseCalculatorEvaluator``'s docstring.
    """

    _environment_key = "vasp"

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        num_parallel: int,
        singlepoint_script: Optional[os.PathLike] = None,
        incar_kwargs: Optional[dict] = None,
        remote_savepath: Optional[os.PathLike] = None,
        launcher: str = "mpirun -n",
        stub: bool = False,
        loop_time: float = 30,
        required_fraction: float = 0.5,
        n_processors: int = 8,
        slurm_header_add: str = "",
        incar_config_file: Optional[os.PathLike] = None,
        time_limit: Optional[str] = None,
        collect_only: bool = False,
    ):
        from remote_handler.workflows.dft_runner import resolve_incar_kwargs

        if singlepoint_script is None:
            singlepoint_script = str(_SCRIPTS_DIR / "dft_singlepoint.py")
        self.incar_kwargs = resolve_incar_kwargs(
            incar_kwargs, incar_config_file
        )
        self.launcher = launcher
        calc_config = {
            "INCAR_kwargs": self.incar_kwargs,
            "launch_command": f"{launcher} {n_processors} vasp_std",
        }
        super().__init__(
            singlepoint_script=singlepoint_script,
            remote_fs=remote_fs,
            cluster_config=cluster_config,
            num_parallel=num_parallel,
            calc_config=calc_config,
            remote_savepath=remote_savepath,
            loop_time=loop_time,
            required_fraction=required_fraction,
            n_processors=n_processors,
            stub=stub,
            slurm_header_add=slurm_header_add,
            time_limit=time_limit,
            collect_only=collect_only,
        )

    @property
    def vasp_handler(self):
        """Backward-compatible alias for the underlying handler."""
        return self.handler


class ORCAEvaluator(AseCalculatorEvaluator):
    """ORCA DFT preset of :class:`AseCalculatorEvaluator`.

    Args:
        cluster_config: Target cluster configuration.
        remote_fs: Filesystem handler for the evaluation machine.
        num_parallel: Number of parallel SLURM jobs to queue.
        orcasimpleinput: ORCA simple-input line (functional, basis, method).
        singlepoint_script: Remote payload; defaults to the shipped
            ``orca_singlepoint.py``.
        remote_savepath: Remote path for the calculation files.
        loop_time: Seconds between polls for completion.
        required_fraction: Fraction of jobs that must succeed.
        n_processors: CPU cores per ORCA job.
        orca_binary: Full path to the ORCA executable.
        stub: If True, produce mock outputs without running ORCA.
        slurm_header_add: Legacy per-call SLURM header lines. If given it takes
            precedence over the cluster's ``environment["orca"]`` (and emits a
            DeprecationWarning when both are set). Prefer putting the ORCA
            module load in that cluster JSON entry.
        collect_only: If True, ``update`` reconstructs the array submitted by
            an earlier, ``collect_only=False`` call instead of
            allocating/submitting a new one. See
            ``AseCalculatorEvaluator``'s docstring.
    """

    _environment_key = "orca"

    def __init__(
        self,
        cluster_config: ClusterConfig,
        remote_fs: FileSystemHandlerBase,
        num_parallel: int,
        orcasimpleinput: str,
        singlepoint_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        required_fraction: float = 0.8,
        n_processors: int = 2,
        orca_binary: str = "/opt/ohpc/pub/dft/orca_6_1_0/orca",
        stub: bool = False,
        slurm_header_add: str = "",
        time_limit: Optional[str] = None,
        collect_only: bool = False,
    ):
        if singlepoint_script is None:
            singlepoint_script = str(_SCRIPTS_DIR / "orca_singlepoint.py")
        self.orcasimpleinput = orcasimpleinput
        calc_config = {
            "orcasimpleinput": orcasimpleinput,
            "orcablocks": f"%pal nprocs {n_processors} end",
            "orca_binary": orca_binary,
        }
        super().__init__(
            singlepoint_script=singlepoint_script,
            remote_fs=remote_fs,
            cluster_config=cluster_config,
            num_parallel=num_parallel,
            calc_config=calc_config,
            remote_savepath=remote_savepath,
            loop_time=loop_time,
            required_fraction=required_fraction,
            n_processors=n_processors,
            stub=stub,
            slurm_header_add=slurm_header_add,
            time_limit=time_limit,
            collect_only=collect_only,
        )

    @property
    def orca_handler(self):
        """Backward-compatible alias for the underlying handler."""
        return self.handler


class EMTEvaluator(AseCalculatorEvaluator):
    """Cheap EMT ground-truth preset (pure ASE, no external binary).

    Uses the generic ``ase_singlepoint.py`` payload with ASE's EMT calculator.
    EMT supports only a handful of elements (Ni, Cu, Pd, Ag, Pt, Au, Al, H, C,
    N, O), but needs no licensed software -- ideal for examples and CI runs
    over the ``direct`` executor, where VASP/ORCA cannot run.

    Args:
        remote_fs: Filesystem handler for the evaluation machine.
        cluster_config: Target cluster configuration.
        num_parallel: Number of parallel jobs to queue.
        singlepoint_script: Remote payload; defaults to the shipped
            ``ase_singlepoint.py``.
        calculator_kwargs: Keyword arguments forwarded to ``EMT(...)``.
        remote_savepath: Remote path for the calculation files.
        loop_time: Seconds between polls for completion.
        required_fraction: Fraction of jobs that must succeed.
        n_processors: CPU cores per job.
        stub: If True, produce mock outputs without running anything.
        slurm_header_add: Extra SLURM header lines.
        collect_only: If True, ``update`` reconstructs the array submitted by
            an earlier, ``collect_only=False`` call instead of
            allocating/submitting a new one. See
            ``AseCalculatorEvaluator``'s docstring.
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        num_parallel: int,
        singlepoint_script: Optional[os.PathLike] = None,
        calculator_kwargs: Optional[dict] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        required_fraction: float = 0.5,
        n_processors: int = 1,
        stub: bool = False,
        slurm_header_add: str = "",
        time_limit: Optional[str] = None,
        collect_only: bool = False,
    ):
        if singlepoint_script is None:
            singlepoint_script = str(_SCRIPTS_DIR / "ase_singlepoint.py")
        calc_config = {
            "calculator": "emt",
            "calculator_kwargs": calculator_kwargs or {},
        }
        super().__init__(
            singlepoint_script=singlepoint_script,
            remote_fs=remote_fs,
            cluster_config=cluster_config,
            num_parallel=num_parallel,
            calc_config=calc_config,
            remote_savepath=remote_savepath,
            loop_time=loop_time,
            required_fraction=required_fraction,
            n_processors=n_processors,
            stub=stub,
            slurm_header_add=slurm_header_add,
            time_limit=time_limit,
            collect_only=collect_only,
        )

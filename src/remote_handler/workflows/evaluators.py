"""
Evaluators for Active Learning Workflows.

Evaluators perform ground truth calculations (e.g., DFT) on sampled configurations.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
from ase import Atoms

from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.utils import grep
from remote_handler.exceptions import WorkflowError

from .active_learning_base import GroundTruthEvaluator

logger = logging.getLogger(__name__)


class VASPEvaluator(GroundTruthEvaluator):
    """
    Ground truth evaluator using VASP DFT calculations.

    Runs VASP single-point calculations on configurations in parallel using SLURM.
    Supports stub mode for testing without actual VASP execution.

    Args:
        singlepoint_script: Path to dft_singlepoint.py or compatible evaluation script
        remote_fs: FileSystemHandlerBase for the machine where evaluation runs
        num_parallel: Number of parallel SLURM jobs to queue
        incar_kwargs: VASP INCAR tags as dictionary
        remote_savepath: Path on remote machine for calculation files
        launcher: Launch command for VASP (e.g., "mpirun -n")
        stub: If True, creates mock outputs without running VASP
        loop_time: Interval (seconds) to poll for job completion
        required_fraction: Fraction of samples that must succeed
        n_processors: Number of CPU cores per VASP job
    """

    def __init__(
        self,
        singlepoint_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        num_parallel: int,
        incar_kwargs: dict,
        remote_savepath: Optional[os.PathLike] = None,
        launcher: str = "mpirun -n",
        stub: bool = False,
        loop_time: float = 30,
        required_fraction: float = 0.5,
        n_processors: int = 2,
    ):
        super().__init__()
        self.singlepoint_script = singlepoint_script
        self.incar_kwargs = incar_kwargs
        self.remote_fs = remote_fs
        self.num_parallel = num_parallel
        self.stub = stub
        self.launcher = launcher
        self.loop_time = loop_time
        self.required_fraction = required_fraction
        self.remote_savepath = (
            Path(remote_savepath) if remote_savepath else None
        )
        self.n_processors = n_processors

    def update(
        self, calcdir: Path, evaluator_input: Path | list[Path], **kwargs
    ) -> None:
        """
        Prepare evaluator for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            evaluator_input: Path to trajectory file(s) with configurations to evaluate
            **kwargs: Additional parameters
        """
        self.dft_dir = Path(calcdir) / "dft"
        self.results_dir = self.dft_dir / "results"
        os.makedirs(self.dft_dir, exist_ok=True)

        traj_files = (
            [str(evaluator_input)]
            if isinstance(evaluator_input, Path)
            else [str(p) for p in evaluator_input]
        )

        # Store for later use
        self.traj_files = traj_files

        # In stub mode, we don't need VASPHandler
        if not self.stub:
            from remote_handler.core.cluster_handler import ClusterConfig
            from remote_handler.dft_runner import VASPHandler

            # Create minimal cluster config
            cluster_config = ClusterConfig(
                jobfile_name="run_vasp.sh",
                n_processors=self.n_processors,
            )

            self.vasp_handler = VASPHandler(
                traj_files=traj_files,
                cluster_config=cluster_config,
                remote_fs=self.remote_fs,
                savepath=str(self.remote_savepath)
                if self.remote_savepath
                else "/tmp",
                num_parallel=self.num_parallel,
                loop_time=self.loop_time,
                jobname="dft",
                every=1,
                start=None,
                end=None,
                local_savepath=self.results_dir,
                logfile="dft.log",
                collect_only=False,
                tag="dft",
                python_script_path=self.singlepoint_script,
                stub=self.stub,
                launcher=self.launcher,
                n_processors=self.n_processors,
                incar_kwargs=self.incar_kwargs,
            )

    def _create_stub_outputs(self, num_configs: int = 5) -> list[Atoms]:
        """
        Create mock VASP outputs for testing.

        Generates fake DFT results with realistic energies and forces.

        Args:
            num_configs: Number of configurations to generate

        Returns:
            List of Atoms objects with mock DFT results
        """
        logger.info("Creating stub VASP outputs for testing")

        mock_atoms = []
        for i in range(num_configs):
            # Create a simple water molecule as mock structure
            positions = np.array(
                [
                    [0.0, 0.0, 0.0],  # O
                    [0.96, 0.0, 0.0],  # H
                    [-0.24, 0.93, 0.0],  # H
                ]
            )
            # Add small random perturbations
            positions += np.random.normal(0, 0.1, positions.shape)

            atoms = Atoms(
                symbols="OH2",
                positions=positions,
                cell=[10, 10, 10],
                pbc=True,
            )

            # Mock DFT results
            energy = -14.5 + np.random.normal(0, 0.1)  # Mock energy in eV
            forces = np.random.normal(0, 0.5, (3, 3))  # Mock forces

            atoms.info["REF_energy"] = energy
            atoms.arrays["REF_forces"] = forces
            atoms.calc = None  # No calculator attached

            mock_atoms.append(atoms)

        return mock_atoms

    def get_evaluated_configs(self) -> Path:
        """
        Run VASP evaluations and collect results.

        Returns:
            Path to XYZ file containing evaluated configurations

        Raises:
            Exception: If less than required_fraction of calculations succeed
        """
        if self.stub:
            # In stub mode, create mock outputs directly
            logger.info("Running in stub mode - creating mock VASP outputs")
            all_ats = self._create_stub_outputs(num_configs=5)
        else:
            # Run actual VASP calculations
            self.vasp_handler.run()

            if not self.vasp_handler.finish_check(self.required_fraction):
                raise WorkflowError(
                    f"Less than {self.required_fraction * 100}% ran successfully. "
                    "Something probably went wrong. Otherwise, lower the "
                    "required_fraction parameter."
                )

            # Collect results from VASP runs
            all_ats = []
            for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
                try:
                    at = ase.io.read(Path(calcpath) / "vasprun.xml")
                    all_ats.append(at)
                except Exception as e:
                    logger.info(f"VASP run failed for {calcpath=}: {e}")

            # Add REF labels to results
            for s in all_ats:
                s.info["REF_energy"] = s.get_potential_energy()
                s.arrays["REF_forces"] = s.get_forces()
                s.calc = None

        # Write results to file
        os.makedirs(self.results_dir, exist_ok=True)
        output_path = self.results_dir / "evaluated_configs.extxyz"
        ase.io.write(output_path, all_ats)
        logger.info(
            f"Wrote {len(all_ats)} evaluated configurations to {output_path}"
        )

        return output_path

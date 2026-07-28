"""
Samplers for Active Learning Workflows.

Samplers generate new configurations to be evaluated by exploring configuration space.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.exceptions import WorkflowError

from .active_learning_base import Sampler

logger = logging.getLogger(__name__)

# Remote payloads shipped with the package live in ``scripts/`` next to this
# module's package root. Each sampler defaults its ``sampling_script`` to the
# shipped payload it was written for, so the path need not be passed in.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def update_default_args(args, default):
    """
    Update the default arguments with user-provided values.
    """
    for key, value in args.items():
        default[key] = value


class CMAESSampler(Sampler):
    """
    Sampler using CMA-ES evolutionary optimization with MACE models.

    Runs CMA-ES evolution to explore configuration space, optimizing atomic
    structures using neural network force field energy evaluation.
    Supports stub mode for testing without actual CMA-ES execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        cluster_config: ClusterConfig with partition, conda env, template settings
        founder_path: Path to founder POSCAR structure file
        sampling_script: Remote payload; defaults to the shipped
            ``clinamen2/cmaes_sampling.py``.
        label: Run label for the evolution
        generations: Number of CMA-ES generations to run
        step_size: Initial CMA-ES step size
        random_seed: Random seed for evolution
        pop_size: Population size (None uses CMA-ES default)
        save_nth: Save every nth generation
        batch: If True, NN evaluation is vmapped
        loss_mode: Loss function ('energy-only', 'loss-threshold', etc.)
        loss_threshold: Lower threshold for loss
        scaled_center: Scaled center as [x, y, z] for DOF sphere
        radius: Radius of sphere around scaled_center
        c_r: c_r parameter for covariance matrix bias
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        do_optimization: If True, optimize the winner after evolution
        dof_mode: Degree-of-freedom mode for optimization ('full-dof' or 'defect')
        surface_axis: Surface normal axis for optimization (0, 1, or 2)
        fix_range: Scaled range of atoms to fix during optimization
        bulk_terminate: If True, remove overlayer from bottom layer
        opt_optimizer: ASE optimizer class name for structure optimization
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        founder_path: os.PathLike,
        sampling_script: Optional[os.PathLike] = None,
        label: str = "cmaes_run",
        generations: int = 1000,
        step_size: float = 0.12,
        random_seed: int = 0,
        pop_size: Optional[int] = None,
        save_nth: int = 1,
        batch: bool = False,
        loss_mode: str = "energy-only",
        loss_threshold: float = -1e6,
        scaled_center: Optional[list[float]] = None,
        radius: Optional[float] = None,
        c_r: float = 20.0,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        do_optimization: bool = False,
        dof_mode: str = "full-dof",
        surface_axis: int = 2,
        fix_range: Optional[list[float]] = None,
        bulk_terminate: bool = False,
        opt_optimizer: str = "FIRE",
    ):
        super().__init__()
        if sampling_script is None:
            sampling_script = _SCRIPTS_DIR / "clinamen2" / "cmaes_sampling.py"
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.sampling_script = sampling_script
        self.founder_path = founder_path
        self.label = label
        self.generations = generations
        self.step_size = step_size
        self.random_seed = random_seed
        self.pop_size = pop_size
        self.save_nth = save_nth
        self.batch = batch
        self.loss_mode = loss_mode
        self.loss_threshold = loss_threshold
        self.scaled_center = scaled_center
        self.radius = radius
        self.c_r = c_r
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub
        self.do_optimization = do_optimization
        self.dof_mode = dof_mode
        self.surface_axis = surface_axis
        self.fix_range = fix_range
        self.bulk_terminate = bulk_terminate
        self.opt_optimizer = opt_optimizer

    def update(
        self, calcdir: Path, sampler_input: list[Path], **kwargs
    ) -> None:
        """
        Prepare sampler for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: List of model file paths
            **kwargs: Additional parameters
        """
        self.sample_dir = calcdir / "sample"
        self.results_dir = self.sample_dir / "results"
        os.makedirs(self.sample_dir, exist_ok=True)

        self.model_files = sampler_input

        if not self.stub:
            script_input = {
                "founder": Path(self.founder_path).name,
                "label": self.label,
                "models": [p.name for p in self.model_files],
                "random_seed": self.random_seed,
                "step_size": self.step_size,
                "generations": self.generations,
                "pop_size": self.pop_size,
                "save_nth": self.save_nth,
                "batch": self.batch,
                "loss_mode": self.loss_mode,
                "loss_threshold": self.loss_threshold,
                "scaled_center": self.scaled_center,
                "radius": self.radius,
                "c_r": self.c_r,
                "do_optimization": self.do_optimization,
                "dof_mode": self.dof_mode,
                "surface_axis": self.surface_axis,
                "fix_range": self.fix_range,
                "bulk_terminate": self.bulk_terminate,
                "optimizer": self.opt_optimizer,
            }

            self.script_input_file = self.sample_dir / "input_cmaes.json"

            update_default_args(kwargs, script_input)

            with open(self.script_input_file, "w") as f:
                f.write(json.dumps(script_input, indent=2))

            self.sampling_task = SingleScriptTask(
                local_savepath=self.results_dir,
                input_file=self.script_input_file,
                cluster_config=self.cluster_config,
                remote_fs=self.remote_fs,
                script_file=self.sampling_script,
                remote_savepath=self.remote_savepath,
                jobfile_name="run_cmaes.sh",
                jobname="cmaes_sample",
                logfile="cmaes_sample.log",
                n_processors=1,
                loop_time=self.loop_time,
                collect_only=False,
            )

            self.sampling_task.allocate_data()

            # Copy required files to remote
            self.sampling_task.copy_file(self.founder_path)
            for p in self.model_files:
                self.sampling_task.copy_file(p)

    def _create_stub_outputs(self) -> list[Atoms]:
        """
        Create mock CMA-ES sampling outputs for testing.

        Returns:
            List of Atoms objects with mock evolution results
        """
        logger.info("Creating stub CMA-ES samples")

        mock_samples = []
        for i in range(5):
            symbols = ["Si"] * 63
            positions = np.random.uniform(0, 10.86, (len(symbols), 3))

            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[10.86, 10.86, 10.86],
                pbc=True,
            )

            atoms.info["generation"] = i
            atoms.info["loss"] = -100.0 + np.random.uniform(-5, 5)
            mock_samples.append(atoms)

        return mock_samples

    def get_sampled_configs(self) -> Path:
        """
        Run CMA-ES evolution and collect configurations.

        Returns:
            Path to XYZ file containing sampled configurations
        """
        if self.stub:
            logger.info("Running in stub mode - creating mock CMA-ES samples")
            sampled_configs = self._create_stub_outputs()

            output_dir = self.results_dir / "out"
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / "samples.extxyz"
            ase.io.write(output_path, sampled_configs)
        else:
            self.sampling_task.submit()
            if not self.sampling_task.finish_check():
                raise WorkflowError(
                    "CMA-ES sampling did not finish successfully "
                    "(no is_finished.info marker). Check the job logs in "
                    f"{self.sampling_task.calcdir}."
                )
            self.sampling_task.collect_results()
            output_path = self.results_dir / self.label

        logger.info(f"CMA-ES sampled configurations written to {output_path}")
        return output_path


class BatchCMAESSampler(CMAESSampler):
    """
    Runs a batch of CMA-ES evolutions with different random seeds.

    Each evolution is an independent CMAESSampler run. Child seeds are derived
    deterministically from the master random_seed. If do_optimization=True on
    the parent, each evolution also optimizes its winner within the same
    SLURM job.

    Args:
        batch_size: Number of independent CMA-ES evolutions to run
        **remaining args: Passed to CMAESSampler
    """

    def __init__(
        self,
        batch_size: int,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        founder_path: os.PathLike,
        sampling_script: Optional[os.PathLike] = None,
        label: str = "cmaes_run",
        generations: int = 1000,
        step_size: float = 0.12,
        random_seed: int = 0,
        pop_size: Optional[int] = None,
        save_nth: int = 1,
        batch: bool = False,
        loss_mode: str = "energy-only",
        loss_threshold: float = -1e6,
        scaled_center: Optional[list[float]] = None,
        radius: Optional[float] = None,
        c_r: float = 20.0,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        do_optimization: bool = False,
        dof_mode: str = "full-dof",
        surface_axis: int = 2,
        fix_range: Optional[list[float]] = None,
        bulk_terminate: bool = False,
        opt_optimizer: str = "FIRE",
    ):
        super().__init__(
            remote_fs,
            cluster_config,
            founder_path,
            sampling_script,
            label,
            generations,
            step_size,
            random_seed,
            pop_size,
            save_nth,
            batch,
            loss_mode,
            loss_threshold,
            scaled_center,
            radius,
            c_r,
            remote_savepath,
            loop_time,
            stub,
            do_optimization,
            dof_mode,
            surface_axis,
            fix_range,
            bulk_terminate,
            opt_optimizer,
        )
        self.batch_size = batch_size

    def update(
        self, calcdir: Path, sampler_input: list[Path], **kwargs
    ) -> None:
        """
        Prepare N CMA-ES evolution tasks with different random seeds.

        Args:
            calcdir: Base directory for this AL iteration
            sampler_input: List of model file paths
            **kwargs: Additional parameters passed to each CMAESSampler.update()
        """
        self.base_calcdir = calcdir
        self.model_files = sampler_input

        rng = np.random.default_rng(self.random_seed)
        child_seeds = rng.integers(0, 2**31, size=self.batch_size)

        self.sampling_tasks = []

        # Temporarily mutate label/seed per run, restore after loop
        _orig_label = self.label
        _orig_seed = self.random_seed

        for run_idx, seed in enumerate(child_seeds):
            self.random_seed = int(seed)
            self.label = f"{_orig_label}_{run_idx}"

            run_calcdir = Path(calcdir) / f"run_{run_idx}"
            super().update(
                calcdir=run_calcdir, sampler_input=sampler_input, **kwargs
            )

            if not self.stub:
                self.sampling_tasks.append(self.sampling_task)

        self.random_seed = _orig_seed
        self.label = _orig_label

    def get_sampled_configs(self) -> list[Path]:
        """
        Submit all CMA-ES evolutions and collect results.

        Returns:
            List of Paths to each evolution's output directory
        """
        if self.stub:
            logger.info("Running in stub mode - creating mock batch samples")
            output_paths = []
            for run_idx in range(self.batch_size):
                run_dir = (
                    self.base_calcdir / f"run_{run_idx}" / "sample" / "results"
                )
                output_dir = run_dir / "out"
                os.makedirs(output_dir, exist_ok=True)
                mock_samples = self._create_stub_outputs()
                output_path = output_dir / "samples.extxyz"
                ase.io.write(output_path, mock_samples)
                output_paths.append(output_path)
            return output_paths

        output_paths = []
        for run_idx, task in enumerate(self.sampling_tasks):
            label = f"{self.label}_{run_idx}"
            task.submit()
            if not task.finish_check():
                raise WorkflowError(
                    f"CMA-ES evolution {run_idx} did not finish successfully "
                    "(no is_finished.info marker). Check the job logs in "
                    f"{task.calcdir}."
                )
            task.collect_results()

            results_dir = Path(task.local_savepath)
            if self.do_optimization:
                winner_poscar = (
                    results_dir / "optimized" / f"POSCAR_{label}_winner_end"
                )
                winner_extxyz = (
                    results_dir / "optimized" / f"{label}_winner_end.extxyz"
                )
                at = ase.io.read(winner_poscar, format="vasp")
                ase.io.write(winner_extxyz, at)
                output_paths.append(winner_extxyz)
            else:
                output_paths.append(results_dir / f"{label}_winner.extxyz")

        logger.info(f"Batch CMA-ES completed {len(output_paths)} evolutions")
        return output_paths

"""
Samplers for Active Learning Workflows.

Samplers generate new configurations to be evaluated by exploring configuration space.
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import ase.io
import numpy as np
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.exceptions import (
    DependencyError,
    FileSystemError,
    WorkflowError,
)
from remote_handler.scripts.input_models import (
    CmaesSamplingInput,
    LammpsSamplingInput,
    TorchSimSamplingInput,
)
from remote_handler.workflows.selectors import ConfigSelector

from .active_learning_base import Sampler

logger = logging.getLogger(__name__)


def update_default_args(args, default):
    """
    Update the default arguments with user-provided values.
    """
    for key, value in args.items():
        default[key] = value


class LammpsSampler(Sampler):
    """
    MD sampler using LAMMPS with uncertainty-based sampling.

    Runs LAMMPS molecular dynamics and selects configurations based on model uncertainty.
    Supports stub mode for testing without actual LAMMPS execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        sampling_script: Path to lammps_sampling.py or compatible script
        template_path: Path to jinja2 template for lammps.in file
        data_path: Path to lammps.data file with starting configuration
        element_mapping: Dict mapping atom type numbers to element symbols
        num_samples: Number of configurations to sample per generation
        uncertainty_cutoff_radius: Cutoff radius (Angstrom) for local uncertainty
        remote_lammps_executable_path: Path to LAMMPS executable on remote machine
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        sampling_script: os.PathLike,
        template_path: os.PathLike,
        data_path: os.PathLike,
        element_mapping: dict,
        num_samples: int,
        uncertainty_cutoff_radius: float,
        remote_lammps_executable_path: os.PathLike,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
    ):
        super().__init__()
        self.remote_fs = remote_fs
        self.sampling_script = sampling_script
        self.template_path = template_path
        self.data_path = data_path
        self.element_mapping = element_mapping
        self.num_samples = num_samples
        self.uncertainty_cutoff_radius = uncertainty_cutoff_radius
        self.remote_lammps_executable_path = remote_lammps_executable_path
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub

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
        os.makedirs(self.results_dir, exist_ok=True)

        self.model_files = sampler_input

        # In stub mode, we don't need to create the task
        if not self.stub:
            script_input = {
                "lammps_path": Path(
                    self.remote_lammps_executable_path
                ).as_posix(),
                "template_file": Path(self.template_path).name,
                "data_file": Path(self.data_path).name,
                "model_files": [p.name for p in self.model_files],
                "lammps_config_name": "lammps.in",
                "element_mapping": self.element_mapping,
                "cutoff_radius": self.uncertainty_cutoff_radius,
                "bin_count": self.num_samples,
                "seed": np.random.randint(1e6),
            }

            # Validate input dict against the script's expected schema
            LammpsSamplingInput.from_dict(script_input)

            self.script_input_file = self.results_dir / "input_sample.json"
            with open(self.script_input_file, "w") as f:
                f.write(json.dumps(script_input, indent=2))

            self.sampling_task = SingleScriptTask(
                local_savepath=self.results_dir,
                input_file=self.script_input_file,
                remote_fs=self.remote_fs,
                script_file=self.sampling_script,
                remote_savepath=self.remote_savepath,
                jobfile_name="run_sample.sh",
                jobname="sample",
                logfile="sample.log",
                n_processors=1,
                loop_time=self.loop_time,
                collect_only=False,
            )

            self.sampling_task.allocate_data()

            # Copy required files to remote
            self.sampling_task.copy_file(self.template_path)
            self.sampling_task.copy_file(self.data_path)
            for p in self.model_files:
                self.sampling_task.copy_file(p)

    def _create_stub_outputs(self, num_samples: int = None) -> list[Atoms]:
        """
        Create mock LAMMPS sampling outputs for testing.

        Generates fake MD trajectory with realistic uncertainty scores.

        Args:
            num_samples: Number of samples to generate (uses self.num_samples if None)

        Returns:
            List of Atoms objects with mock uncertainties
        """
        if num_samples is None:
            num_samples = self.num_samples

        logger.info(f"Creating {num_samples} stub LAMMPS samples")

        mock_samples = []
        for i in range(num_samples):
            # Create mock Cu/Co surface structure
            symbols = ["Cu"] * 20 + ["Co"] * 4 + ["H"] * 2
            positions = np.random.uniform(0, 10, (len(symbols), 3))

            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[10, 10, 15],
                pbc=True,
            )

            # Add mock uncertainty scores
            atoms.info["max_uncertainty"] = np.random.uniform(0.1, 2.0)
            atoms.info["mean_uncertainty"] = np.random.uniform(0.05, 1.0)
            atoms.info["step"] = i * 100

            mock_samples.append(atoms)

        return mock_samples

    def get_sampled_configs(self) -> Path:
        """
        Run LAMMPS sampling and collect configurations.

        Returns:
            Path to XYZ file containing sampled configurations
        """
        if self.stub:
            # In stub mode, create mock outputs directly
            logger.info("Running in stub mode - creating mock LAMMPS samples")
            sampled_configs = self._create_stub_outputs()

            # Write to expected output location
            output_dir = self.results_dir / "out"
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / "samples.xyz"
            ase.io.write(output_path, sampled_configs)
        else:
            # Run actual LAMMPS sampling
            self.sampling_task.submit()
            self.sampling_task.collect_results()
            output_path = (
                Path(self.sampling_task.local_savepath) / "out/samples.xyz"
            )

        logger.info(f"Sampled configurations written to {output_path}")
        return output_path


class CMAESSampler(Sampler):
    """
    Sampler using CMA-ES evolutionary optimization with MACE models.

    Runs CMA-ES evolution to explore configuration space, optimizing atomic
    structures using neural network force field energy evaluation.
    Supports stub mode for testing without actual CMA-ES execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        cluster_config: ClusterConfig with partition, conda env, template settings
        sampling_script: Path to cmaes_sampling.py or compatible script
        founder_path: Path to founder POSCAR structure file
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
        sampling_script: os.PathLike,
        founder_path: os.PathLike,
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

            # Validate input dict against the script's expected schema
            CmaesSamplingInput.from_dict(script_input)

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
        sampling_script: os.PathLike,
        founder_path: os.PathLike,
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
            sampling_script,
            founder_path,
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


class NSTrajectorySampler(Sampler):
    """
    Sampler using Nested Sampling with JAXNEST.

    Performs nested sampling runs to explore configuration space with uncertainty quantification.
    Supports stub mode for testing without actual NS execution.

    Args:
        ns_exe: Path to JAXNEST executable on remote machine
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        initial_input_file: Path to NS input file
        n_gpu_parallel: Number of GPUs to use in parallel
        sample_size: Total number of configurations to sample
        resample_size: Number of configs to resample based on uncertainty (optional)
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        energy_uncty_weight: Weight for energy uncertainty in resampling
        force_uncty_weight: Weight for force uncertainty in resampling
        optimize_traj: Optional trajectory optimization function
        optimize_walkers: Optional walker optimization function
    """

    def __init__(
        self,
        ns_exe: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        initial_input_file: os.PathLike,
        n_gpu_parallel: int,
        sample_size: int = 100,
        resample_size: Optional[int] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        energy_uncty_weight: float = 1.0,
        force_uncty_weight: float = 1.0,
        optimize_traj: Optional[Callable] = None,
        optimize_walkers: Optional[Callable] = None,
        stub: bool = False,
    ):
        super().__init__()
        self.ns_exe = ns_exe
        self.remote_fs = remote_fs
        self.n_gpu_parallel = n_gpu_parallel
        self.remote_savepath = (
            Path(remote_savepath) if remote_savepath else None
        )
        self.sample_size = sample_size
        self.resample_size = resample_size
        self.loop_time = loop_time
        self.optimize_traj = optimize_traj
        self.optimize_walkers = optimize_walkers
        self.energy_uncty_weight = energy_uncty_weight
        self.force_uncty_weight = force_uncty_weight
        self.initial_input_file = initial_input_file
        self.stub = stub

        if self.resample_size and self.resample_size > self.sample_size:
            raise WorkflowError(
                f"Invalid resample configuration: resample_size ({self.resample_size}) "
                f"must be <= sample_size ({self.sample_size})"
            )

    def update(
        self, calcdir: os.PathLike, model_file: os.PathLike, **kwargs
    ) -> None:
        """
        Prepare sampler for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            model_file: Path to model file for uncertainty estimation
            **kwargs: Additional NS parameters
        """
        self.ns_dir = Path(calcdir) / "ns_sampler"
        self.results_dir = self.ns_dir / "results"
        os.makedirs(self.ns_dir, exist_ok=True)

        self.model_file = model_file
        it = str(calcdir).split("/")[-1]

        # In stub mode, we don't need to create the task
        if not self.stub:
            try:
                from jaxnest import input_parser
            except ImportError:
                raise DependencyError(
                    "jaxnest",
                    "NSTrajectorySampler requires the 'jaxnest' package. "
                    "Install it according to the guide on the JAXNEST gitlab repo\n"
                    "Or use stub=True for testing.",
                )

            # Verify ns_exe exists on remote
            if not self.remote_fs.exists(self.ns_exe):
                raise FileSystemError(
                    f"Nested sampling executable not found on remote system: {self.ns_exe}"
                )

            from remote_handler.core.cluster_handler import ClusterConfig

            cluster_config = ClusterConfig(
                jobfile_name="run_slurm.sh",
                n_processors=self.n_gpu_parallel,
                job_type_template="body_ns.sh.j2",
            )

            self.ns_task = SingleScriptTask(
                cluster_config=cluster_config,
                script_file=self.ns_exe,
                input_file=self.initial_input_file,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                logfile="ns.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                loop_time=self.loop_time,
                jobname=f"AL_NS_{it}",
            )

            # Prepare NS input file with model file path
            model_file_remote = self.ns_task.copy_file(model_file)

            with open(self.initial_input_file, "r") as f:
                input_kwargs = input_parser.input_str_to_dict(f.read())

            kwargs["pickle_file"] = str(model_file_remote)
            update_default_args(kwargs, input_kwargs)
            input_parser.write_ns_inputfile(
                self.initial_input_file, input_kwargs
            )

            # Allocate data (copies input file and writes jobfile)
            self.ns_task.allocate_data()

    def _create_stub_outputs(self) -> list[Atoms]:
        """
        Create mock NS trajectory outputs for testing.

        Generates fake NS trajectories with uncertainty information.

        Returns:
            List of Atoms objects with mock NS results
        """
        logger.info(f"Creating {self.sample_size} stub NS samples")

        n_runs = 4  # Mock 4 parallel runs
        samples_per_run = self.sample_size // n_runs

        all_ats = []
        for run_id in range(n_runs):
            for sample_idx in range(samples_per_run):
                # Create mock atomic structure
                symbols = ["Cu"] * 16 + ["O"] * 4
                positions = np.random.uniform(0, 8, (len(symbols), 3))

                atoms = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=[8, 8, 8],
                    pbc=True,
                )

                # Add NS-specific information
                atoms.info["run_id"] = run_id
                atoms.info["iter"] = np.random.randint(0, 1000)
                atoms.info["energy_uncert"] = np.random.uniform(0.01, 0.5)
                atoms.info["forces_uncert"] = np.random.uniform(0.1, 2.0)
                atoms.info["log_L"] = -100 + np.random.uniform(-50, 0)

                all_ats.append(atoms)

        # Apply resampling if requested
        if self.resample_size:
            all_ats = self._resample_configs(all_ats)

        return all_ats

    def _resample_configs(self, all_ats: list[Atoms]) -> list[Atoms]:
        """
        Resample configurations based on uncertainty scores.

        Args:
            all_ats: List of all sampled configurations

        Returns:
            Resampled list of configurations
        """
        funcs = np.array([at.info["forces_uncert"] for at in all_ats])
        euncs = np.array([at.info["energy_uncert"] for at in all_ats])

        # Normalize uncertainties
        funcs_norm = funcs / np.sum(funcs)
        euncs_norm = euncs / np.sum(euncs)

        # Compute weighted score
        score = (
            self.force_uncty_weight * funcs_norm
            + self.energy_uncty_weight * euncs_norm
        )
        score = score / np.sum(score)

        # Sample based on uncertainty scores
        indices = np.arange(len(all_ats))
        selected_inds = np.random.choice(
            indices, self.resample_size, replace=False, p=score
        )
        selected_inds.sort()

        resampled_ats = [all_ats[i] for i in selected_inds]

        logger.info(
            f"Resampled {len(resampled_ats)} configurations from {len(all_ats)}"
        )

        # Save resampling indices
        if hasattr(self, "results_dir"):
            os.makedirs(self.results_dir, exist_ok=True)
            np.savetxt(
                self.results_dir / "resampled_inds.csv",
                selected_inds,
                header="selected_indices",
            )

        return resampled_ats

    def get_sampled_configs(self) -> Path:
        """
        Run NS sampling and collect configurations.

        Returns:
            Path to XYZ file containing sampled configurations
        """
        if self.stub:
            # In stub mode, create mock outputs directly
            logger.info("Running in stub mode - creating mock NS samples")
            all_ats = self._create_stub_outputs()
        else:
            # Run actual NS
            self.ns_task.run()

            ns_output_data = self.results_dir / "output_data"
            if not os.path.exists(ns_output_data):
                raise WorkflowError(
                    f"Nested sampling did not complete successfully\n"
                    f"Missing output directory: {ns_output_data}"
                )

            # Load trajectories from NS runs
            from remote_handler.core.utils import grep

            tfiles = grep(str(ns_output_data), "traj")

            n_runs = len(tfiles)
            if n_runs == 0:
                raise WorkflowError(
                    f"Failed to collect trajectory files from nested sampling run\n"
                    f"Results directory: {self.results_dir}"
                )

            n_samples = len(ase.io.read(tfiles[0], index=":"))
            if n_samples == 0:
                raise WorkflowError(
                    f"Nested sampling trajectories are empty\n"
                    f"Check NS run in: {self.results_dir}"
                )

            # Calculate sampling interval
            sample_size_per_run = self.sample_size // n_runs
            every = n_samples // sample_size_per_run
            if sample_size_per_run > n_samples:
                logger.warning(
                    f"sample_size_per_run={sample_size_per_run} is larger than "
                    f"n_samples={n_samples}. Taking all samples."
                )
                every = 1

            # Optimize trajectories if requested
            if self.optimize_traj is not None:
                optimizer = self.optimize_traj(tfiles, self.model_file)
                optimizer.run()

            # Load and label trajectories
            all_ats = []
            for run_id, tfile in enumerate(tfiles):
                ats = ase.io.read(tfile, index=f"::{every}")

                for at in ats:
                    at.info["run_id"] = run_id

                all_ats.extend(ats)

            # Apply resampling if requested
            if self.resample_size is not None:
                all_ats = self._resample_configs(all_ats)

        logger.info(f"Collected {len(all_ats)} sampled configurations")

        # Write sampled configurations to file for ActiveLearningHandler
        os.makedirs(self.results_dir, exist_ok=True)
        output_file = self.results_dir / "sampled_configs.extxyz"
        ase.io.write(output_file, all_ats)
        logger.info(f"Wrote sampled configurations to {output_file}")

        return output_file


def _read_trajectory_frames(traj) -> list[Atoms]:
    """Read all frames from a TorchSimTrajectory, attaching custom properties.

    ``TorchSimTrajectory.get_atoms`` only returns basic structural data
    (positions, cell, numbers, pbc).  This helper additionally reads any
    custom arrays stored by ``TrajectoryReporter`` (e.g. from
    ``prop_calculators``) and attaches them to each ``Atoms`` object:

    * Scalar-per-frame arrays (like ``energy_uncertainty``) are placed in
      ``atoms.info[name]``.
    * Per-atom arrays whose first dimension matches the atom count (like
      ``forces_uncertainty`` with shape ``[n_atoms, 3]``) are placed in
      ``atoms.arrays[name]``.
    """
    n_frames = traj._file.root.data.positions.shape[0]
    frames = [traj.get_atoms(frame=i) for i in range(n_frames)]

    # State arrays written by write_state — skip these when attaching props
    _state_keys = {"positions", "cell", "masses", "atomic_numbers", "pbc"}
    prop_names = [k for k in traj.array_registry if k not in _state_keys]

    for name in prop_names:
        try:
            arr = traj.get_array(name)
        except Exception:
            continue

        for i, at in enumerate(frames):
            if i >= arr.shape[0]:
                break
            val = arr[i]
            if val.ndim == 0 or (val.ndim == 1 and val.shape[0] == 1):
                # Scalar → atoms.info
                at.info[name] = float(val)
            elif val.ndim >= 1 and val.shape[0] == len(at):
                # Per-atom array → atoms.arrays
                at.arrays[name] = np.ascontiguousarray(val)
            else:
                at.info[name] = val

    return frames


class TorchSimSampler(Sampler):
    """
    Sampler using torch_sim parallel tempering with MACE models.

    Runs parallel tempering MD (NVT or NPT ensemble) to explore configuration
    space at multiple temperatures. Supports stub mode for testing without
    actual GPU execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        cluster_config: ClusterConfig with partition, conda env, template settings
        sampling_script: Path to torch_sim_sampling.py or compatible script
        founder_path: Path to founder structure file (ASE-readable)
        ensemble: Ensemble type, "nvt" or "npt"
        temperatures: List of replica temperatures in K
        target_temperature: Target temperature in K (required for NVT ensemble)
        model_paths: List of MACE model identifiers. Use ["foundation"] for
            the MACE-MP foundation model. A single-element list runs a
            standard simulation; multiple entries activate the
            MaceEnsembleModel (shared-body ensemble with uncertainty).
        dt: Integration timestep in ps (default: 0.001)
        gamma: Friction coefficient for Langevin thermostat (optional)
        alpha: Friction coefficient for particle Langevin (NPT, optional)
        cell_alpha: Friction coefficient for cell Langevin (NPT, optional)
        n_steps_npt: Number of NPT equilibration steps (default: 0)
        n_steps_pt: Number of production PT steps (default: 100)
        seed: Random seed (default: 123)
        device: Torch device string (default: "cuda")
        dtype: Torch dtype string (default: "float32")
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        selector: Optional ConfigSelector strategy for choosing which
            configurations from the PT trajectories to return. If None,
            the script's default output (final frame per replica) is used.
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        sampling_script: os.PathLike,
        founder_path: os.PathLike,
        ensemble: str,
        temperatures: list[float],
        target_temperature: Optional[float] = None,
        model_paths: Optional[list[str]] = None,
        dt: float = 0.001,
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
        cell_alpha: Optional[float] = None,
        n_steps_npt: int = 0,
        n_steps_pt: int = 100,
        seed: int = 123,
        device: str = "cuda",
        dtype: str = "float32",
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        selector: Optional[ConfigSelector] = None,
    ):
        super().__init__()
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.sampling_script = sampling_script
        self.founder_path = founder_path
        self.ensemble = ensemble
        self.temperatures = temperatures
        self.target_temperature = target_temperature
        self.model_paths = (
            model_paths if model_paths is not None else ["foundation"]
        )
        self.dt = dt
        self.gamma = gamma
        self.alpha = alpha
        self.cell_alpha = cell_alpha
        self.n_steps_npt = n_steps_npt
        self.n_steps_pt = n_steps_pt
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub
        self.selector = selector

        # UncertaintySelector requires an ensemble model (multiple model_paths)
        # to produce the energy_uncertainty / forces_uncertainty it relies on.
        if selector is not None and len(self.model_paths) < 2:
            from remote_handler.workflows.selectors import (
                PipelineSelector,
                UncertaintySelector,
            )

            needs_ensemble = isinstance(selector, UncertaintySelector)
            if isinstance(selector, PipelineSelector):
                needs_ensemble = any(
                    isinstance(s, UncertaintySelector)
                    for s in selector.selectors
                )
            if needs_ensemble:
                raise ValueError(
                    f"UncertaintySelector requires an ensemble model "
                    f"(model_paths with >= 2 entries) to produce uncertainty "
                    f"estimates. Got model_paths={self.model_paths!r}."
                )

    def update(
        self, calcdir: Path, sampler_input: list[Path], **kwargs
    ) -> None:
        """
        Prepare sampler for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: Path or list of paths to model files. The filenames
                are used to populate the ``model_paths`` field in the script
                input JSON. When the AL handler provides finetuned model files
                (from training), those override the initial ``model_paths``
                setting (e.g. ``["foundation"]``).
            **kwargs: Additional parameters
        """
        self.sample_dir = calcdir / "sample"
        self.results_dir = self.sample_dir / "results"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.model_files = (
            sampler_input
            if isinstance(sampler_input, list)
            else [sampler_input]
        )

        if not self.stub:
            # Build model_paths for the script input.
            # self.model_paths is the *initial* setting (e.g. ["foundation"]).
            # When the AL handler provides finetuned model files (from
            # training), those take precedence over the initial setting.
            script_model_paths = [Path(p).name for p in self.model_files]
            is_foundation = script_model_paths == ["foundation"]

            script_input = {
                "founder": Path(self.founder_path).name,
                "model_paths": script_model_paths,
                "ensemble": self.ensemble,
                "temperatures": self.temperatures,
                "target_temperature": self.target_temperature,
                "dt": self.dt,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "cell_alpha": self.cell_alpha,
                "n_steps_npt": self.n_steps_npt,
                "n_steps_pt": self.n_steps_pt,
                "seed": self.seed,
                "device": self.device,
                "dtype": self.dtype,
            }

            # Validate input dict against the script's expected schema
            TorchSimSamplingInput.from_dict(script_input)

            self.script_input_file = self.results_dir / "input_torchsim.json"
            with open(self.script_input_file, "w") as f:
                f.write(json.dumps(script_input, indent=2))

            self.sampling_task = SingleScriptTask(
                local_savepath=self.results_dir,
                input_file=self.script_input_file,
                cluster_config=self.cluster_config,
                remote_fs=self.remote_fs,
                script_file=self.sampling_script,
                remote_savepath=self.remote_savepath,
                jobfile_name="run_torchsim.sh",
                jobname="torchsim_sample",
                logfile="torchsim_sample.log",
                n_processors=1,
                loop_time=self.loop_time,
                collect_only=False,
            )

            self.sampling_task.allocate_data()

            # Copy founder structure to remote
            self.sampling_task.copy_file(self.founder_path)

            # Copy model files to remote (unless using foundation model)
            if not is_foundation:
                for p in self.model_files:
                    self.sampling_task.copy_file(p)

    def _create_stub_outputs(self) -> list[Atoms]:
        """
        Create mock torch_sim sampling outputs for testing.

        Returns:
            List of Atoms objects with mock PT results
        """
        n_replicas = len(self.temperatures)
        logger.info(f"Creating {n_replicas} stub torch_sim samples")

        mock_samples = []
        for i in range(n_replicas):
            symbols = ["Si"] * 32 + ["C"] * 32 + ["Ti"] * 16
            positions = np.random.uniform(0, 12, (len(symbols), 3))

            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[12, 12, 24],
                pbc=True,
            )

            atoms.info["replica"] = i
            atoms.info["ensemble"] = self.ensemble
            atoms.info["temperature"] = self.temperatures[i]
            mock_samples.append(atoms)

        return mock_samples

    def get_sampled_configs(self) -> Path:
        """
        Run torch_sim parallel tempering and collect configurations.

        If a ``selector`` was provided, trajectory h5md files are passed
        to the selector which picks configurations according to its
        strategy. Otherwise the script's default output (final frame per
        replica) is returned.

        Returns:
            Path to extxyz file containing sampled configurations
        """
        if self.stub:
            logger.info(
                "Running in stub mode - creating mock torch_sim samples"
            )
            sampled_configs = self._create_stub_outputs()

            output_dir = self.results_dir / "out"
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / "samples.extxyz"
            ase.io.write(output_path, sampled_configs)
        else:
            self.sampling_task.submit()
            self.sampling_task.collect_results()

            if self.selector is not None:
                # Read all trajectory frames and pass to selector
                results_path = Path(self.sampling_task.local_savepath)
                traj_paths = sorted(results_path.glob("trajectory_*.h5md"))
                if not traj_paths:
                    logger.warning(
                        "No trajectory_*.h5md files found in %s, "
                        "falling back to script output",
                        results_path,
                    )
                    output_path = results_path / "out/samples.extxyz"
                else:
                    from torch_sim.trajectory import TorchSimTrajectory

                    grouped_frames: list[list[Atoms]] = []
                    for traj_path in traj_paths:
                        traj = TorchSimTrajectory(str(traj_path), mode="r")
                        frames = _read_trajectory_frames(traj)
                        traj.close()
                        grouped_frames.append(frames)

                    selected = self.selector.select(grouped_frames)
                    output_dir = results_path / "out"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = output_dir / "samples.extxyz"
                    ase.io.write(output_path, selected)
            else:
                output_path = (
                    Path(self.sampling_task.local_savepath)
                    / "out/samples.extxyz"
                )

        logger.info(
            f"Torch_sim sampled configurations written to {output_path}"
        )
        return output_path


class SSPDSampler(Sampler):
    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        sampling_script: os.PathLike,  # SSPD_sampling.py
        data_path: os.PathLike,  # path to .xyz for starting configuration
        cluster_config: ClusterConfig,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        selector: Optional[ConfigSelector] = None,
    ):
        """
        Args:
            remote_fs:FileSystemHandlerBase` for the machine on which the sampling
                should be run
            sampling_script: Path to SSPD_sampling.py or another compatible sampling
                script
            remote_savepath: Path in which files for the remote calculations are stored
            loop_time: interval (in seconds) at which to poll if the sampling slurm job
                has finished
        """
        super().__init__()
        self.remote_fs = remote_fs
        self.sampling_script = sampling_script
        self.data_path = data_path
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.cluster_config = cluster_config
        self.selector = selector

    def update(
        self, calcdir: Path, sampler_input: list[Path], **kwargs
    ) -> None:
        self.sample_dir = calcdir / "sample"
        self.results_dir = self.sample_dir / "results"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.model_files = sampler_input
        # suggestion: go over everything in kwargs and just write it in there, 
        # implement defaults for each omission
        # see the nested sampling sampler above
        script_input = {
            "input_struct_path": Path(self.data_path).name,
            "threshold": kwargs["threshold"],
            "inv_beta": kwargs["inv_beta"],
            "max_acceptance": 1.0,
            "min_acceptance": 0.5,
            "initial_timestep": kwargs["initial_timestep"],
            "num_iterations": kwargs["num_iterations"],
            "sampling_threshold": kwargs["sampling_threshold"],
            "sampling_frequency": kwargs["sampling_frequency"],
            "N": kwargs["N"],
            "model_file": Path(self.model_files).name,
            "seed": np.random.randint(1e6),
            "num_iterations_OL": kwargs.get("num_iterations_OL"),
            "hessian_method": kwargs["hessian_method"],
        }

        self.script_input_file = (
            self.results_dir / "input_sample.json"
        )  # script input file
        with open(self.script_input_file, "w") as f:
            f.write(json.dumps(script_input))

        self.sampling_task = SingleScriptTask(
            local_savepath=self.results_dir,
            input_file=self.script_input_file,
            cluster_config=self.cluster_config,
            remote_fs=self.remote_fs,
            script_file=self.sampling_script,
            remote_savepath=self.remote_savepath,
            jobfile_name="run_sample.sh",
            jobname="sample",
            logfile="sample.log",
            n_processors=1,
            loop_time=self.loop_time,
            collect_only=False,
        )

        self.sampling_task.allocate_data()  # moves script and input files

        self.sampling_task.copy_file(self.data_path)
        self.sampling_task.copy_file(self.model_files)


    def get_sampled_configs(self) -> Path:
        self.sampling_task.submit()
        self.sampling_task.collect_results()

        if self.selector is not None:
            # Read all trajectory frames and pass to selector
            results_path = Path(self.sampling_task.local_savepath)
            results_path = results_path / "out/"
            traj_paths = sorted(results_path.glob("*.xyz"))
            if not traj_paths:
                logger.warning(
                    "No *.xyz files found in %s, "
                    "falling back to script output",
                    results_path,
                )
                output_path = results_path / "out/samples.extxyz"
            else:
                grouped_frames: list[list[Atoms]] = []
                for traj_path in traj_paths:
                    traj = ase.io.read(str(traj_path), index=":")
                    grouped_frames.append(traj)

                selected = self.selector.select(grouped_frames)
                output_dir = results_path / "out"
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir / "samples.extxyz"
                ase.io.write(output_path, selected)
        else:
            output_path = (
                Path(self.sampling_task.local_savepath)
                / "out/samples.extxyz"
            )

        logger.info(
            f"SSPD sampled configurations written to {output_path}"
        )
        return output_path
"""
Samplers for Active Learning Workflows.

Samplers generate new configurations to be evaluated by exploring configuration space.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import ase.io
import numpy as np
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.exceptions import WorkflowError
from remote_handler.workflows.selectors import ConfigSelector

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


class LammpsSampler(Sampler):
    """
    MD sampler using LAMMPS with uncertainty-based sampling.

    Runs LAMMPS molecular dynamics and selects configurations based on model uncertainty.
    Supports stub mode for testing without actual LAMMPS execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        template_path: Path to jinja2 template for lammps.in file
        data_path: Path to lammps.data file with starting configuration
        element_mapping: Dict mapping atom type numbers to element symbols
        num_samples: Number of configurations to sample per generation
        uncertainty_cutoff_radius: Cutoff radius (Angstrom) for local uncertainty
        remote_lammps_executable_path: Path to LAMMPS executable on remote machine
        sampling_script: Remote payload; defaults to the shipped
            ``lammps/lammps_sampling.py``.
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        template_path: os.PathLike,
        data_path: os.PathLike,
        element_mapping: dict,
        num_samples: int,
        uncertainty_cutoff_radius: float,
        remote_lammps_executable_path: os.PathLike,
        sampling_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
    ):
        super().__init__()
        if sampling_script is None:
            sampling_script = _SCRIPTS_DIR / "lammps" / "lammps_sampling.py"
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
            if not self.sampling_task.finish_check():
                raise WorkflowError(
                    "LAMMPS sampling did not finish successfully "
                    "(no is_finished.info marker). Check the job logs in "
                    f"{self.sampling_task.calcdir}."
                )
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
        founder_path: Path to founder structure file (ASE-readable)
        ensemble: Ensemble type, "nvt" or "npt"
        temperatures: List of replica temperatures in K
        sampling_script: Remote payload; defaults to the shipped
            ``torch_sim/torch_sim_sampling.py``.
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
        founder_path: os.PathLike,
        ensemble: str,
        temperatures: list[float],
        sampling_script: Optional[os.PathLike] = None,
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
        if sampling_script is None:
            sampling_script = (
                _SCRIPTS_DIR / "torch_sim" / "torch_sim_sampling.py"
            )
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
            if not self.sampling_task.finish_check():
                raise WorkflowError(
                    "torch_sim sampling did not finish successfully "
                    "(no is_finished.info marker). Check the job logs in "
                    f"{self.sampling_task.calcdir}."
                )
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
        data_path: os.PathLike,  # path to .xyz for starting configuration
        cluster_config: ClusterConfig,
        sampling_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        selector: Optional[ConfigSelector] = None,
        torch_sim: bool = False,
    ):
        """
        Args:
            remote_fs:FileSystemHandlerBase` for the machine on which the sampling
                should be run
            sampling_script: Remote payload; defaults to the shipped
                ``sspd/SSPD_torch_sim_sampling.py`` when ``torch_sim`` is set,
                else ``sspd/SSPD_sampling.py``.
            remote_savepath: Path in which files for the remote calculations are stored
            loop_time: interval (in seconds) at which to poll if the sampling slurm job
                has finished
        """
        super().__init__()
        if sampling_script is None:
            sampling_script = (
                _SCRIPTS_DIR
                / "sspd"
                / (
                    "SSPD_torch_sim_sampling.py"
                    if torch_sim
                    else "SSPD_sampling.py"
                )
            )
        self.remote_fs = remote_fs
        self.sampling_script = sampling_script
        self.data_path = data_path
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.cluster_config = cluster_config
        self.selector = selector
        self.torch_sim = torch_sim

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
        if "struct" in kwargs.keys():
            self.data_path = kwargs["struct"]
        script_input = {
            "input_struct_path": Path(self.data_path).name,
            "threshold": kwargs.get("threshold"),
            "inv_beta": kwargs.get("inv_beta", 0.05),
            "max_acceptance": 1.0,
            "min_acceptance": 0.5,
            "initial_timestep": kwargs.get("initial_timestep", 0.01),
            "num_iterations": kwargs.get("num_iterations", 10000),
            "sampling_threshold": kwargs.get("sampling_threshold", 0),
            "sampling_frequency": kwargs.get("sampling_frequency", 50),
            "N": kwargs.get("N", 100),
            "model_file": Path(self.model_files).name,
            "seed": np.random.randint(1e6),
            "num_iterations_OL": kwargs.get("num_iterations_OL"),
            "hessian_method": kwargs.get("hessian_method", "hessian"),
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
        if not self.sampling_task.finish_check():
            raise WorkflowError(
                "SSPD sampling did not finish successfully "
                "(no is_finished.info marker). Check the job logs in "
                f"{self.sampling_task.calcdir}."
            )
        self.sampling_task.collect_results()
        # TODO: check possibilities with the sampler
        # would be nice to separate the "reader" functions. Or simply discontinue the support for the non torch-sim version
        results_path = Path(self.sampling_task.local_savepath) / "out"
        if self.torch_sim:
            # there is also the option to write the trajectory in the script, but the h5 format is nicer
            torch_sim_alist = []
            from torch_sim.trajectory import TorchSimTrajectory

            traj_paths = sorted(results_path.glob("SSPD_trajectory_*.h5md"))
            for tp in traj_paths:
                with TorchSimTrajectory(tp) as traj:
                    for i in range(len(traj)):
                        atoms = traj.get_atoms(frame=i)
                        # TODO: figure out why the following is required
                        atoms.cell.array = atoms.cell.array.T
                        torch_sim_alist.append(atoms)

        if self.selector is not None:
            grouped_frames = None
            if not self.torch_sim:
                # Read all trajectory frames and pass to selector
                traj_paths = sorted(results_path.glob("*.xyz"))
                if traj_paths:
                    grouped_frames: list[list[Atoms]] = []
                    for traj_path in traj_paths:
                        traj = ase.io.read(str(traj_path), index=":")
                        grouped_frames.append(traj)

            else:
                grouped_frames = [torch_sim_alist]
            if grouped_frames:
                selected = self.selector.select(grouped_frames)
                os.makedirs(results_path, exist_ok=True)
                output_path = results_path / "samples.extxyz"
                ase.io.write(output_path, selected)
            else:
                logger.warning(
                    "No output trajectory files found in %s, "
                    "falling back to script output",
                    results_path,
                )
                output_path = results_path / "samples.extxyz"

        else:
            output_path = results_path / "samples.extxyz"
            if self.torch_sim:
                ase.io.write(output_path, torch_sim_alist)

        logger.info(f"SSPD sampled configurations written to {output_path}")
        return output_path


class AmorphousEmbeddingSampler(Sampler):
    """
    Extracts regions of interest (ROIs) from an uncertainty-tagged
    trajectory, carves/orients/cleans up an amorphous-matrix box around
    each, and anneals every box with a multi-stage LAMMPS deck.

    Two remote stages, each a separate ``SingleScriptTask`` (potentially on
    different machines/envs -- see ``anneal_cluster_config``/
    ``anneal_remote_fs``): extraction
    (``scripts/embedding/extract.py``, needs the ``embedding`` extra when
    ``topology_mode`` is ``"surface"``/``"interface"``) and, one task per
    extracted box, annealing (``scripts/lammps/embedding_anneal.py``, needs a
    LAMMPS build).

    Args:
        remote_fs: FileSystemHandlerBase for the extraction stage
        cluster_config: ClusterConfig for the extraction stage
        trajectory_path: Path to the uncertainty-tagged trajectory (must
            carry a per-atom ``uncertainty`` array)
        threshold: Minimum per-atom uncertainty for an atom to seed a ROI
        cluster_cutoff: Distance (Angstrom) grouping high-uncertainty atoms
            into the same ROI
        box_x: Half-width of the carved box along x
        box_y: Half-width of the carved box along y
        box_z_below: Half-width of the carved box below the ROI along z
        protect_radius: Radius around each ROI that cleanup never modifies
        lammps_path: Path to the LAMMPS executable on the anneal machine
        pair_style: LAMMPS ``pair_style`` line contents
        pair_coeff_path: Path to the potential file or directory (e.g. a
            GRACE model, normally saved as a directory)
        species: Element symbols in LAMMPS type order
        masses: Atomic masses aligned with ``species``
        anneal_stages: One dict per anneal stage, run in order; each stage
            is frozen while every other stage anneals (the ROI stays frozen
            throughout every stage). A single-entry list is a single-stage
            anneal; two or more chain stages, each annealed while the
            others are held still. Species must not repeat across stages.
            Each dict has keys:

            * ``species`` (list[str]): this stage's species group.
            * ``temperatures`` (list[float]): temperature waypoints (K),
              e.g. ``[300, 600, 300]`` for a ramp-up-then-down, or any
              longer sequence for more segments; a flat ``[300, 300]`` for
              one constant-temperature segment. At least 2 entries.
            * ``run_steps`` (list[int]): steps for each segment between
              consecutive waypoints -- ``len(run_steps) ==
              len(temperatures) - 1``.
            * ``pressure_xy`` (float, optional): target in-plane pressure
              (bar) for NPT dilation for this stage; omit/``None`` leaves
              x/y uncoupled.
            * ``pressure_z`` (float, optional): target out-of-plane
              pressure (bar); omit/``None`` leaves z uncoupled. A stage
              with both pressures unset runs NVT instead of NPT.

            Example, a two-stage metal-then-solvent anneal::

                anneal_stages=[
                    {
                        "species": ["Cu"],
                        "temperatures": [300, 600, 300],
                        "run_steps": [50000, 100000],
                        "pressure_xy": 1.0,
                    },
                    {
                        "species": ["H", "O"],
                        "temperatures": [300, 300],
                        "run_steps": [50000],
                        "pressure_z": 1.0,
                    },
                ]
        box_z_above: Half-width above the ROI along z; defaults to
            ``box_z_below`` (symmetric box)
        buffer: Extra padding added to each carved-cell dimension
        topology_mode: ``"bulk"``, ``"surface"``, or ``"interface"`` --
            see ``MatrixEmbeddingInput``/``EmbeddingAnnealInput``
        remove_species: Phase stripped (on an internal OVITO copy) before
            meshing for the interface normal. Required for ``"interface"``.
        phase_species: Species defining the condensed phase, used only by
            the normal-detection fallback heuristic
        cleanup_mode: ``"none"``, ``"molecule"``, or ``"stoichiometry"``
        solvent: For ``cleanup_mode="molecule"``: either an ASE ``Atoms``
            solvent template (staged and sent as a file) or an
            ``ase.build.molecule`` g2 label string (e.g. ``"H2O"``)
        stoichiometry: Target ratio dict for ``cleanup_mode="stoichiometry"``
        specorder: Species order for ``lammps-data`` output
        smoothing_level: OVITO ``ConstructSurfaceMeshModifier`` smoothing
        mesh_radius: OVITO probe-sphere radius; ``None`` uses OVITO's default
        normal_patch_radius: Approximate geodesic radius over which
            edge-connected surface-mesh facet normals are area-averaged
            after anchoring at the closest surface point. This smooths local
            atomic-scale roughness without mixing in a disconnected nearby
            surface. ``None`` (default) uses ``protect_radius``.
        anneal_cutoff: Radius of the anneal's frozen ``fixSphere``; defaults
            to ``protect_radius``
        anneal_template_path: Jinja2 LAMMPS deck template; defaults to the
            shipped ``lammps/templates/embedding_anneal.in.j2``
        roi_repel_margin: See ``EmbeddingAnnealInput.roi_repel_margin`` --
            gap subtracted from ``anneal_cutoff`` for the ROI's repulsive
            wall. A single setting shared by every stage's mobile atoms
            (unlike ``z_wall_margin`` below, which is per-stage): the wall
            protects the frozen core itself, so it stays uniformly active
            regardless of which species is currently mobile. Disable the
            wall with ``roi_repel_epsilon=0.0``.
        roi_repel_epsilon: See ``EmbeddingAnnealInput.roi_repel_epsilon``.
        roi_repel_wall_cutoff: See
            ``EmbeddingAnnealInput.roi_repel_wall_cutoff``.
        z_wall_margin: See ``EmbeddingAnnealInput.z_wall_margin`` -- deck-wide
            default padding around each stage's mobile atoms' initial
            z-extent, used by any stage in ``anneal_stages`` that doesn't
            set its own ``z_wall_margin``/``z_wall_epsilon``/
            ``z_wall_cutoff`` keys (e.g. to tune the wall differently for a
            rigid metal-lattice stage vs. a mobile solvent stage). Disable
            deck-wide with ``z_wall_epsilon=0.0``.
        z_wall_epsilon: See ``EmbeddingAnnealInput.z_wall_epsilon``. ``None``
            (default) auto-decides from ``topology_mode``: ``10.0`` for
            ``"surface"``/``"interface"`` (where a free surface means there
            may be vacuum/solvent headspace along z for atoms to drift
            into), ``0.0`` (disabled) for ``"bulk"`` (fully periodic in z --
            there is no vacuum to escape into, so the wall would be inert
            overhead). Pass an explicit value to override either way.
        z_wall_cutoff: See ``EmbeddingAnnealInput.z_wall_cutoff``.
        timestep: MD timestep (ps)
        seed: Random seed for velocity initialization
        dump_every: Dump-output frequency (steps)
        lammps_flags: Extra ``lmp`` command-line flags (e.g. Kokkos/GPU
            acceleration) passed verbatim to the anneal; depends on the
            LAMMPS build and force field, empty runs a plain ``lmp``
        extract_script: Remote payload; defaults to the shipped
            ``embedding/extract.py``
        anneal_script: Remote payload; defaults to the shipped
            ``lammps/embedding_anneal.py``
        anneal_cluster_config: ClusterConfig for the anneal stage; defaults
            to ``cluster_config``
        anneal_remote_fs: FileSystemHandlerBase for the anneal stage;
            defaults to ``remote_fs``
        remote_savepath: Path on the extraction machine for calculation files
        anneal_remote_savepath: Path on the anneal machine for calculation
            files; defaults to ``remote_savepath``
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, skip remote execution and produce mock outputs
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        trajectory_path: os.PathLike,
        threshold: float,
        cluster_cutoff: float,
        box_x: float,
        box_y: float,
        box_z_below: float,
        protect_radius: float,
        lammps_path: os.PathLike,
        pair_style: str,
        pair_coeff_path: os.PathLike,
        species: list[str],
        masses: list[float],
        anneal_stages: list[dict],
        box_z_above: Optional[float] = None,
        buffer: float = 2.0,
        topology_mode: str = "bulk",
        remove_species: Optional[list[str]] = None,
        phase_species: Optional[list[str]] = None,
        cleanup_mode: str = "none",
        solvent: Optional[Union[Atoms, str]] = None,
        stoichiometry: Optional[dict[str, int]] = None,
        specorder: Optional[list[str]] = None,
        smoothing_level: int = 8,
        mesh_radius: Optional[float] = None,
        normal_patch_radius: Optional[float] = None,
        anneal_cutoff: Optional[float] = None,
        anneal_template_path: Optional[os.PathLike] = None,
        roi_repel_margin: float = 0.5,
        roi_repel_epsilon: float = 10.0,
        roi_repel_wall_cutoff: float = 1.0,
        z_wall_margin: float = 1.0,
        z_wall_epsilon: Optional[float] = None,
        z_wall_cutoff: float = 1.0,
        timestep: float = 0.0005,
        seed: int = 42,
        dump_every: int = 1000,
        lammps_flags: str = "",
        extract_script: Optional[os.PathLike] = None,
        anneal_script: Optional[os.PathLike] = None,
        anneal_cluster_config: Optional[ClusterConfig] = None,
        anneal_remote_fs: Optional[FileSystemHandlerBase] = None,
        remote_savepath: Optional[os.PathLike] = None,
        anneal_remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
    ):
        super().__init__()
        if extract_script is None:
            extract_script = _SCRIPTS_DIR / "embedding" / "extract.py"
        if anneal_script is None:
            anneal_script = _SCRIPTS_DIR / "lammps" / "embedding_anneal.py"
        if anneal_template_path is None:
            anneal_template_path = (
                _SCRIPTS_DIR
                / "lammps"
                / "templates"
                / "embedding_anneal.in.j2"
            )

        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.anneal_cluster_config = anneal_cluster_config or cluster_config
        self.anneal_remote_fs = anneal_remote_fs or remote_fs
        self.extract_script = extract_script
        self.anneal_script = anneal_script
        self.anneal_template_path = anneal_template_path

        self.trajectory_path = trajectory_path
        self.threshold = threshold
        self.cluster_cutoff = cluster_cutoff
        self.box_x = box_x
        self.box_y = box_y
        self.box_z_below = box_z_below
        self.box_z_above = box_z_above
        self.buffer = buffer
        self.protect_radius = protect_radius
        self.topology_mode = topology_mode
        self.remove_species = remove_species
        self.phase_species = phase_species
        self.cleanup_mode = cleanup_mode
        self.solvent = solvent
        self.stoichiometry = stoichiometry
        self.specorder = specorder
        self.smoothing_level = smoothing_level
        self.mesh_radius = mesh_radius
        self.normal_patch_radius = normal_patch_radius

        self.lammps_path = lammps_path
        self.pair_style = pair_style
        self.pair_coeff_path = pair_coeff_path
        self.species = species
        self.masses = masses
        self.anneal_stages = anneal_stages
        self.anneal_cutoff = (
            anneal_cutoff if anneal_cutoff is not None else protect_radius
        )
        self.roi_repel_margin = roi_repel_margin
        self.roi_repel_epsilon = roi_repel_epsilon
        self.roi_repel_wall_cutoff = roi_repel_wall_cutoff
        self.z_wall_margin = z_wall_margin
        self.z_wall_epsilon = (
            z_wall_epsilon
            if z_wall_epsilon is not None
            else (10.0 if topology_mode in ("surface", "interface") else 0.0)
        )
        self.z_wall_cutoff = z_wall_cutoff
        self.timestep = timestep
        self.seed = seed
        self.dump_every = dump_every
        self.lammps_flags = lammps_flags

        self.remote_savepath = remote_savepath
        self.anneal_remote_savepath = anneal_remote_savepath or remote_savepath
        self.loop_time = loop_time
        self.stub = stub

    def update(self, calcdir: Path, sampler_input=None, **kwargs) -> None:
        """
        Prepare the extraction task for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: Unused (this sampler reads directly from
                ``trajectory_path``, which does not change per iteration)
            **kwargs: Additional parameters
        """
        self.sample_dir = calcdir / "sample"
        self.extract_dir = self.sample_dir / "extract"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)

        if self.stub:
            return

        solvent_label = None
        solvent_file = None
        if self.cleanup_mode == "molecule":
            if isinstance(self.solvent, Atoms):
                solvent_file = self.extract_dir / "solvent_template.extxyz"
                ase.io.write(solvent_file, self.solvent, format="extxyz")
            elif isinstance(self.solvent, str):
                solvent_label = self.solvent
            else:
                raise WorkflowError(
                    "cleanup_mode='molecule' requires 'solvent' to be an "
                    "ase.Atoms object or an ase.build.molecule label "
                    f"string, got {type(self.solvent)}"
                )

        script_input = {
            "trajectory": Path(self.trajectory_path).name,
            "threshold": self.threshold,
            "cluster_cutoff": self.cluster_cutoff,
            "box_x": self.box_x,
            "box_y": self.box_y,
            "box_z_below": self.box_z_below,
            "box_z_above": self.box_z_above,
            "buffer": self.buffer,
            "protect_radius": self.protect_radius,
            "topology_mode": self.topology_mode,
            "remove_species": self.remove_species,
            "phase_species": self.phase_species,
            "cleanup_mode": self.cleanup_mode,
            "solvent_label": solvent_label,
            "solvent_file": (
                solvent_file.name if solvent_file is not None else None
            ),
            "stoichiometry": self.stoichiometry,
            # The anneal deck fixes LAMMPS type order to ``species`` (mass /
            # pair_coeff / Z_of_type lines), so every carved box must be
            # written with that same full ordering. Without an explicit
            # specorder ASE numbers types per-box from the symbols it happens
            # to contain, which breaks boxes missing a species. Default to
            # ``species`` unless the caller overrides.
            "specorder": (
                self.specorder if self.specorder is not None else self.species
            ),
            "smoothing_level": self.smoothing_level,
            "mesh_radius": self.mesh_radius,
            "normal_patch_radius": self.normal_patch_radius,
        }

        self.extract_input_file = self.extract_dir / "input_embedding.json"
        with open(self.extract_input_file, "w") as f:
            f.write(json.dumps(script_input, indent=2))

        self.extract_task = SingleScriptTask(
            local_savepath=self.extract_dir,
            input_file=self.extract_input_file,
            cluster_config=self.cluster_config,
            remote_fs=self.remote_fs,
            script_file=self.extract_script,
            remote_savepath=self.remote_savepath,
            jobfile_name="run_embed_extract.sh",
            jobname="embed_extract",
            logfile="embed_extract.log",
            n_processors=1,
            loop_time=self.loop_time,
            collect_only=False,
        )

        self.extract_task.allocate_data()
        self.extract_task.copy_file(self.trajectory_path)
        if solvent_file is not None:
            self.extract_task.copy_file(solvent_file)

    def _create_stub_outputs(self, num_boxes: int = 3) -> list[Atoms]:
        """
        Create mock embedded/annealed outputs for testing.

        Args:
            num_boxes: Number of mock boxes to generate

        Returns:
            List of Atoms objects mimicking annealed embedding boxes
        """
        logger.info(f"Creating {num_boxes} stub amorphous embedding boxes")

        mock_boxes = []
        for i in range(num_boxes):
            symbols = self.species * 4
            positions = np.random.uniform(0, 2 * self.box_x, (len(symbols), 3))
            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[
                    2 * self.box_x,
                    2 * self.box_y,
                    self.box_z_below + (self.box_z_above or self.box_z_below),
                ],
                pbc=True,
            )
            atoms.info["box_index"] = i
            atoms.info["topology_mode"] = self.topology_mode
            mock_boxes.append(atoms)
        return mock_boxes

    def get_sampled_configs(self) -> Path:
        """
        Run extraction, then anneal every extracted box.

        Returns:
            Path to extxyz file containing the annealed configurations
        """
        if self.stub:
            logger.info(
                "Running in stub mode - creating mock amorphous embedding "
                "boxes"
            )
            sampled_configs = self._create_stub_outputs()
            output_dir = self.sample_dir / "out"
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / "samples.extxyz"
            ase.io.write(output_path, sampled_configs)
            return output_path

        self.extract_task.submit()
        if not self.extract_task.finish_check():
            raise WorkflowError(
                "Amorphous matrix embedding extraction did not finish "
                "successfully (no is_finished.info marker). Check the job "
                f"logs in {self.extract_task.calcdir}."
            )
        self.extract_task.collect_results()

        extract_out = Path(self.extract_task.local_savepath) / "out"
        manifest = json.loads((extract_out / "manifest.json").read_text())

        if not manifest:
            logger.warning(
                "Amorphous matrix embedding extraction produced no regions "
                "of interest (no atom exceeded threshold=%s). No boxes will "
                "be annealed.",
                self.threshold,
            )

        anneal_dir = self.sample_dir / "anneal"
        os.makedirs(anneal_dir, exist_ok=True)

        anneal_tasks = []
        for entry in manifest:
            box_dir = anneal_dir / f"box_{entry['index']}"
            os.makedirs(box_dir, exist_ok=True)

            anneal_input = {
                "data_file": Path(entry["data_file"]).name,
                "lammps_path": Path(self.lammps_path).as_posix(),
                "template_file": Path(self.anneal_template_path).name,
                "cutoff": self.anneal_cutoff,
                "roi_x": entry["roi_position"][0],
                "roi_y": entry["roi_position"][1],
                "roi_z": entry["roi_position"][2],
                "pair_style": self.pair_style,
                "pair_coeff_path": Path(self.pair_coeff_path).name,
                "species": self.species,
                "masses": self.masses,
                "stages": self.anneal_stages,
                "roi_repel_margin": self.roi_repel_margin,
                "roi_repel_epsilon": self.roi_repel_epsilon,
                "roi_repel_wall_cutoff": self.roi_repel_wall_cutoff,
                "z_wall_margin": self.z_wall_margin,
                "z_wall_epsilon": self.z_wall_epsilon,
                "z_wall_cutoff": self.z_wall_cutoff,
                "timestep": self.timestep,
                "seed": self.seed,
                "dump_every": self.dump_every,
                "lammps_flags": self.lammps_flags,
            }
            anneal_input_file = box_dir / "input_anneal.json"
            with open(anneal_input_file, "w") as f:
                f.write(json.dumps(anneal_input, indent=2))

            task = SingleScriptTask(
                local_savepath=box_dir,
                input_file=anneal_input_file,
                cluster_config=self.anneal_cluster_config,
                remote_fs=self.anneal_remote_fs,
                script_file=self.anneal_script,
                remote_savepath=self.anneal_remote_savepath,
                jobfile_name="run_embed_anneal.sh",
                jobname=f"embed_anneal_{entry['index']}",
                logfile="embed_anneal.log",
                n_processors=1,
                loop_time=self.loop_time,
                collect_only=False,
            )
            task.allocate_data()
            task.copy_file(extract_out / entry["data_file"])
            task.copy_file(self.anneal_template_path)
            # copy_path (not copy_file): potentials aren't always a single
            # file -- e.g. a GRACE model is normally saved as a directory.
            task.copy_path(self.pair_coeff_path)
            anneal_tasks.append((entry["index"], task))

        annealed = []
        for index, task in anneal_tasks:
            task.submit()
            if not task.finish_check():
                raise WorkflowError(
                    f"Embedding anneal for box {index} did not finish "
                    "successfully (no is_finished.info marker). Check the "
                    f"job logs in {task.calcdir}."
                )
            task.collect_results()
            out_extxyz = Path(task.local_savepath) / "out" / "samples.extxyz"
            annealed.append(ase.io.read(out_extxyz))

        # Deliberately under sample_dir, not extract_dir: this is the
        # pipeline's final combined output, not an extraction artifact --
        # keeping it out of extract_dir avoids it looking like one.
        output_path = self.sample_dir / "out" / "samples.extxyz"
        os.makedirs(output_path.parent, exist_ok=True)
        ase.io.write(output_path, annealed)

        logger.info(
            f"Amorphous matrix embedding produced {len(annealed)} annealed "
            f"configurations, written to {output_path}"
        )
        return output_path

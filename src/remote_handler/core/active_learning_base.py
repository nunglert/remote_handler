import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from json.encoder import ESCAPE_DCT
from pathlib import Path
from typing import Callable, Literal, Optional

import ase.io
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


class Sampler(ABC):
    @abstractmethod
    def update(
        self, calcdir: os.PathLike, model_file: os.PathLike, **kwargs
    ) -> None:
        """
        Function updating the sampling logic for the current
        iteration. This in general can include file/directory creation or
        movement, as well as changes to the logic of the sampling itself, like
        e.g. parameters for a nested sampling run.

        Args:
            calcdir: The directory in `calchome` (see `ActiveLearningHandler`)
                of the current iteration. This is passed here such that the
                `Sampler` can eventually write into it.
            model_file: Path to a file containing parameters for a MLFF
                model used by `Sampler`. These are usually pickle files.
            **kwargs: Optional parameters returned from a schedule to modify
                the sampling logic (see also `Schedule`)
        """
        pass

    @abstractmethod
    def get_sampled_configs(self) -> list[Atoms]:
        """Function must return a list of `ase.Atoms` objects which are
        unlabelled."""
        pass


class GroundTruthEvaluator(ABC):
    @abstractmethod
    def update(
        self, calcdir: os.PathLike, collected_file: os.PathLike, **kwargs
    ) -> None:
        """Function updating the ground truth evaluation logic for the current
        iteration. This in general can include file/directory creation or
        movement, as well as changes to the logic of the groun truth evaluation
        itself, like e.g. parameters for DFT.

        Args:
            calcdir: The directory in `calchome` (see `ActiveLearningHandler`)
                of the current iteration. This is passed here such that the
                `GroundTruthEvaluator` can eventually write into it.
            collected_file: Path to a file containing unlabelled configurations
                which shall be labelled.
            **kwargs: Optional parameters returned from a schedule to modify
                the labelling logic (see also `Schedule`)
        """
        pass

    @abstractmethod
    def get_evaluated_configs(self) -> list[Atoms]:
        """Function must return a list of `ase.Atoms` objects which are
        labelled by a ground truth method like DFT. The miniumum requirement
        is that a valid calculator is attached, such that
        `get_potential_energy` and `get_forces` can be called."""
        pass


class ModelTrainer(ABC):
    @abstractmethod
    def update(
        self, calcdir: os.PathLike, db_files: list[os.PathLike], **kwargs
    ) -> None:
        """Function updating the training logic for the current iteration.

        Args:
            calcdir: The directory in `calchome` (see `ActiveLearningHandler`)
                of the current iteration. This is passed here such that the
                `ModelTrainer` can eventually write into it.
            db_files: List of paths to a files containing labelled
                configurations which shall be used for training a MLFF model.
            **kwargs: Optional parameters returned from a schedule to modify
                the labelling logic (see also `Schedule`)
        """
        pass

    @abstractmethod
    def get_trained_model(self) -> os.PathLike:
        """Function must return a path to an existing model file."""
        pass


def process_progress(
    progress: list[dict],
    num_iterations: int,
    # db_files: list[os.PathLike],
    homepath: Path,
) -> tuple[
    int,  # iteration_idx
    Path,  # collected_file
    list[Path],  # db_files
    Path,  # model_file
    dict,  # progress
]:
    """Parses the contents of a dumped `AL_progress.json` file and returns
    the quantities necessary for resuming a active learning loop with
    `ActiveLearningHandler`.
    """
    keys = range(len(progress))
    # collect all config dbs from the previous iterations because they
    # are all needed for retraining.
    assert len(keys) <= num_iterations

    idx = 0
    # pickle_file = db_files[0]
    model_file = None
    collected_file = None
    db_files = []
    if len(keys) > 1:
        for key in keys[:-1]:
            db_file = progress[key].get("db_file")
            if db_file is not None:
                db_files.append(db_file)
            idx += 1
        model_file = progress[key]["model_file"]

    last_iteration_key = keys[-1]
    last_iter_keys = progress[last_iteration_key].keys()
    if "steps" in last_iter_keys:
        steps = progress[last_iteration_key]["steps"]

        remaining_steps = np.sum(steps)
        if remaining_steps > 0:
            # last loaded iteration is incomplete, proceed with missing
            iteration_idx = idx
            # if configs from NS were already collected
            if steps == [1, 1, 1]:
                model_file = progress[last_iteration_key]["model_file"]
            elif steps == [0, 1, 1]:
                collected_file = progress[last_iteration_key]["collected_file"]
            # if dft was already run, collect db_file
            if not steps[1]:
                db_files.append(progress[last_iteration_key]["db_file"])

        else:
            # last loaded iteration is complete, proceed with next iter
            iteration_idx = idx + 1
            collected_file = None
            # sampler_todo, evaluation_todo, training_todo = steps
            db_file = progress[last_iteration_key].get("db_file")
            if db_file is not None:
                db_files.append(db_file)
            model_file = progress[last_iteration_key]["model_file"]
            progress.append({"steps": [1, 1, 1]})

    return iteration_idx, collected_file, db_files, model_file, progress


class Schedule(ABC):
    """Utility class to parametrize schedules for active learning cycles.
    We enforce that the schedule needs to provide the same parameters
    for each iteration (their values can change of course). This makes it
    much easier to check and avoids potential failures.

    Args:
        num_iterations: Number of iterations that should be run with this
            schedule.
    """

    def __init__(self, num_iterations: int):
        super().__init__()
        self.num_iterations = num_iterations
        self.check()

    def visualize(self, column_width: int = 20) -> None:
        """
        Visualizes the parameter that change during the defined schedule.
        It is recommended to check this before actually starting an active
        learning run to see whether you actually implemented what you wanted.

        Args:
            column_width: Width of the column in which parameter is printed to
                stdout.

        Example:
            For a schedule modifying the parameters 'n_walkers', 'gpu_n_batch'
            and 'MC_cell_P' of a nested sampling run, the output of this would
            look like the following:

            n_walkers           gpu_n_batch         MC_cell_P
            200                 200                 3 7 11 15
            300                 400                 2 4 6 8 10 12 14 16
            400                 600                 2 4 6 8 10 12 14 16
        """
        # Get the parameters names
        params = self.get_kwargs(0)
        header_str = ""
        for k in params.keys():
            header_str += f"{k:<{column_width}}"
        logger.info(header_str)

        for i in range(self.num_iterations):
            # Get the parameters values
            params = self.get_kwargs(i)

            iter_str = ""
            for v in params.values():
                iter_str += f"{v:<{column_width}}"
            logger.info(iter_str)

    def check(self) -> None:
        """
        Checks whether at each iteration the returned parameters are the
        same."""
        params = self.get_kwargs(0)
        for i in range(self.num_iterations):
            try:
                kwargs = self.get_kwargs(i)
            except Exception as e:
                raise Exception(
                    "The parametrization of your Schedule seems to be "
                    f"incorrect. Failed due to {e}"
                )

            # Check that the same parameters are returned at each iteration
            if kwargs.keys() != params.keys():
                raise Exception(
                    "Schedule needs to return the same parameters for every "
                    "iteration!"
                )

    def test(self, kwargs_consumer: Callable[[dict], None]) -> None:
        """Eventually, `Schedule` will be used in `ActiveLearningBase` to feed
        some additional kwargs usually into a `Sampler.update`. This function
        can be used to test your schedule before actually running the active
        learning loop. You need to pass a `kwargs_consumer`, which is a
        function that can check the kwargs you pass for validity. While
        potentially being a small additional effort to implement such a
        `kwargs_consumer` it can easily prevent your active loop from crashing
        during the loop.

        Example:
            If the `Sampler` instance handles a nested sampling simulation,
            at each iteration a dictionary is updated by `Sampler.update` and
            parsed by the nested sampling code. This parser checks the
            parameters for validity and can be casted into a `kwargs_consumer`.
        """
        for i in range(self.num_iterations):
            try:
                kwargs_consumer(self.get_kwargs(i))
            except Exception as e:
                raise Exception(
                    f"The kwargs your schedule returns for iteration {i} "
                    f"break your scheduler target. It throws the following "
                    f"error:\n{e}"
                )

    @abstractmethod
    def get_kwargs(self, iteration_idx: int) -> dict:
        """Here you have to encode the logic of your schedule.

        Example:
            For a nested sampling active learning workflow, a simple schedule
            could look like the following:

            def get_kwargs(self, iteration_idx: int):
                if iteration_idx == 0:
                    kwargs = {
                        "n_walkers": 400,
                        "gpu_n_batch": 200,
                        "MC_cell_P": "3 7 11 15",
                    }
                if iteration_idx > 0:
                    kwargs = {
                        "n_walkers": 400,
                        "gpu_n_batch": 400,
                        "MC_cell_P": "2 4 6 8 10 12 14 16",
                    }
                return kwargs
        """
        pass


class ActiveLearningHandler(ABC):
    sampler: Sampler
    evaluator: GroundTruthEvaluator
    trainer: ModelTrainer

    sampler_schedule: Schedule
    evaluator_schedule: Schedule
    trainer_schedule: Schedule

    model_file: os.PathLike
    collected_file: os.PathLike
    db_files: list[Path]

    iteration_idx: int

    """Implementation of the core functionality of an active learning loop
    consisting of three actors: A sampler, a ground truth evaluator and a 
    model trainer. `ActiveLearningHandler` handles the bookkeeping of created
    files and passes required files to the corresponding actors. It offers
    functionality to resume broken runs by caching the current progress and
    at which point to enter/exit the cycle.

    Args:
        sampler: A `Sampler` instance for sampling new configurations.
        evaluator: A `GroundTruthEvaluator` instance for running DFT.
        trainer: A `ModelTrainer` instance for retraining your MLFF model.
        calchome: The working directory for the active learning loop.
        num_iterations: Number of active learning iterations to perform.
        cycle_entry_point: Defines the actor where your loop starts. Can either
            be 'sampler', 'evaluation' or 'training'
        cycle_exit_point: Defines the actor where your loop ends. Can either
            be 'sampler', 'evaluation' or 'training'
        initial_model_file: An initial model file (usually .pickle) containing
            the parameters for your MLFF model, which is required by `Sampler`.
            This is only required if you start from scratch and enter the loop 
            with cycle_entry_point='sampler'
        initial_sample_file: An initial file of unlabeled configurations. This
            is only required if you start from scratch and enter the loop with
            cycle_entry_point='evaluator'
        initial_db_file: An initial file of labeled configurations. This
            is only required if you start from scratch and enter the loop with
            cycle_entry_point='training'
        sampler_schedule: A `Schedule` instance encoding a schedule for the 
            sampler
        evaluator_schedule: A `Schedule` instance encoding a schedule for the 
            evaluator
        trainer_schedule: A `Schedule` instance encoding a schedule for the 
            trainer

    Examples:
        In general `ActiveLearningHandler` will create a folder structure like 
        the following:

            calchome 
            ├── 0
            │   ├── sampler
            │   ├── dft
            │   └── training
            ├── 1
            │   ├── sampler
            │   ├── dft
            │   └── training
            ├── ...
            ├── evaluated_configs
            │   ├── evaluated_0.extxyz
            │   ├── evaluated_1.extxyz
            │   ├── ...
            ├── sampled_configs
            │   ├── sampled_0.extxyz
            │   ├── sampled_1.extxyz
            │   ├── ...
            ├── trained_models
            │   ├── model_0.pkl
            │   ├── model_1.pkl
            │   ├── ...
            ├── AL_progress.json
            └── al_loop.py

        Here, `AL_progress.json` contains the bookkeeping. During each 
        iteration, it updates a dictionary that contains information on the 
        progress of the loop. Once a full loop is finished, it could look like 
        this:

        {
            'steps': [0, 0, 0],
            'collected_file': PATH_TO_COLLECTED,
            'db_file': PATH_TO_DB,
            'model_file': PATH_TO_MODEL
        }

        where 'steps' indicate which actor still needs to be run (hence its 0
        if it was already successfully run). Hence, if e.g. only the sampler 
        finished successfully so far, the entry for the current iteration in 
        `AL_progress.json` would look like this:
        
        {
            'steps': [0, 1, 1],
            'collected_file': PATH_TO_COLLECTED,
        }

        This logic allows the `ActiveLearningHandler` to easily resume runs
        that crashed during some part of the loop.
    """

    def __init__(
        self,
        sampler: Sampler,
        evaluator: GroundTruthEvaluator,
        trainer: ModelTrainer,
        calchome: os.PathLike,
        num_iterations: int = 5,
        cycle_entry_point: Literal[
            "sampler", "evaluation", "training"
        ] = "sampler",
        cycle_exit_point: Literal[
            "sampler", "evaluation", "training"
        ] = "sampler",
        initial_model_file: Optional[os.PathLike] = None,
        initial_sample_file: Optional[os.PathLike] = None,
        initial_db_file: Optional[os.PathLike] = None,
        sampler_schedule: Optional[Schedule] = None,
        evaluator_schedule: Optional[Schedule] = None,
        trainer_schedule: Optional[Schedule] = None,
    ) -> None:
        super().__init__()
        # Main objects of the AL loop
        self.sampler = sampler
        self.evaluator = evaluator
        self.trainer = trainer

        # Schedules for the AL loop
        self.sampler_schedule = sampler_schedule
        self.evaluator_schedule = evaluator_schedule
        self.trainer_schedule = trainer_schedule

        # Maximum number of iterations
        self.num_iterations = num_iterations
        # Main path where the AL loop is run from
        self.calchome = calchome

        # Tracking of the progress of the AL loop
        self.progress_file = self.calchome / "AL_progress.json"
        
        # Function to get paths relative to calchome
        self.get_rel = lambda x: Path(x).relative_to(self.calchome)

        # Check whether we continue a crashed run or start from scratch
        if os.path.exists(self.progress_file):
            logger.info("Continuing crashed run...")
            with open(self.progress_file, "r") as f:
                self.progress = json.load(f)

            (
                self.iteration_idx,
                self.collected_file,
                self.db_files,
                self.model_file,
                self.progress,
            ) = process_progress(
                self.progress, self.num_iterations, self.calchome
            )

            steps = self.progress[self.iteration_idx]["steps"]
            if steps == [1, 1, 1]:
                assert self.model_file is not None
            if steps == [0, 1, 1]:
                assert self.collected_file is not None
            if steps == [0, 0, 1]:
                assert len(self.db_files) > 0

            if len(self.db_files) > self.iteration_idx + 1:
                logger.warning(
                    "Usually each AL iteration adds one db_file. Take care.."
                )
        else:
            #### Start from scratch
            logger.info(
                f"Starting from scratch. Entry at {cycle_entry_point}..."
            )

            self.db_files = []
            self.progress = []
            self.iteration_idx = 0

            # Entry point of the AL loop
            if cycle_entry_point == "sampler":
                # We need an initial model file to start sampling
                assert initial_model_file is not None
                # Move the initial model file to the working directory
                initial_model_file = self.move_file_to_home(initial_model_file)
                # Set the model file path relative to calchome
                self.model_file = self.get_rel(initial_model_file)
                self.progress.append({"steps": [1, 1, 1]})

            elif cycle_entry_point == "evaluation":
                # We need an initial sample file to start evaluation
                assert initial_sample_file is not None
                # Move the initial sample file to the working directory
                initial_sample_file = self.move_file_to_home(
                    initial_sample_file
                )
                # Set the collected file path relative to calchome
                self.collected_file = self.get_rel(initial_sample_file)
                self.progress.append({"steps": [0, 1, 1]})

            elif cycle_entry_point == "training":
                # We need an initial db file to start training
                assert initial_db_file is not None
                # Move the initial db file to the working directory
                initial_db_file = self.move_file_to_home(initial_db_file)
                # Set the db file path relative to calchome
                self.db_files.append(self.get_rel(initial_db_file))
                self.progress.append({"steps": [0, 0, 1]})

            else:
                raise Exception(
                    "Unknown cycle_entry_point. "
                    "Use either sampler, evaluation or training"
                )

        # Fill the remaining iterations with full cycles
        self.remaining_iterations = (
            self.num_iterations - len(self.progress) + 1
        )
        if self.remaining_iterations <= 0:
            raise Exception("No iterations left, increase num_iterations!")

        # Create directories for sampled/evaluated configs and trained models
        # if they do not exist yet
        self.sampled_configs_dir = self.calchome / "sampled_configs"
        self.evaluated_configs_dir = self.calchome / "evaluated_configs"
        self.trained_models_dir = self.calchome / "trained_models"
        for path in [
            self.sampled_configs_dir,
            self.evaluated_configs_dir,
            self.trained_models_dir,
        ]:
            os.makedirs(path, exist_ok=True)

        # Define how the last cycle should look like
        if cycle_exit_point == "sampler":
            self.last_cycle_steps = [1, 0, 0]
        elif cycle_exit_point == "evaluation":
            self.last_cycle_steps = [1, 1, 0]
        elif cycle_exit_point == "training":
            self.last_cycle_steps = [1, 1, 1]
        else:
            raise Exception(
                "Unknown cycle_exit_point. "
                "Use either sampler, evaluation or training"
            )

        logger.info(
            f"Starting active learning cycle for {self.remaining_iterations} "
            "iterations."
        )

    def move_file_to_home(self, file):
        """Moves a file to calchome if it is not already there."""
        if not Path(file).is_relative_to(self.calchome):
            file_home = self.calchome / os.path.split(file)[1] # should't be there a -1?
            shutil.copy(file, file_home)
            file = file_home
        return file

    def update_progress_file(self) -> None:
        """Updates the `AL_progress.json` file."""
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def sample(self, **kwargs) -> None:
        """Runs the Sampler, requiring `self.model_file` and updating
        `self.collected_file`
        It calls `self.sampler.update` to prepare the sampling and then
        retrieves the sampled configurations with `self.sampler.get_sampled_configs`.
        The sampled configurations are written to
        `sampled_configs/samples_{iteration_idx}.extxyz`
        
        Args:
            **kwargs: kwargs passed to the `update` function of `sampler`.
        """
        # Check if model file exists
        assert self.model_file is not None
        logger.info("SAMPLING...")

        # Call the sampler to prepare the directory, the submission script etc.
        # and pass the model file
        self.sampler.update(
            self.current_iter_dir, self.calchome / self.model_file, **kwargs
        )

        # Get the list of atoms objects sampled by the sampler
        sampled_configs = self.sampler.get_sampled_configs()
        
        self.collected_file = self.get_rel(
            self.sampled_configs_dir / f"samples_{self.iteration_idx}.extxyz"
        )
        # Write the sampled configurations to file in a specific directory
        # together with the iteration index
        ase.io.write(self.calchome / self.collected_file, sampled_configs)

        # Update the progress file
        self.progress[self.iteration_idx]["steps"] = [0, 1, 1]
        self.progress[self.iteration_idx]["collected_file"] = str(
            self.collected_file
        )
        self.update_progress_file()

    def evaluate(self, **kwargs) -> None:
        """Runs the GroundTruthEvaluator, requiring `self.model_file` and
        updating `self.db_files`
        It calls `self.evaluator.update` to prepare the evaluation and then
        retrieves the evaluated configurations with
        `self.evaluator.get_evaluated_configs`.
        The evaluated configurations are written to
        `evaluated_configs/evaluated_{iteration_idx}.extxyz`.

        Args:
            **kwargs: kwargs passed to the `update` function of `evaluator`.
        """
        # Check if collected file exists
        assert self.collected_file is not None
        logger.info("EVALUATING GROUND TRUTH...")

        # Call the evaluator to prepare the directory, the submission script
        # etc. and pass the collected file
        self.evaluator.update(
            self.current_iter_dir,
            self.calchome / self.collected_file,
            **kwargs,
        )
        # Get the list of atoms objects evaluated by the evaluator
        evaluated_configs = self.evaluator.get_evaluated_configs()

        # Write the evaluated configurations to file in a specific directory
        # together with the iteration index
        evaluated_file = self.get_rel(
            self.evaluated_configs_dir
            / f"evaluated_{self.iteration_idx}.extxyz"
        )
        ase.io.write(self.calchome / evaluated_file, evaluated_configs)
        self.db_files.append(evaluated_file)
        
        # Update the progress file
        self.progress[self.iteration_idx]["db_file"] = str(evaluated_file)
        self.progress[self.iteration_idx]["steps"] = [0, 0, 1]
        self.update_progress_file()

    def retrain(self, **kwargs) -> None:
        """Runs the ModelTrainer, requiring `self.db_files` and updating
        `self.model_file`
        It calls `self.trainer.update` to prepare the training and then
        retrieves the trained model with `self.trainer.get_trained_model`.
        The trained model is copied to `trained_models/model_{iteration_idx}.pkl`
        At every iteration, all db_files are used for training.

        Args:
            **kwargs: kwargs passed to the `update` function of `trainer`.
        """
        # Check if db files exist
        assert self.db_files
        logger.info("RETRAINING MODEL...")

        # Call the trainer to prepare the directory, the submission script
        # etc. and pass the db files
        self.trainer.update(
            self.current_iter_dir,
            [self.calchome / db_file for db_file in self.db_files],
            **kwargs,
        )
        # Get the path to the trained model
        results_model_file = self.trainer.get_trained_model()
        self.model_file = self.get_rel(
            self.trained_models_dir / f"model_{self.iteration_idx}.pkl"
        )
        # Copy the trained model to a specific directory together with the
        # iteration index
        shutil.copy(results_model_file, self.calchome / self.model_file)

        # Update the progress file
        self.progress[self.iteration_idx]["steps"] = [0, 0, 0]
        self.progress[self.iteration_idx]["model_file"] = str(self.model_file)
        self.update_progress_file()

    def single_iteration(self) -> None:
        """Runs a single active learning iteration. For each iteration it
        creates a new calcdir in `calchome`. 
        If provided, schedules are applied to the individual actors.
        """
        # Create a new directory for this iteration
        self.current_iter_dir = self.calchome / str(self.iteration_idx)
        os.makedirs(self.current_iter_dir, exist_ok=True)

        # Determine which steps to perform this iteration
        (do_sample, do_eval, do_retrain) = self.progress[self.iteration_idx][
            "steps"
        ]
        if do_sample:
            # Sample new configurations
            if self.sampler_schedule is None:
                kwargs = {}
            else:
                # Update kwargs according to the schedule
                kwargs = self.sampler_schedule.get_kwargs(self.iteration_idx)

            self.sample(**kwargs)

        if do_eval:
            # Evaluate the sampled configurations with DFT
            if self.evaluator_schedule is None:
                kwargs = {}
            else:
                # Update kwargs according to the schedule
                kwargs = self.evaluator_schedule.get_kwargs(self.iteration_idx)

            self.evaluate(**kwargs)

        if do_retrain:
            # Retrain the MLFF model
            if self.trainer_schedule is None:
                kwargs = {}
            else:
                # Update kwargs according to the schedule
                kwargs = self.trainer_schedule.get_kwargs(self.iteration_idx)

            self.retrain(**kwargs)

    def run(self) -> None:
        """Runs the active learning cycle. In case a crashed run is restarted,
        it only runs the remaining iterations until `num_iterations` is reached.
        """
        for i in range(self.remaining_iterations):
            self.single_iteration()
            self.iteration_idx += 1

            steps = [1, 1, 1]
            # If this is the second last iteration, we might not want to
            # run a full cycle, but look at self.last_cycle_steps
            if i == self.remaining_iterations - 2:
                steps = self.last_cycle_steps

            # Prepare for the next iteration
            self.progress.append({"steps": steps})
            print("_____________________________________________")
            sys.stdout.flush()

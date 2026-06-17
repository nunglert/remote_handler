import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from typing import Callable, Literal, Optional

import ase.io
import numpy as np
from ase import Atoms

from remote_handler.exceptions import WorkflowError

logger = logging.getLogger(__name__)


class Sampler(ABC):
    @abstractmethod
    def update(
        self, calcdir: Path, sampler_input: Path | list[Path], **kwargs
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
            sampler_input: Path or list of paths to files needed to run `Sampler`.
                For example this could contain one or more paths to MLFF model-files
                for the `Sampler` to run MD and/or perform ensemble evaluations with.
            **kwargs: Optional parameters returned from a schedule to modify
                the sampling logic (see also `Schedule`)
        """
        pass

    @abstractmethod
    def get_sampled_configs(self) -> Path | list[Path]:
        """Function returns a `JsonDict` object containing necessary information
        for upcoming `GroundTruthEvaluator` step."""
        pass


class GroundTruthEvaluator(ABC):
    @abstractmethod
    def update(
        self, calcdir: Path, evaluator_input: Path | list[Path], **kwargs
    ) -> None:
        """Function updating the ground truth evaluation logic for the current
        iteration. This in general can include file/directory creation or
        movement, as well as changes to the logic of the groun truth evaluation
        itself, like e.g. parameters for DFT.

        Args:
            calcdir: The directory in `calchome` (see `ActiveLearningHandler`)
                of the current iteration. This is passed here such that the
                `GroundTruthEvaluator` can eventually write into it.
            evaluator_input: Path or list of Paths to files needed for running
                `GroundTruthEvaluator`. For example this could point to one or
                more .extxyz files with unlabelled configurations for the
                `GroundTruthEvaluator` to run DFT on.
            **kwargs: Optional parameters returned from a schedule to modify
                the labelling logic (see also `Schedule`)
        """
        pass

    @abstractmethod
    def get_evaluated_configs(self) -> Path | list[Path]:
        """Function returns a `JsonDict` object containing necessary information
        for upcoming `ModelTrainer` step."""
        pass


class ModelTrainer(ABC):
    @abstractmethod
    def update(
        self, calcdir: Path, trainer_input: Path | list[Path], **kwargs
    ) -> None:
        """Function updating the training logic for the current iteration.

        Args:
            calcdir: The directory in `calchome` (see `ActiveLearningHandler`)
                of the current iteration. This is passed here such that the
                `ModelTrainer` can eventually write into it.
            trainer_input: Path or list of Paths to files needed for running
                `ModelTrainer`. For example this could point to one or more
                .extxyz files with DFT-evaluated configurations to use for
                the next training iteration.
            **kwargs: Optional parameters returned from a schedule to modify
                the labelling logic (see also `Schedule`)
        """
        pass

    @abstractmethod
    def get_trained_model(self) -> Path | list[Path]:
        """Function returns a `JsonDict` object containing necessary information
        for upcoming `Sampler` step"""
        pass


def append_or_extend(item, append_list):
    """Appends item to append_list if item is not a list. If item is a list,
    appends each individual entry of item to append_list instead.
    """
    if isinstance(item, list):
        append_list.extend(item)
    else:
        append_list.append(item)


def process_progress(
    progress: list[dict],
    num_iterations: int,
    combine_sampler_outputs: bool,
    combine_evaluator_outputs: bool,
    combine_trainer_outputs: bool,
) -> tuple[
    int,  # iteration_idx
    list[Path],  # evaluator input
    list[Path],  # trainer input
    list[Path],  # sampler input
    list[dict],  # progress
]:
    """Parses the contents of a dumped `AL_progress.json` file and returns
    the quantities necessary for resuming a active learning loop with
    `ActiveLearningHandler`.
    """
    # TODO: Unsure what happens when restarting in final generation with exit point that is not training
    # TODO: This appends the wrong steps if current steps are [0,0,0] and next gen is final one with steps != [1,1,1]
    keys = range(len(progress))
    assert len(keys) <= num_iterations

    evaluator_in = []
    trainer_in = []
    sampler_in = []
    if combine_sampler_outputs:
        if len(keys) > 1:
            for key in keys[:-1]:
                if "sampler_output" in progress[key].keys():
                    append_or_extend(
                        progress[key].get("sampler_output"), evaluator_in
                    )

    if combine_evaluator_outputs:
        if len(keys) > 1:
            for key in keys[:-1]:
                if "evaluator_output" in progress[key].keys():
                    append_or_extend(
                        progress[key].get("evaluator_output"), trainer_in
                    )

    if combine_trainer_outputs:
        if len(keys) > 1:
            for key in keys[:-1]:
                if "trainer_output" in progress[key].keys():
                    append_or_extend(
                        progress[key].get("trainer_output"), sampler_in
                    )

    last_iteration_key = keys[-1]
    last_iter_keys = progress[last_iteration_key].keys()
    if "steps" in last_iter_keys:
        steps = progress[last_iteration_key]["steps"]
        iteration_idx = last_iteration_key

        if steps in [[1, 1, 1], [1, 1, 0], [1, 0, 0]]:
            if not combine_trainer_outputs and last_iteration_key != 0:
                # not needed if combine is true since the previous generation
                # trainer_output would have already been added above
                append_or_extend(
                    progress[last_iteration_key - 1]["trainer_output"],
                    sampler_in,
                )

        # if configs from sampling were already collected
        elif steps in [[0, 1, 1] or [0, 1, 0]]:
            append_or_extend(
                progress[last_iteration_key]["sampler_output"], evaluator_in
            )

        # if dft was already run, collect sampler_out as well as evaluator_out
        elif steps == [0, 0, 1]:
            if combine_sampler_outputs:
                append_or_extend(
                    progress[last_iteration_key]["sampler_output"],
                    evaluator_in,
                )

            append_or_extend(
                progress[last_iteration_key]["evaluator_output"], trainer_in
            )

        else:
            # if steps are [0,0,0], all combinable outputs need to be appended
            # even if combine_trainer_output=False, trainer_output is needed for
            # the next generation's sampling
            if combine_sampler_outputs:
                append_or_extend(
                    progress[last_iteration_key]["sampler_output"],
                    evaluator_in,
                )
            if combine_evaluator_outputs:
                append_or_extend(
                    progress[last_iteration_key]["evaluator_output"],
                    trainer_in,
                )
            append_or_extend(
                progress[last_iteration_key]["trainer_output"], sampler_in
            )

            # last loaded iteration is complete, proceed with next iter
            iteration_idx = last_iteration_key + 1
            progress.append({"steps": [1, 1, 1]})

    evaluator_in = [Path(p) for p in evaluator_in]
    trainer_in = [Path(p) for p in trainer_in]
    sampler_in = [Path(p) for p in sampler_in]

    return iteration_idx, evaluator_in, trainer_in, sampler_in, progress


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
                raise WorkflowError(
                    "The parametrization of your Schedule seems to be "
                    f"incorrect. Failed due to {e}"
                )

            # Check that the same parameters are returned at each iteration
            if kwargs.keys() != params.keys():
                raise WorkflowError(
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
                raise WorkflowError(
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
        initial_trainer_out: Any input similar to the outputs of the used
            `ModelTrainer` which should be used by the `Sampler`. May for example
            be the path(s) to one or several model files that should be used in
            the sampling step. Required when starting from scratch with
            cycle_entry_point='sampler', however will also be used if
            combine_trainer_outputs=True.
        initial_sampler_out: Any input similar to the outputs of the used
            `Sampler` which should be used by the `GroundTruthEvaluator`.
            May for example be the path(s) to one or several .extxyz files
            that should be evaluated in the evaluation step. Required when
            starting from scratch with cycle_entry_point='evaluation', however
            will also be used if combine_sampler_outputs=True.
        initial_evaluator_out: Any input similar to the outputs of the used
            `GroundTruthEvaluator` which should be used by the `ModelTrainer`.
            May for example be the path(s) to one or several .extxyz files
            that should be used in the training step. Required when starting
            from scratch with cycle_entry_point='training', however will also
            be used if combine_evaluator_outputs=True.
        sampler_schedule: A `Schedule` instance encoding a schedule for the
            sampler
        evaluator_schedule: A `Schedule` instance encoding a schedule for the
            evaluator
        trainer_schedule: A `Schedule` instance encoding a schedule for the
            trainer
        combine_sampler_outputs: If true, the outputs of all previous sampling
            steps (as well as initial_sampler_out) will be passed to the
            evaluator. Otherwise only the outputs of the most recent sampling step
            (or initial_sampler_out if starting from scratch at the evaluation
            step) will be passed.
        combine_evaluator_outputs: If true, the outputs of all previous evaluation
            steps (as well as initial_evaluator_out) will be passed to the
            trainer. Otherwise only the outputs of the most recent evaluation step
            (or initial_evaluator_out if starting from scratch at the training
            step) will be passed.
        combine_trainer_outputs: If true, the outputs of all previous training
            steps (as well as initial_trainer_out) will be passed to the
            sampler. Otherwise only the outputs of the most recent training step
            (or initial_trainer_out if starting from scratch at the training
            step) will be passed.
        sampler_output_extension: File extension for the `Sampler`'s output files.
        evaluator_output_extension: File extension for the `GroundTruthEvaluator`'s
            output files.
        trainer_output_extension: File extension for the `ModelTrainer`'s output
            files.

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
            'sampler_output': [PATH_TO_SAMPLES],
            'evaluator_output': [PATH_TO_DB],
            'trainer_output': [PATH_TO_MODEL1, PATH_TO_MODEL2, ...]
        }

        where 'steps' indicate which actor still needs to be run (hence its 0
        if it was already successfully run). Hence, if e.g. only the sampler
        finished successfully so far, the entry for the current iteration in
        `AL_progress.json` would look like this:

        {
            'steps': [0, 1, 1],
            'sampler_output': [PATH_TO_SAMPLES],
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
        initial_trainer_out: Optional[Path | list[Path]] = None,
        initial_sampler_out: Optional[Path | list[Path]] = None,
        initial_evaluator_out: Optional[Path | list[Path]] = None,
        sampler_schedule: Optional[Schedule] = None,
        evaluator_schedule: Optional[Schedule] = None,
        trainer_schedule: Optional[Schedule] = None,
        combine_sampler_outputs: bool = False,
        combine_evaluator_outputs: bool = True,
        combine_trainer_outputs: bool = False,
        sampler_output_extension: str = "extxyz",
        evaluator_output_extension: str = "extxyz",
        trainer_output_extension: str = "pkl",
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
        self.combine_sampler_outputs = combine_sampler_outputs
        self.combine_evaluator_outputs = combine_evaluator_outputs
        self.combine_trainer_outputs = combine_trainer_outputs
        self.sampler_output_extension = sampler_output_extension
        self.evaluator_output_extension = evaluator_output_extension
        self.trainer_output_extension = trainer_output_extension

        # Maximum number of iterations
        self.num_iterations = num_iterations
        self.calchome = Path(calchome)

        # Tracking of the progress of the AL loop
        self.progress_file = self.calchome / "AL_progress.json"
        self.sampler_in = []
        self.evaluator_in = []
        self.trainer_in = []

        self.get_rel = lambda x: Path(x).relative_to(self.calchome)

        # Check whether we continue a crashed run or start from scratch
        if os.path.exists(self.progress_file):
            logger.info("Continuing crashed run...")
            with open(self.progress_file, "r") as f:
                self.progress = json.load(f)

            (
                self.iteration_idx,
                self.evaluator_in,
                self.trainer_in,
                self.sampler_in,
                self.progress,
            ) = process_progress(
                self.progress,
                self.num_iterations,
                self.combine_sampler_outputs,
                self.combine_evaluator_outputs,
                self.combine_trainer_outputs,
            )

            if self.iteration_idx + 1 == self.num_iterations and self.progress[
                self.iteration_idx
            ]["steps"] == [1, 1, 1]:
                if cycle_exit_point == "sampler":
                    self.progress[self.iteration_idx]["steps"] = [1, 0, 0]
                elif cycle_exit_point == "evaluator":
                    self.progress[self.iteration_idx]["steps"] = [1, 1, 0]

            steps = self.progress[self.iteration_idx]["steps"]
            if steps in [[1, 1, 1], [1, 1, 0], [1, 0, 0]]:
                assert len(self.sampler_in) > 0
            if steps in [[0, 1, 1], [0, 1, 0]]:
                assert len(self.evaluator_in) > 0
            if steps == [0, 0, 1]:
                assert len(self.trainer_in) > 0

        else:
            #### Start from scratch
            logger.info(
                f"Starting from scratch. Entry at {cycle_entry_point}..."
            )

            # self.db_files = []
            self.progress = []
            self.iteration_idx = 0

            # Entry point of the AL loop
            if cycle_entry_point == "sampler":
                assert initial_trainer_out is not None
                if not combine_trainer_outputs:
                    initial_trainer_out = (
                        [self.move_file_to_home(initial_trainer_out)]
                        if isinstance(initial_trainer_out, Path)
                        else [
                            self.move_file_to_home(p)
                            for p in initial_trainer_out
                        ]
                    )
                    initial_trainer_out = [
                        self.get_rel(p) for p in initial_trainer_out
                    ]
                    self.sampler_in = initial_trainer_out
                self.progress.append({"steps": [1, 1, 1]})

            elif cycle_entry_point == "evaluation":
                assert initial_sampler_out is not None
                if not combine_sampler_outputs:
                    initial_sampler_out = (
                        [self.move_file_to_home(initial_sampler_out)]
                        if isinstance(initial_sampler_out, Path)
                        else [
                            self.move_file_to_home(p)
                            for p in initial_sampler_out
                        ]
                    )
                    initial_sampler_out = [
                        self.get_rel(p) for p in initial_sampler_out
                    ]
                    self.evaluator_in = initial_sampler_out
                self.progress.append({"steps": [0, 1, 1]})

            elif cycle_entry_point == "training":
                assert initial_evaluator_out is not None
                if not combine_evaluator_outputs:
                    initial_evaluator_out = (
                        [self.move_file_to_home(initial_evaluator_out)]
                        if isinstance(initial_evaluator_out, Path)
                        else [
                            self.move_file_to_home(p)
                            for p in initial_evaluator_out
                        ]
                    )
                    initial_evaluator_out = [
                        self.get_rel(p) for p in initial_evaluator_out
                    ]
                    self.trainer_in = initial_evaluator_out
                self.progress.append({"steps": [0, 0, 1]})

        # make sure that initial inputs are always part of lists if corresponding
        # combine parameter is set
        if initial_trainer_out is not None and combine_trainer_outputs:
            initial_trainer_out = (
                [self.move_file_to_home(initial_trainer_out)]
                if isinstance(initial_trainer_out, Path)
                else [self.move_file_to_home(p) for p in initial_trainer_out]
            )
            initial_trainer_out = [
                self.get_rel(p) for p in initial_trainer_out
            ]
            self.sampler_in.extend(initial_trainer_out)

        if initial_sampler_out is not None and combine_sampler_outputs:
            initial_sampler_out = (
                [self.move_file_to_home(initial_sampler_out)]
                if isinstance(initial_sampler_out, Path)
                else [self.move_file_to_home(p) for p in initial_sampler_out]
            )
            initial_sampler_out = [
                self.get_rel(p) for p in initial_sampler_out
            ]
            self.evaluator_in.extend(initial_sampler_out)

        if initial_evaluator_out is not None and combine_evaluator_outputs:
            initial_evaluator_out = (
                [self.move_file_to_home(initial_evaluator_out)]
                if isinstance(initial_evaluator_out, Path)
                else [self.move_file_to_home(p) for p in initial_evaluator_out]
            )
            initial_evaluator_out = [
                self.get_rel(p) for p in initial_evaluator_out
            ]
            self.trainer_in.extend(initial_evaluator_out)

        # Fill the remaining iterations with full cycles
        self.remaining_iterations = (
            self.num_iterations - len(self.progress) + 1
        )
        if self.remaining_iterations <= 0:
            raise WorkflowError("No iterations left, increase num_iterations!")

        self.sampler_output_dir = self.calchome / "sampler_output"
        self.evaluator_output_dir = self.calchome / "evaluator_output"
        self.trainer_output_dir = self.calchome / "trainer_output"
        for path in [
            self.sampler_output_dir,
            self.evaluator_output_dir,
            self.trainer_output_dir,
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
            raise WorkflowError(
                "Unknown cycle_exit_point. "
                "Use either sampler, evaluation or training"
            )

        logger.info(
            f"Starting active learning cycle for {self.remaining_iterations} "
            "iterations."
        )

    def move_file_to_home(self, file: Path):
        """Moves a file from an external path into the AL home directory

        Args:
            file: path of file which should be moved to AL home

        Returns:
            Path to moved file
        """
        if not file.is_relative_to(self.calchome):
            # calchome may not exist yet: initial_*_out files are moved in
            # during __init__, before the per-iteration dir structure is built.
            os.makedirs(self.calchome, exist_ok=True)
            file_home = self.calchome / file.name
            # initial_*_out may be a directory (e.g. a mace-jax model bundle),
            # not just a single file -- copy the whole tree in that case.
            if file.is_dir():
                shutil.copytree(file, file_home, dirs_exist_ok=True)
            else:
                shutil.copy(file, file_home)
            file = file_home
        return file

    def update_progress_file(self) -> None:
        """Updates the `AL_progress.json` file."""
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def sample(self, **kwargs) -> None:
        """Runs the Sampler, requiring `self.sampler_in` and updating
        `self.evaluator_in`

        Args:
            **kwargs: kwargs passed to the `update` function of `sampler`.
        """
        assert self.sampler_in is not None and len(self.sampler_in) > 0
        logger.info("SAMPLING...")
        input_arg = (
            self.calchome / self.sampler_in[0]
            if len(self.sampler_in) == 1
            else [self.calchome / p for p in self.sampler_in]
        )
        self.sampler.update(self.current_iter_dir, input_arg, **kwargs)
        sampled_configs = self.sampler.get_sampled_configs()
        sampled_configs = (
            [sampled_configs]
            if isinstance(sampled_configs, Path)
            else sampled_configs
        )

        new_paths = []
        for i, p in enumerate(sampled_configs):
            path = self.get_rel(
                self.sampler_output_dir
                / f"sampler_it{self.iteration_idx}_{i}.{self.sampler_output_extension}"
            )
            shutil.copy(p, self.calchome / path)
            new_paths.append(path)

        if self.combine_sampler_outputs:
            self.evaluator_in.extend(new_paths)
        else:
            self.evaluator_in = new_paths

        self.progress[self.iteration_idx]["steps"] = [0, 1, 1]
        self.progress[self.iteration_idx]["sampler_output"] = [
            str(p) for p in new_paths
        ]
        self.update_progress_file()

    def evaluate(self, **kwargs) -> None:
        """Runs the GroundTruthEvaluator, requiring `self.evaluator_in` and
        updating `self.trainer_in`

        Args:
            **kwargs: kwargs passed to the `update` function of `evaluator`.
        """
        assert self.evaluator_in is not None and len(self.evaluator_in) > 0
        logger.info("EVALUATING GROUND TRUTH...")
        input_arg = (
            self.calchome / self.evaluator_in[0]
            if len(self.evaluator_in) == 1
            else [self.calchome / p for p in self.evaluator_in]
        )

        self.evaluator.update(self.current_iter_dir, input_arg, **kwargs)
        evaluated_configs = self.evaluator.get_evaluated_configs()
        evaluated_configs = (
            [evaluated_configs]
            if isinstance(evaluated_configs, Path)
            else evaluated_configs
        )

        new_paths = []
        for i, p in enumerate(evaluated_configs):
            path = self.get_rel(
                self.evaluator_output_dir
                / f"evaluator_it{self.iteration_idx}_{i}.{self.evaluator_output_extension}"
            )
            shutil.copy(p, self.calchome / path)
            new_paths.append(path)

        if self.combine_evaluator_outputs:
            self.trainer_in.extend(new_paths)
        else:
            self.trainer_in = new_paths

        self.progress[self.iteration_idx]["steps"] = [0, 0, 1]
        self.progress[self.iteration_idx]["evaluator_output"] = [
            str(p) for p in new_paths
        ]
        self.update_progress_file()

    def retrain(self, **kwargs) -> None:
        """Runs the ModelTrainer, requiring `self.trainer_in` and updating
        `self.sampler_in`

        Args:
            **kwargs: kwargs passed to the `update` function of `trainer`.
        """
        assert self.trainer_in is not None and len(self.trainer_in) > 0
        logger.info("RETRAINING MODEL...")
        input_arg = (
            self.calchome / self.trainer_in[0]
            if len(self.trainer_in) == 1
            else [self.calchome / p for p in self.trainer_in]
        )
        self.trainer.update(
            self.current_iter_dir,
            input_arg,
            **kwargs,
        )
        results_model_file = self.trainer.get_trained_model()
        results_model_file = (
            [results_model_file]
            if isinstance(results_model_file, Path)
            else results_model_file
        )

        new_paths = []
        for i, p in enumerate(results_model_file):
            path = self.get_rel(
                self.trainer_output_dir
                / f"trainer_it{self.iteration_idx}_{i}.{self.trainer_output_extension}"
            )
            shutil.copy(p, self.calchome / path)
            new_paths.append(path)

        if self.combine_trainer_outputs:
            self.sampler_in.extend(new_paths)
        else:
            self.sampler_in = new_paths

        # Update the progress file
        self.progress[self.iteration_idx]["steps"] = [0, 0, 0]
        self.progress[self.iteration_idx]["trainer_output"] = [
            str(p) for p in new_paths
        ]
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

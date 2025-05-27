import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from json.encoder import ESCAPE_DCT
from typing import Callable, Optional

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
        pass

    @abstractmethod
    def get_sampled_configs(self) -> list[Atoms]:
        pass


class GroundTruthEvaluator(ABC):
    @abstractmethod
    def update(
        self, calcdir: os.PathLike, collected_file: os.PathLike, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def get_evaluated_configs(self) -> list[Atoms]:
        pass


class ModelTrainer(ABC):
    @abstractmethod
    def update(
        self, calcdir: os.PathLike, db_files: list[os.PathLike], **kwargs
    ) -> None:
        pass

    @abstractmethod
    def get_trained_model(self) -> os.PathLike:
        pass


def process_progress(
    progress,
    num_iterations,
    db_files,
):
    keys = range(len(progress))
    # collect all config dbs from the previous iterations because they
    # are all needed for retraining.
    assert len(keys) <= num_iterations

    idx = 0
    # pickle_file = db_files[0]
    model_file = None
    collected_file = None
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
            sampler_todo, evaluation_todo, training_todo = steps
            db_file = progress[last_iteration_key].get("db_file")
            if db_file is not None:
                db_files.append(db_file)
            model_file = progress[last_iteration_key]["model_file"]
            progress.append({"steps": [1, 1, 1]})

    return iteration_idx, collected_file, db_files, model_file, progress


class Schedule(ABC):
    """Utility class to parametrize schedules for active learning cycles."""

    def __init__(self, num_iterations):
        super().__init__()
        self.num_iterations = num_iterations
        self.check()

    def visualize(self, column_width=20):
        params = self.get_kwargs(0)
        header_str = ""
        for k in params.keys():
            header_str += f"{k:<{column_width}}"
        print(header_str)

        for i in range(self.num_iterations):
            params = self.get_kwargs(i)

            iter_str = ""
            for v in params.values():
                iter_str += f"{v:<{column_width}}"
            print(iter_str)

    def check(self):
        params = self.get_kwargs(0)
        for i in range(self.num_iterations):
            try:
                kwargs = self.get_kwargs(i)
            except Exception as e:
                raise Exception(
                    "The parametrization of your Schedule seems to be "
                    f"incorrect. Failed due to {e}"
                )

            if kwargs.keys() != params.keys():
                raise Exception(
                    "Schedule needs to return the same parameters for every "
                    "iteration!"
                )

    def test(self, kwargs_consumer: Callable[[dict], None]):
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
    db_files: list[os.PathLike]

    iteration_idx: int

    def __init__(
        self,
        sampler: Sampler,
        evaluator: GroundTruthEvaluator,
        trainer: ModelTrainer,
        calcdir: os.PathLike,
        initial_sample_file: Optional[os.PathLike] = None,
        initial_db_file: Optional[os.PathLike] = None,
        initial_model_file: Optional[os.PathLike] = None,
        sampler_schedule: Optional[Schedule] = None,
        evaluator_schedule: Optional[Schedule] = None,
        trainer_schedule: Optional[Schedule] = None,
        num_iterations: int = 5,
        cycle_entry_point: str = "sampler",
        cycle_exit_point: str = "sampler",
    ):
        super().__init__()
        self.sampler = sampler
        self.evaluator = evaluator
        self.trainer = trainer
        self.sampler_schedule = sampler_schedule
        self.evaluator_schedule = evaluator_schedule
        self.trainer_schedule = trainer_schedule

        self.num_iterations = num_iterations
        self.calcdir = calcdir

        self.progress_file = self.calcdir / "AL_progress.json"

        self.db_files = []
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
                self.progress, self.num_iterations, self.db_files
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
            logger.info(
                f"Starting from scratch. Entry at {cycle_entry_point}..."
            )

            self.progress = []
            self.iteration_idx = 0

            if cycle_entry_point == "sampler":
                assert initial_model_file is not None
                self.model_file = initial_model_file
                self.progress.append({"steps": [1, 1, 1]})

            elif cycle_entry_point == "evaluation":
                assert initial_sample_file is not None
                self.collected_file = initial_sample_file
                self.progress.append({"steps": [0, 1, 1]})

            elif cycle_entry_point == "training":
                assert initial_db_file is not None
                self.db_files.append(initial_db_file)
                self.progress.append({"steps": [0, 0, 1]})

            else:
                raise Exception(
                    "Unknown cycle_entry_point. "
                    "Use either sampler, evaluation or training"
                )

        self.remaining_iterations = (
            self.num_iterations - len(self.progress) + 1
        )
        if self.remaining_iterations <= 0:
            raise Exception("No iterations left, increase num_iterations!")

        self.sampled_configs_dir = self.calcdir / "sampled_configs"
        self.evaluated_configs_dir = self.calcdir / "evaluated_configs"
        self.trained_models_dir = self.calcdir / "trained_models"
        for path in [
            self.sampled_configs_dir,
            self.evaluated_configs_dir,
            self.trained_models_dir,
        ]:
            os.makedirs(path, exist_ok=True)

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

    def update_progress_file(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def sample(self, **kwargs) -> None:
        """Runs the Sampler, requiring `self.model_file` and updating
        `self.collected_file`"""
        assert self.model_file is not None
        self.sampler.update(self.current_iter_dir, self.model_file, **kwargs)
        sampled_configs = self.sampler.get_sampled_configs()
        self.collected_file = (
            self.sampled_configs_dir / f"samples_{self.iteration_idx}.extxyz"
        )
        ase.io.write(self.collected_file, sampled_configs)
        self.progress[self.iteration_idx]["steps"] = [0, 1, 1]
        self.progress[self.iteration_idx]["collected_file"] = str(
            self.collected_file
        )
        self.update_progress_file()

    def evaluate(self, **kwargs) -> None:
        """Runs the GroundTruthEvaluator, requiring `self.model_file` and
        updating `self.db_files`"""
        assert self.collected_file is not None
        self.evaluator.update(
            self.current_iter_dir, self.collected_file, **kwargs
        )
        evaluated_configs = self.evaluator.get_evaluated_configs()
        evaluated_file = (
            self.evaluated_configs_dir
            / f"evaluated_{self.iteration_idx}.extxyz"
        )
        ase.io.write(evaluated_file, evaluated_configs)
        self.db_files.append(evaluated_file)
        self.progress[self.iteration_idx]["db_file"] = str(evaluated_file)
        self.progress[self.iteration_idx]["steps"] = [0, 0, 1]
        self.update_progress_file()

    def retrain(self, **kwargs):
        """Runs the ModelTrainer, requiring `self.db_files` and updating
        `self.model_file`"""
        assert self.db_files
        self.trainer.update(self.current_iter_dir, self.db_files, **kwargs)
        results_model_file = self.trainer.get_trained_model()
        self.model_file = (
            self.trained_models_dir / f"model_{self.iteration_idx}.pkl"
        )
        shutil.copy(results_model_file, self.model_file)

        self.progress[self.iteration_idx]["steps"] = [0, 0, 0]
        self.progress[self.iteration_idx]["model_file"] = str(self.model_file)
        self.update_progress_file()

    def single_iteration(self):
        self.current_iter_dir = self.calcdir / str(self.iteration_idx)
        os.makedirs(self.current_iter_dir, exist_ok=True)

        (do_sample, do_eval, do_retrain) = self.progress[self.iteration_idx][
            "steps"
        ]
        if do_sample:
            if self.sampler_schedule is None:
                kwargs = {}
            else:
                kwargs = self.sampler_schedule.get_kwargs(self.iteration_idx)

            self.sample(**kwargs)

        if do_eval:
            if self.evaluator_schedule is None:
                kwargs = {}
            else:
                kwargs = self.evaluator_schedule.get_kwargs(self.iteration_idx)

            self.evaluate()

        if do_retrain:
            if self.trainer_schedule is None:
                kwargs = {}
            else:
                kwargs = self.trainer_schedule.get_kwargs(self.iteration_idx)

            self.retrain()

    def run(self):
        """Runs the active learning cycle. In case a crashed run is restarted,
        it only runs the remaining iterations until `num_iterations` is reached.
        """
        for i in range(self.remaining_iterations):
            self.single_iteration()
            self.iteration_idx += 1

            steps = [1, 1, 1]
            if i == self.remaining_iterations - 2:
                steps = self.last_cycle_steps

            self.progress.append({"steps": steps})
            print("_____________________________________________")
            sys.stdout.flush()

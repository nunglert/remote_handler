import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

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
    if len(keys) > 1:
        for key in keys[:-1]:
            db_files.append(progress[key]["db_file"])
            idx += 1
        model_file = progress[key]["model_file"]

    last_iteration_key = keys[-1]
    last_iter_keys = progress[last_iteration_key].keys()
    if "steps" in last_iter_keys:
        steps = progress[last_iteration_key]["steps"]

        if np.sum(steps) > 0:
            # last loaded iteration is incomplete, proceed with missing
            iteration_idx = idx
            # if configs from NS were already collected
            if not steps[0]:
                collected_file = progress[last_iteration_key]["collected_file"]
            # if dft was already run, collect db_file
            if not steps[1]:
                db_files.append(progress[last_iteration_key]["db_file"])

        else:
            # last loaded iteration is complete, proceed with next iter
            iteration_idx = idx + 1
            collected_file = None
            db_files.append(progress[last_iteration_key]["db_file"])
            model_file = progress[last_iteration_key]["model_file"]
            progress.append({"steps": [1, 1, 1]})

    return iteration_idx, collected_file, db_files, model_file, progress


class ActiveLearningHandler(ABC):
    sampler: Sampler
    evaluator: GroundTruthEvaluator
    trainer: ModelTrainer

    model_file: os.PathLike
    collected_file: os.PathLike
    db_files: list[os.PathLike]

    def __init__(
        self,
        sampler: Sampler,
        evaluator: GroundTruthEvaluator,
        trainer: ModelTrainer,
        calcdir: os.PathLike,
        initial_model_file: Optional[os.PathLike] = None,
        initial_db_file: Optional[os.PathLike] = None,
        num_iterations: int = 5,
    ):
        super().__init__()
        self.sampler = sampler
        self.evaluator = evaluator
        self.trainer = trainer

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
            # starting from scratch
            assert initial_model_file is not None
            assert initial_db_file is not None
            self.progress = []
            self.iteration_idx = 0
            self.model_file = initial_model_file
            self.db_files.append(initial_db_file)
            self.progress.append({"steps": [1, 1, 1]})

        self.remaining_iterations = (
            self.num_iterations - len(self.progress) + 1
        )
        assert self.remaining_iterations > 0

        self.sampled_configs_dir = self.calcdir / "sampled_configs"
        self.evaluated_configs_dir = self.calcdir / "evaluated_configs"
        for path in [self.sampled_configs_dir, self.evaluated_configs_dir]:
            os.makedirs(path, exist_ok=True)

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
        self.model_file = self.trainer.get_trained_model()
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
            self.sample()

        if do_eval:
            self.evaluate()

        if do_retrain:
            self.retrain()

    def run(self):
        """Runs the active learning cycle. In case a crashed run is restarted,
        it only runs the remaining iterations until `num_iterations` is reached.
        """
        for _ in range(self.remaining_iterations):
            self.single_iteration()
            self.iteration_idx += 1
            self.progress.append({"steps": [1, 1, 1]})
            print("_____________________________________________")
            sys.stdout.flush()

import os
import shutil
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.core.active_learning_base import (
    ActiveLearningHandler,
    GroundTruthEvaluator,
    ModelTrainer,
    Sampler,
    process_progress,
)


class SamplerMock(Sampler):
    def __init__(self, path):
        self.sample_file = Path(path) / "sample.xyz"
        with open(self.sample_file, "w") as f:
            f.write("test")

    def update(self, calcdir, sampler_input, **kwargs):
        if isinstance(sampler_input, Path):
            assert os.path.exists(sampler_input.absolute())
        else:
            for p in sampler_input:
                assert os.path.exists(p.absolute())
        os.makedirs(Path(calcdir) / "sampler_dir")

    def get_sampled_configs(self):
        return self.sample_file


class EvaluatorMock(GroundTruthEvaluator):
    def __init__(self, path):
        self.db_file = Path(path) / "db.xyz"
        with open(self.db_file, "w") as f:
            f.write("test")

    def update(self, calcdir, evaluator_input, **kwargs):
        if isinstance(evaluator_input, Path):
            assert os.path.exists(evaluator_input.absolute())
        else:
            for p in evaluator_input:
                assert os.path.exists(p.absolute())
        os.makedirs(Path(calcdir) / "evaluator_dir")

    def get_evaluated_configs(self):
        return self.db_file


class TrainerMock(ModelTrainer):
    def __init__(self, path):
        self.model_file = Path(path) / "model.pkl"
        with open(self.model_file, "w") as f:
            f.write("keklelsos")

    def update(self, calcdir, trainer_input, **kwargs):
        if isinstance(trainer_input, Path):
            assert os.path.exists(trainer_input.absolute())
        else:
            for p in trainer_input:
                assert os.path.exists(p.absolute())
        os.makedirs(Path(calcdir) / "trainer_dir")

    def get_trained_model(self):
        return self.model_file


@pytest.mark.parametrize(
    "cycle_entry_point", ["sampler", "evaluation", "training"]
)
def test_ActiveLearningHandler(data_dir, cycle_entry_point):
    asset_dir = data_dir / "active_learning/handler"
    temp_dir = data_dir / "temp/active_learning/handler"

    local_dir = temp_dir / "local_dir"
    local_results = temp_dir / "results"

    for path in [local_dir, local_results]:
        if path is not None:
            if os.path.exists(path):
                shutil.rmtree(path)
    os.makedirs(local_dir, exist_ok=False)

    sampler = SamplerMock(temp_dir)
    evaluator = EvaluatorMock(temp_dir)
    trainer = TrainerMock(temp_dir)

    initial_model_file = local_dir / "model.pkl"
    with open(initial_model_file, "w") as f:
        f.write("keklelsos")

    initial_db_file = local_dir / "initial_db.extxyz"
    with open(initial_db_file, "w") as f:
        f.write("keklelsos")

    initial_sample_file = local_dir / "initial_sample.extxyz"
    with open(initial_sample_file, "w") as f:
        f.write("keklelsos")

    al_handler = ActiveLearningHandler(
        sampler,
        evaluator,
        trainer,
        calchome=local_dir,
        initial_trainer_out=initial_model_file,
        initial_sampler_out=initial_sample_file,
        initial_evaluator_out=initial_db_file,
        num_iterations=5,
        cycle_entry_point=cycle_entry_point,
    )

    al_handler.run()


@pytest.mark.parametrize(
    "cycle_exit_point", ["sampler", "evaluation", "training"]
)
def test_restart_ActiveLearningHandler(data_dir, cycle_exit_point):
    asset_dir = data_dir / "active_learning/handler_restart"
    temp_dir = data_dir / "temp/active_learning/handler_restart"

    local_dir = temp_dir / "local_dir"
    local_results = temp_dir / "results"

    for path in [local_dir, local_results]:
        if path is not None:
            if os.path.exists(path):
                shutil.rmtree(path)
    os.makedirs(local_dir, exist_ok=False)

    sampler = SamplerMock(temp_dir)
    evaluator = EvaluatorMock(temp_dir)
    trainer = TrainerMock(temp_dir)

    initial_model_file = temp_dir / "model.pkl"
    with open(initial_model_file, "w") as f:
        f.write("keklelsos")

    initial_db_file = temp_dir / "initial_db.extxyz"
    with open(initial_db_file, "w") as f:
        f.write("keklelsos")

    initial_sample_file = local_dir / "initial_sample.extxyz"
    with open(initial_sample_file, "w") as f:
        f.write("keklelsos")

    al_handler = ActiveLearningHandler(
        sampler,
        evaluator,
        trainer,
        calchome=local_dir,
        initial_trainer_out=initial_model_file,
        initial_evaluator_out=initial_db_file,
        initial_sampler_out=initial_sample_file,
        num_iterations=5,
        cycle_exit_point=cycle_exit_point,
    )

    al_handler.run()

    al_handler = ActiveLearningHandler(
        sampler,
        evaluator,
        trainer,
        calchome=local_dir,
        initial_trainer_out=initial_model_file,
        initial_evaluator_out=initial_db_file,
        initial_sampler_out=initial_sample_file,
        num_iterations=10,
        cycle_entry_point="sampler",  # This will be ignored!
        cycle_exit_point="sampler",
    )

    al_handler.run()


class Test_process_progress:
    def test_fully_completed(self):
        progress = [
            dict(
                steps=[0, 0, 0],
                sampler_output=["collected_0"],
                evaluator_output=["db_00", "db_01"],
                trainer_output=["model_00", "model_01"],
            ),
            dict(
                steps=[0, 0, 0],
                sampler_output=["collected_1"],
                evaluator_output=["db_10"],
                trainer_output=["model_10"],
            ),
        ]

        (
            iteration_idx,
            evaluator_input,
            trainer_input,
            sampler_input,
            progress,
        ) = process_progress(progress, 10, False, True, False)

        assert iteration_idx == 2
        assert len(evaluator_input) == 0
        assert len(trainer_input) == 3
        assert sampler_input == [Path("model_10")]
        assert len(progress) == 3
        assert list(progress[-1].keys()) == ["steps"]

    def test_sampler_completed(self):
        progress = [
            dict(
                steps=[0, 0, 0],
                sampler_output=["collected_0"],
                evaluator_output=["db_0"],
                trainer_output=["pickle_0"],
            ),
            dict(
                steps=[0, 1, 1],
                sampler_output=["collected_1"],
            ),
        ]
        (
            iteration_idx,
            evaluator_input,
            trainer_input,
            sampler_input,
            progress,
        ) = process_progress(progress, 10, False, False, False)

        assert iteration_idx == 1
        assert evaluator_input == [Path("collected_1")]
        assert len(trainer_input) == 0
        assert len(sampler_input) == 0
        assert len(progress) == 2
        assert list(progress[-1].keys()) == ["steps", "sampler_output"]

    def test_sampler_evaluator_completed(self):
        progress = [
            dict(
                steps=[0, 0, 0],
                sampler_output=["collected_0"],
                evaluator_output=["db_0"],
                trainer_output=["pickle_0"],
            ),
            dict(
                steps=[0, 0, 1],
                trainer_output=["collected_1"],
                evaluator_output=["db_1"],
            ),
        ]

        (
            iteration_idx,
            evaluator_input,
            trainer_input,
            sampler_input,
            progress,
        ) = process_progress(progress, 10, False, False, True)

        assert iteration_idx == 1
        assert len(evaluator_input) == 0
        assert trainer_input == [Path("db_1")]
        assert sampler_input == [Path("pickle_0")]
        assert len(progress) == 2
        assert list(progress[-1].keys()) == [
            "steps",
            "trainer_output",
            "evaluator_output",
        ]

    def test_none_completed(self):
        progress = [
            dict(
                steps=[0, 0, 0],
                sampler_output=["collected_0"],
                evaluator_output=["db_0"],
                trainer_output=["pickle_0"],
            ),
            dict(
                steps=[1, 1, 1],
            ),
        ]

        (
            iteration_idx,
            evaluator_input,
            trainer_input,
            sampler_input,
            progress,
        ) = process_progress(progress, 10, False, False, False)

        assert iteration_idx == 1
        assert sampler_input == [Path("pickle_0")]
        assert len(evaluator_input) == 0
        assert len(trainer_input) == 0
        assert len(progress) == 2
        assert list(progress[-1].keys()) == ["steps"]

    def test_single_sampler_completed(self):
        progress = [
            dict(
                steps=[0, 1, 1],
                sampler_output=["collected_0"],
            ),
        ]

        (
            iteration_idx,
            evaluator_input,
            trainer_input,
            sampler_input,
            progress,
        ) = process_progress(progress, 10, False, False, False)

        assert iteration_idx == 0
        assert evaluator_input == [Path("collected_0")]
        assert len(trainer_input) == 0
        assert len(sampler_input) == 0
        assert len(progress) == 1
        assert list(progress[-1].keys()) == ["steps", "sampler_output"]

    def test_single_sampler_evaluator_completed(self):
        progress = [
            dict(
                steps=[0, 0, 1],
                sampler_output=["collected_0"],
                evaluator_output=["db_0"],
            ),
        ]

        (
            iteration_idx,
            evaluator_input,
            trainer_input,
            sampler_input,
            progress,
        ) = process_progress(progress, 10, True, False, False)

        assert iteration_idx == 0
        assert evaluator_input == [Path("collected_0")]
        assert trainer_input == [Path("db_0")]
        assert sampler_input == []
        assert len(progress) == 1
        assert list(progress[-1].keys()) == [
            "steps",
            "sampler_output",
            "evaluator_output",
        ]

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


# class Test_process_progress:
#     def test_fully_completed(self):
#         progress_case0 = [
#             dict(
#                 steps=(0, 0, 0),
#                 collected_file="collected_0",
#                 db_file="db_0",
#                 pickle_file="pickle_0",
#             ),
#             dict(
#                 steps=(0, 0, 0),
#                 collected_file="collected_1",
#                 db_file="db_1",
#                 pickle_file="pickle_1",
#             ),
#         ]

#         (
#             iteration_idx,
#             collected_file,
#             db_files,
#             pickle_file,
#             progress,
#         ) = al_loop.process_progress(
#             progress_case0, num_iterations=10, db_files=["base_model"]
#         )

#         assert iteration_idx == 2
#         assert collected_file is None
#         assert len(db_files) == 3
#         assert pickle_file == "pickle_1"
#         assert len(progress) == 3
#         assert list(progress[-1].keys()) == ["steps"]

#     def test_ns_completed(self):
#         progress_case1 = [
#             dict(
#                 steps=(0, 0, 0),
#                 collected_file="collected_0",
#                 db_file="db_0",
#                 pickle_file="pickle_0",
#             ),
#             dict(
#                 steps=(0, 1, 1),
#                 collected_file="collected_1",
#             ),
#         ]
#         (
#             iteration_idx,
#             collected_file,
#             db_files,
#             pickle_file,
#             progress,
#         ) = al_loop.process_progress(
#             progress_case1, num_iterations=10, db_files=["base_model"]
#         )

#         assert iteration_idx == 1
#         assert collected_file == "collected_1"
#         assert len(db_files) == 2
#         assert pickle_file == "pickle_0"
#         assert len(progress) == 2
#         assert list(progress[-1].keys()) == ["steps", "collected_file"]

#     def test_ns_dft_completed(self):
#         progress_case2 = [
#             dict(
#                 steps=(0, 0, 0),
#                 collected_file="collected_0",
#                 db_file="db_0",
#                 pickle_file="pickle_0",
#             ),
#             dict(
#                 steps=(0, 0, 1),
#                 collected_file="collected_1",
#                 db_file="db_1",
#             ),
#         ]

#         (
#             iteration_idx,
#             collected_file,
#             db_files,
#             pickle_file,
#             progress,
#         ) = al_loop.process_progress(
#             progress_case2, num_iterations=10, db_files=["base_model"]
#         )

#         assert iteration_idx == 1
#         assert collected_file == "collected_1"
#         assert len(db_files) == 3
#         assert pickle_file == "pickle_0"
#         assert len(progress) == 2
#         assert list(progress[-1].keys()) == [
#             "steps",
#             "collected_file",
#             "db_file",
#         ]

#     def test_single_ns_completed(self):
#         progress_case2 = [
#             dict(
#                 steps=(0, 1, 1),
#                 collected_file="collected_0",
#             ),
#         ]

#         (
#             iteration_idx,
#             collected_file,
#             db_files,
#             pickle_file,
#             progress,
#         ) = al_loop.process_progress(
#             progress_case2, num_iterations=10, db_files=["base_model"]
#         )

#         assert iteration_idx == 0
#         assert collected_file == "collected_0"
#         assert len(db_files) == 1
#         assert pickle_file == "base_model"
#         assert len(progress) == 1
#         assert list(progress[-1].keys()) == ["steps", "collected_file"]

#     def test_single_ns_dft_completed(self):
#         progress_case2 = [
#             dict(
#                 steps=(0, 0, 1),
#                 collected_file="collected_0",
#                 db_file="db_0",
#             ),
#         ]

#         (
#             iteration_idx,
#             collected_file,
#             db_files,
#             pickle_file,
#             progress,
#         ) = al_loop.process_progress(
#             progress_case2, num_iterations=10, db_files=["base_model"]
#         )

#         assert iteration_idx == 0
#         assert collected_file == "collected_0"
#         assert len(db_files) == 2
#         assert pickle_file == "base_model"
#         assert len(progress) == 1
#         assert list(progress[-1].keys()) == [
#             "steps",
#             "collected_file",
#             "db_file",
#         ]

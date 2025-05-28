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


def get_dummy_configs():
    np.random.seed(0)
    n_configs = 10
    at0 = Atoms("Si3", cell=np.eye(3) * 7, pbc=True)
    ats = []
    for idx in range(n_configs):
        at = at0.copy()
        at.set_positions(np.random.random(size=(3, 3)) * 7)
        at.set_cell(at.cell[...] * np.random.random(size=(3, 3)))
        at.info["shard_idx"] = idx
        ats.append(at)
    return ats


dummy_configs = get_dummy_configs()


class SamplerMock(Sampler):
    def update(self, calcdir, model_file, **kwargs):
        assert os.path.exists(model_file.absolute())
        os.makedirs(Path(calcdir) / "sampler_dir")

    def get_sampled_configs(self):
        return dummy_configs


class EvaluatorMock(GroundTruthEvaluator):
    def update(self, calcdir, collected_file, **kwargs):
        assert os.path.exists(collected_file.absolute())
        os.makedirs(Path(calcdir) / "evaluator_dir")

    def get_evaluated_configs(self):
        return dummy_configs


class TrainerMock(ModelTrainer):
    def __init__(self, path):
        self.model_file = Path(path) / "model.pkl"
        with open(self.model_file, "w") as f:
            f.write("keklelsos")

    def update(self, calcdir, db_files, **kwargs):
        for file in db_files:
            assert os.path.exists(file.absolute())
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

    sampler = SamplerMock()
    evaluator = EvaluatorMock()
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
        initial_model_file=initial_model_file,
        initial_db_file=initial_db_file,
        initial_sample_file=initial_sample_file,
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

    sampler = SamplerMock()
    evaluator = EvaluatorMock()
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
        initial_model_file=initial_model_file,
        initial_db_file=initial_db_file,
        initial_sample_file=initial_sample_file,
        num_iterations=5,
        cycle_exit_point=cycle_exit_point,
    )

    al_handler.run()

    al_handler = ActiveLearningHandler(
        sampler,
        evaluator,
        trainer,
        calchome=local_dir,
        initial_model_file=initial_model_file,
        initial_db_file=initial_db_file,
        initial_sample_file=initial_sample_file,
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

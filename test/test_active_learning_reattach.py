"""Reattaching to an in-flight job across a simulated process restart.

If the *driving process* dies (dropped SSH session, killed local process)
after an actor has submitted a job but before it collected results,
restarting the AL loop script must not resubmit a duplicate -- it should
reattach via the manifest that submission already wrote (see
``DetachableTaskMixin`` / ``pending_manifest`` /
``ActiveLearningHandler._reattach_if_pending`` /
``StagedActiveLearningHandler._reattach_if_pending``).

Exercised end-to-end using a real ``MaceModelTrainer`` over the scheduler-
free ``direct`` executor -- no SLURM/mace stack needed, since the fake
payload script just writes a marker file. The "crash" is simulated by
detaching the job (``submit_detached()``) instead of letting
``get_trained_model()`` run its normal blocking submit/wait/collect, then
raising -- exactly what an actual dropped connection would look like from
the loop's perspective: the job is running remotely, but nothing local
knows it succeeded.
"""

import os
from pathlib import Path

import pytest

from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.core.manifest import TaskHandle
from remote_handler.workflows.active_learning_base import (
    ActiveLearningHandler,
    GroundTruthEvaluator,
    Sampler,
)
from remote_handler.workflows.staged_active_learning import (
    Stage,
    StagedActiveLearningHandler,
)
from remote_handler.workflows.trainers import MaceModelTrainer

FAKE_TRAIN_SCRIPT = """
import json, sys, pickle
from pathlib import Path

calc = Path(sys.argv[1])
data = json.load(open(calc / "input_train.json"))
models_dir = calc / "models"
models_dir.mkdir(parents=True, exist_ok=True)
with open(models_dir / f"{data['model_name']}.model", "wb") as f:
    pickle.dump({"seed": data["seed"]}, f)
(calc / "is_finished.info").write_text("done")
"""


class _NoopSampler(Sampler):
    """Never exercised (cycle_entry_point/exit_point="training", 1
    iteration) -- only needed to satisfy ActiveLearningHandler's contract."""

    def update(self, calcdir, sampler_input, **kwargs):
        pass

    def get_sampled_configs(self):
        return Path("unused")


class _NoopEvaluator(GroundTruthEvaluator):
    def update(self, calcdir, evaluator_input, **kwargs):
        pass

    def get_evaluated_configs(self):
        return Path("unused")


def _direct_config(savepath):
    return ClusterConfig(
        remote_machine="local",
        partition="p",
        conda_env="",
        conda_source="",
        savepath=str(savepath),
        template="unused.sh.j2",
        executor="direct",
    )


def _make_trainer(script, config_file, remote):
    return MaceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_direct_config(remote),
        config_file=str(config_file),
        training_script=str(script),
        remote_mace_train_runner="unused_run_train.py",
        num_models=1,
        loop_time=0.2,
    )


def _crash_after_submit(trainer):
    """Wrap trainer.update so it detaches (submits without waiting) and
    then raises, in place of the real get_trained_model()'s blocking
    submit/wait/collect -- simulating the driving process dying right
    after submission."""
    real_update = trainer.update

    def update_then_crash(calcdir, trainer_input, **kwargs):
        real_update(calcdir, trainer_input, **kwargs)
        trainer.training_task.submit_detached()
        raise RuntimeError("simulated crash: connection dropped")

    trainer.update = update_then_crash


def test_active_learning_handler_reattaches_after_simulated_crash(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN_SCRIPT)
    config_file = tmp_path / "mace.yaml"
    config_file.write_text("dummy: true")
    db_file = tmp_path / "data.extxyz"
    db_file.write_text("dummy db")
    calchome = tmp_path / "al"

    def build_handler(trainer):
        return ActiveLearningHandler(
            sampler=_NoopSampler(),
            evaluator=_NoopEvaluator(),
            trainer=trainer,
            calchome=calchome,
            num_iterations=1,
            cycle_entry_point="training",
            cycle_exit_point="training",
            initial_evaluator_out=db_file,
        )

    # --- "process 1": submits the trainer's job, then "crashes" before
    # collecting.
    trainer1 = _make_trainer(script, config_file, remote)
    _crash_after_submit(trainer1)
    handler1 = build_handler(trainer1)

    with pytest.raises(RuntimeError, match="simulated crash"):
        handler1.run()

    # AL_progress.json was never updated for this step -- still pending.
    assert handler1.progress[0]["steps"] == [0, 0, 1]

    # Give the detached job (a real, independently-running subprocess) a
    # chance to finish before "process 2" starts.
    manifest_path = trainer1.pending_manifest(handler1.current_iter_dir)
    assert manifest_path is not None and manifest_path.exists()
    TaskHandle.from_file(manifest_path).wait()

    # --- "process 2": a fresh script restart -- a brand new trainer
    # object, collect_only defaulting to False, exactly like a real user
    # script that knows nothing about the crash.
    trainer2 = _make_trainer(script, config_file, remote)
    assert trainer2.collect_only is False
    handler2 = build_handler(trainer2)

    handler2.run()

    # Reattached, not resubmitted -- and restored afterward rather than
    # left stuck in collect_only mode.
    assert trainer2.collect_only is False
    assert handler2.progress[0]["steps"] == [0, 0, 0]
    model_rel = handler2.progress[0]["trainer_output"][0]
    assert (handler2.calchome / model_rel).exists()


def test_staged_active_learning_handler_reattaches_after_simulated_crash(
    tmp_path,
):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN_SCRIPT)
    config_file = tmp_path / "mace.yaml"
    config_file.write_text("dummy: true")
    db_file = tmp_path / "data.extxyz"
    db_file.write_text("dummy db")
    calchome = tmp_path / "al"

    def build_handler(trainer):
        return StagedActiveLearningHandler(
            stages=[
                Stage(
                    name="trainer",
                    actor=trainer,
                    initial_input=db_file,
                )
            ],
            calchome=calchome,
            num_iterations=1,
        )

    trainer1 = _make_trainer(script, config_file, remote)
    _crash_after_submit(trainer1)
    handler1 = build_handler(trainer1)

    with pytest.raises(RuntimeError, match="simulated crash"):
        handler1.run()

    assert 0 not in handler1.outputs.get("trainer", {})

    manifest_path = trainer1.pending_manifest(calchome / "0")
    assert manifest_path is not None and manifest_path.exists()
    TaskHandle.from_file(manifest_path).wait()

    trainer2 = _make_trainer(script, config_file, remote)
    assert trainer2.collect_only is False
    handler2 = build_handler(trainer2)

    handler2.run()

    assert trainer2.collect_only is False
    model_rel = handler2.outputs["trainer"][0][0]
    assert (handler2.calchome / model_rel).exists()

"""Integration tests for grace_training.py script and GraceModelTrainer.

Self-skips cleanly (via ``grace_available()``) wherever gracemaker (the
``tensorpotential`` package) isn't installed. The trainer round trip uses
the scheduler-free ``direct`` executor, so no local SLURM is needed either.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from _capabilities import grace_available, grace_train_runner_script
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.workflows.trainers import GraceModelTrainer

# Minimal but complete gracemaker config for a from-scratch fit.
#
# - ``fit.loss`` is not optional: gracemaker's ``load_and_prepare_datasets``
#   does a bare ``fit_config["loss"]`` dict access (no ``.get``) before
#   anything else, so a fit section without it dies with
#   ``KeyError: 'loss'`` before training even starts. ``type: square``
#   (rather than the wizard's default ``"huber"``) needs no extra parameters
#   (huber needs a ``delta``).
# - ``potential.preset`` must be a name actually registered in
#   ``tensorpotential.potentials.REGISTERED_PRESETS`` -- "GRACE_1LAYER" (no
#   suffix) isn't one; the current default is "GRACE_1LAYER_latest".
# - ``fit.optimizer: L-BFGS-B`` sidesteps another required-but-undocumented
#   key: the Adam path needs ``fit.scheduler_params`` (or the legacy
#   ``learning_rate_reduction``), a bare dict access with no default: quasi-
#   Newton optimizers need no LR scheduler at all, so this tiny/fast fit
#   avoids that config entirely.
_GRACE_CONFIG = (
    "cutoff: 5.0\n"
    "potential:\n"
    "  preset: GRACE_1LAYER_latest\n"
    "fit:\n"
    "  loss: {\n"
    "    energy: {weight: 1.0, type: square},\n"
    "    forces: {weight: 1.0, type: square},\n"
    "  }\n"
    "  optimizer: L-BFGS-B\n"
    "  opt_params: {maxcor: 100, maxls: 20, gtol: 1.e-8, iprint: -1}\n"
    "  maxiter: 2\n"
    "  batch_size: 4\n"
)


@pytest.fixture(autouse=True)
def _force_cpu_for_gracemaker(monkeypatch):
    """Force CPU-only for every test in this module.

    gracemaker's TensorFlow build may lack precompiled CUDA kernels for very
    new GPU compute capabilities, raising ``CUDA_ERROR_INVALID_PTX``/
    ``CUDA_ERROR_INVALID_HANDLE`` -- an environment/driver mismatch, not a
    remote_handler bug. This tiny 20-structure dataset gains nothing from a
    GPU anyway, so run CPU-only for a deterministic, portable test. Both the
    raw ``subprocess.run`` call and the ``direct`` executor's launch (which
    also just inherits ``os.environ``) pick this up.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="session")
def integration_temp_dir(data_dir):
    temp_dir = data_dir / "temp" / "integration" / "grace_training"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


@pytest.fixture
def test_atoms_database():
    atoms_list = []
    for i in range(20):
        atoms = Atoms(
            "Cu2",
            positions=[[0, 0, 0], [2.55 + i * 0.01, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        atoms.info["REF_energy"] = -1.0 - i * 0.01
        atoms.arrays["REF_forces"] = np.array([[0, 0, 0.1], [0, 0, -0.1]])
        atoms_list.append(atoms)
    return atoms_list


@pytest.mark.integration
@pytest.mark.skipif(
    not grace_available(),
    reason="GRACE (tensorpotential) not installed",
)
class TestGraceTrainingIntegration:
    """Integration tests for the raw grace_training.py script."""

    def test_script_with_multiple_databases(
        self, integration_temp_dir, test_atoms_database
    ):
        root_dir = integration_temp_dir / "test_multiple_db"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        train_runner = grace_train_runner_script()
        assert train_runner is not None

        db1 = root_dir / "db1.xyz"
        db2 = root_dir / "db2.xyz"
        ase.io.write(db1, test_atoms_database[:10])
        ase.io.write(db2, test_atoms_database[10:])

        config_file = root_dir / "grace_config.yaml"
        config_file.write_text(_GRACE_CONFIG)

        calc_dir = root_dir / "calc"
        calc_dir.mkdir()

        input_json = calc_dir / "input_train.json"
        input_data = {
            "database_files": [str(db1), str(db2)],
            "train_runner": train_runner,
            "config_file": str(config_file),
            "model_name": "test",
            "seed": 42,
            "validation_fraction": 0.2,
        }
        input_json.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.grace.grace_training",
                str(calc_dir),
            ],
            capture_output=True,
            text=True,
            cwd=root_dir,
            check=True,
        )
        print(result.stderr)

        assert (calc_dir / "combined_db.extxyz").exists()
        combined = ase.io.read(calc_dir / "combined_db.extxyz", index=":")
        assert len(combined) == 20
        assert (calc_dir / "models" / "test" / "model.yaml").exists()
        assert (calc_dir / "is_finished.info").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not grace_available(),
    reason="GRACE (tensorpotential) not installed",
)
class TestGraceModelTrainerIntegration:
    """Integration tests for GraceModelTrainer workflow class.

    Uses the scheduler-free ``direct`` executor (see ``examples/local_emt_AL``)
    rather than SLURM: this round trip needs only gracemaker itself, not a
    local scheduler -- so it actually runs in CI/most dev environments once
    the ``grace`` extra is installed, instead of always self-skipping.
    """

    def test_full_local_training(
        self, integration_temp_dir, test_atoms_database, local_fs
    ):
        """Test GraceModelTrainer with actual local gracemaker training."""
        root_dir = integration_temp_dir / "trainer_full"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        train_runner = grace_train_runner_script()

        db_file = root_dir / "training_data.xyz"
        ase.io.write(db_file, test_atoms_database)

        template = root_dir / "grace_config.yaml"
        template.write_text(_GRACE_CONFIG)

        local_cluster_config = ClusterConfig(
            remote_machine="local",
            partition="none",
            conda_env="",
            conda_source="",
            savepath=str(root_dir / "remote_savepath"),
            template="unused.sh.j2",
            executor="direct",
        )

        trainer = GraceModelTrainer(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            config_file=template,
            remote_train_runner=train_runner,
            num_models=2,
            stub=False,
            loop_time=10,
        )

        calcdir = root_dir / "iteration_0"
        calcdir.mkdir()
        trainer.update(calcdir, db_file)

        model_files = trainer.get_trained_model()

        assert len(model_files) == 2
        for model_file in model_files:
            assert Path(model_file).exists()

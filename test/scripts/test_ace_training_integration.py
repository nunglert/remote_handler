"""Integration tests for ace_training.py script and AceModelTrainer.

Self-skips cleanly (via ``ace_available()``) wherever pacemaker (the
``python-ace``/``pyace`` package) isn't installed -- it ships compiled C++
extensions and is typically installed via conda-forge or built from source
(see the ``ace`` extra in ``pyproject.toml``), so this suite rarely runs
outside a dedicated ACE dev environment. The trainer round trip uses the
scheduler-free ``direct`` executor, so no local SLURM is needed either.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from _capabilities import ace_available, ace_train_runner_script
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.workflows.trainers import AceModelTrainer

# Minimal but complete pacemaker config for a from-scratch fit, adapted from
# the exact schema pacemaker itself ships (pyace/data/input_template.yaml)
# rather than guessed: pacemaker's "new" multi-species potential format is
# only used when the potential section has *all three* of "elements",
# "bonds" and "functions" -- with any one missing (e.g. no "functions") it
# silently falls back to the old single-species format, which reads a
# singular potential_config["element"], raising `KeyError: 'element'` for a
# config authored in the (very similarly-shaped) multi-species style. Field
# names inside "bonds"/"functions" (radbase: SBessel, nradmax_by_orders/
# lmax_by_orders) and "fit.loss" (kappa/L1_coeffs/L2_coeffs -- pacemaker's
# own scipy-optimizer loss, unrelated to MACE/GRACE's energy/forces weight
# dicts) are equally version-specific; copy from the shipped template rather
# than another ACE tool's convention.
_ACE_CONFIG = (
    "potential:\n"
    "  deltaSplineBins: 0.001\n"
    "  elements: [Cu]\n"
    "  embeddings:\n"
    "    ALL: {npot: FinnisSinclairShiftedScaled, "
    "fs_parameters: [1, 1, 1, 0.5], ndensity: 2}\n"
    "  bonds:\n"
    "    ALL: {radbase: SBessel, radparameters: [5.25], rcut: 5.0, "
    "dcut: 0.01, NameOfCutoffFunction: cos}\n"
    "  functions:\n"
    "    UNARY: {nradmax_by_orders: [2, 1], lmax_by_orders: [0, 0]}\n"
    "fit:\n"
    "  loss: {kappa: 0.3, L1_coeffs: 1.0e-8, L2_coeffs: 1.0e-8}\n"
    "  optimizer: BFGS\n"
    "  maxiter: 5\n"
    "  repulsion: auto\n"
)


@pytest.fixture(scope="session")
def integration_temp_dir(data_dir):
    temp_dir = data_dir / "temp" / "integration" / "ace_training"
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
    not ace_available(),
    reason="ACE (pyace/pacemaker) not installed",
)
class TestAceTrainingIntegration:
    """Integration tests for the raw ace_training.py script."""

    def test_script_with_multiple_databases(
        self, integration_temp_dir, test_atoms_database
    ):
        root_dir = integration_temp_dir / "test_multiple_db"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        train_runner = ace_train_runner_script()
        assert train_runner is not None

        db1 = root_dir / "db1.xyz"
        db2 = root_dir / "db2.xyz"
        ase.io.write(db1, test_atoms_database[:10])
        ase.io.write(db2, test_atoms_database[10:])

        config_file = root_dir / "ace_config.yaml"
        config_file.write_text(_ACE_CONFIG)

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
                "remote_handler.scripts.ace.ace_training",
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
        assert (calc_dir / "models" / "test.yaml").exists()
        assert (calc_dir / "is_finished.info").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not ace_available(),
    reason="ACE (pyace/pacemaker) not installed",
)
class TestAceModelTrainerIntegration:
    """Integration tests for AceModelTrainer workflow class.

    Uses the scheduler-free ``direct`` executor (see ``examples/local_emt_AL``)
    rather than SLURM: pacemaker's fits are cheap enough to run inline, and
    this way the round trip needs only pacemaker itself, not a local
    scheduler -- so it actually runs in CI/most dev environments once the
    ``ace`` extra is installed, instead of always self-skipping.
    """

    def test_full_local_training(
        self, integration_temp_dir, test_atoms_database, local_fs
    ):
        """Test AceModelTrainer with actual local pacemaker training."""
        root_dir = integration_temp_dir / "trainer_full"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        train_runner = ace_train_runner_script()

        db_file = root_dir / "training_data.xyz"
        ase.io.write(db_file, test_atoms_database)

        template = root_dir / "ace_config.yaml"
        template.write_text(_ACE_CONFIG)

        local_cluster_config = ClusterConfig(
            remote_machine="local",
            partition="none",
            conda_env="",
            conda_source="",
            savepath=str(root_dir / "remote_savepath"),
            template="unused.sh.j2",
            executor="direct",
        )

        trainer = AceModelTrainer(
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

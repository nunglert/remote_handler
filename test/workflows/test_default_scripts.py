"""Each actor defaults its remote ``script`` to the shipped payload.

The sampling/evaluation/training payloads live in the package's ``scripts/``
tree and never change per-run, so the path is optional: when omitted the actor
resolves it to the bundled file. These tests construct each actor without a
script and assert the resolved path points at the expected shipped payload.
"""

from pathlib import Path

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows.evaluators import ORCAEvaluator, VASPEvaluator
from remote_handler.workflows.samplers import (
    _SCRIPTS_DIR,
    CMAESSampler,
    TorchSimSampler,
)
from remote_handler.workflows.trainers import (
    MaceFinetuningTrainer,
    MaceModelTrainer,
    NeuralilILTrainer,
)


def _cfg(tmp_path):
    return ClusterConfig(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath=str(tmp_path),
        template="t.j2",
    )


def _rel(script) -> str:
    """Path of the resolved script relative to the shipped scripts/ dir."""
    return Path(script).resolve().relative_to(_SCRIPTS_DIR).as_posix()


def test_cmaes_default_script(tmp_path):
    s = CMAESSampler(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        founder_path="POSCAR",
    )
    assert _rel(s.sampling_script) == "clinamen2/cmaes_sampling.py"


def test_torch_sim_default_script(tmp_path):
    s = TorchSimSampler(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        founder_path="founder.xyz",
        ensemble="nvt",
        temperatures=[300.0],
        target_temperature=300.0,
    )
    assert _rel(s.sampling_script) == "torch_sim/torch_sim_sampling.py"


def test_vasp_default_script(tmp_path):
    e = VASPEvaluator(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        num_parallel=2,
        incar_kwargs={"encut": 300},
    )
    assert _rel(e.singlepoint_script) == "dft_singlepoint.py"


def test_orca_default_script(tmp_path):
    e = ORCAEvaluator(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        num_parallel=2,
        orcasimpleinput="wB97X 6-31G(d) ENGRAD",
    )
    assert _rel(e.singlepoint_script) == "orca_singlepoint.py"


def test_mace_default_script(tmp_path):
    t = MaceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="mace.yaml",
        remote_mace_train_runner="run_train.py",
        stub=True,
    )
    assert _rel(t.training_script) == "mace/mace_training.py"


def test_mace_finetuning_default_script(tmp_path):
    t = MaceFinetuningTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="mace.yaml",
        foundation_model="foundation.model",
        remote_mace_train_runner="run_train.py",
        stub=True,
    )
    assert _rel(t.training_script) == "mace/mace_finetuning.py"


def test_neuralil_default_script(tmp_path):
    t = NeuralilILTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        stub=True,
    )
    assert _rel(t.training_script) == "neuralil/neuralil_training.py"


def test_explicit_script_overrides_default(tmp_path):
    """Passing a script still wins -- the default only fills the gap."""
    s = CMAESSampler(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        founder_path="POSCAR",
        sampling_script="/custom/my_sampling.py",
    )
    assert str(s.sampling_script) == "/custom/my_sampling.py"

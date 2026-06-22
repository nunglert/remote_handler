"""INCAR resolution: dict and/or YAML file for the VASP DFT config."""

import yaml

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows.dft_runner import (
    _DEFAULT_INCAR_KWARGS,
    resolve_incar_kwargs,
)
from remote_handler.workflows.evaluators import VASPEvaluator


def test_defaults_when_nothing_given():
    assert resolve_incar_kwargs() == _DEFAULT_INCAR_KWARGS


def test_dict_used_as_is():
    # Backward compatibility: a dict replaces the defaults (not merged).
    d = {"xc": "PBE", "encut": 400}
    assert resolve_incar_kwargs(incar_kwargs=d) == d


def test_yaml_file(tmp_path):
    f = tmp_path / "incar.yaml"
    f.write_text(yaml.safe_dump({"encut": 500, "ismear": 0}))
    assert resolve_incar_kwargs(incar_config_file=f) == {
        "encut": 500,
        "ismear": 0,
    }


def test_yaml_base_with_dict_overrides(tmp_path):
    f = tmp_path / "incar.yaml"
    f.write_text(yaml.safe_dump({"encut": 500, "ismear": 0}))
    out = resolve_incar_kwargs(
        incar_kwargs={"encut": 600}, incar_config_file=f
    )
    assert out == {"encut": 600, "ismear": 0}


def test_vasp_evaluator_accepts_incar_config_file(tmp_path):
    f = tmp_path / "incar.yaml"
    f.write_text(yaml.safe_dump({"encut": 450, "xc": "PBE"}))
    cfg = ClusterConfig(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath=str(tmp_path),
        template="t.j2",
    )
    evaluator = VASPEvaluator(
        singlepoint_script="dft_singlepoint.py",
        remote_fs=LocalFileSystem(),
        cluster_config=cfg,
        num_parallel=1,
        incar_config_file=f,
    )
    assert evaluator.incar_kwargs == {"encut": 450, "xc": "PBE"}

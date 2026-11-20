"""Guard: the custom-processor example in the processing docs really works.

``docs/source/examples/custom_processor.py`` is literal-included by
``docs/source/processing.rst`` as *the* recipe for a downstream processor. This
runs it the way a cluster would: a shard directory with an
``input_process.json`` naming the step and its registration module, executed by
the shipped driver -- so a change to the plugin contract (registration,
arg validation, the driver's import step) that breaks the recipe fails here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read, write

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing import REGISTRY, RDFProcessor
from remote_handler.processing.driver import main as run_driver

EXAMPLES_DIR = (
    Path(__file__).resolve().parent.parent / "docs" / "source" / "examples"
)
MODULE = "custom_processor"


@pytest.fixture
def custom_processor(monkeypatch):
    """Import the example as a downstream module, cleaning up after."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop(MODULE, None)
    module = __import__(MODULE)
    yield module
    sys.modules.pop(MODULE, None)
    REGISTRY.pop("coordination", None)


def _shard_dir(tmp_path, processors, savepath="/remote/work"):
    """Write a shard directory as RemoteTrajectoryProcessor would."""
    atoms = [bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2) for _ in range(3)]
    write(tmp_path / "opt.extxyz", atoms)
    cfg = {
        "tag": "proc",
        "DATABASE": "opt.extxyz",
        "STUB": False,
        "steps": [p.name for p in processors],
        "registration_modules": sorted(
            {
                p.registration_module
                for p in processors
                if p.registration_module
            }
        ),
        "args": {p.name: p.build_args(savepath) for p in processors},
    }
    (tmp_path / "input_process.json").write_text(json.dumps(cfg))
    return tmp_path


def test_registration_module_defaults_to_defining_module(custom_processor):
    assert custom_processor.CoordinationProcessor.registration_module == MODULE


def test_driver_runs_custom_processor_after_builtin(
    tmp_path, custom_processor
):
    processors = [
        RDFProcessor(cutoff=4.0, nbins=50),
        custom_processor.CoordinationProcessor(cutoff=2.8),
    ]
    calcdir = _shard_dir(tmp_path, processors)
    # As on the cluster: the driver must import the module itself.
    sys.modules.pop(MODULE, None)
    REGISTRY.pop("coordination", None)

    run_driver(calcdir)

    out = read(calcdir / "proc_opt.extxyz", index=":")
    assert len(out) == 3
    for at in out:
        # fcc nearest-neighbour distance 3.6/sqrt(2) = 2.55 < 2.8: CN = 12.
        assert np.all(at.arrays["coordination"] == 12)
        assert at.info["coordination_mean"] == pytest.approx(12.0)
        assert "rdf_total" in at.info
        assert at.info["is_finished"]


def test_driver_rejects_bad_custom_args_before_compute(
    tmp_path, custom_processor
):
    calcdir = _shard_dir(
        tmp_path, [custom_processor.CoordinationProcessor(cutoff=-1.0)]
    )
    with pytest.raises(ScriptInputError, match="cutoff"):
        run_driver(calcdir)
    assert not (calcdir / "proc_opt.extxyz").exists()

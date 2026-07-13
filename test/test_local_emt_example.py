"""Execute the ``examples/local_emt_AL`` loop end-to-end, no cluster.

The example is a full active-learning cycle (ASE-MD sample -> stride select ->
EMT label -> Morse fit) that runs entirely over the ``direct`` executor on a
``LocalFileSystem``. Because it needs no scheduler and no ML stack -- only
``ase`` (which hard-requires ``scipy``) -- it runs in the default test tier and
in CI, the same way ``test_emt_evaluator`` does.

This both exercises the whole task/orchestration machinery with nothing mocked
*and* guards the shipped example against API drift (a renamed class or removed
kwarg would fail the import / run here).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")  # example fits with scipy (pulled in by ase)

EXAMPLE_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "local_emt_AL"
)


@pytest.fixture
def direct_python_on_path(monkeypatch):
    """Make the direct executor's bare ``python`` an interpreter with deps.

    The ``direct`` launch script runs ``python`` with no conda activation, so
    it resolves via PATH. Prepend the running interpreter's directory -- that
    interpreter has ``ase`` + ``scipy`` + ``remote_handler`` installed.
    """
    bindir = str(Path(sys.executable).parent)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([bindir, os.environ.get("PATH", "")])
    )


def _load_run_al():
    """Import the example's ``run_al`` module (with its dir on sys.path)."""
    if str(EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_DIR))  # so ``import actors`` resolves
    spec = importlib.util.spec_from_file_location(
        "local_emt_run_al", EXAMPLE_DIR / "run_al.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_emt_example_runs(tmp_path, direct_python_on_path):
    run_al = _load_run_al()

    calchome = run_al.main(run_dir=tmp_path, num_iterations=2)

    # One fitted Morse model per iteration.
    models = sorted(Path(calchome).glob("trainer_output/trainer_it*_0.json"))
    assert len(models) == 2

    fits = [json.loads(m.read_text()) for m in models]
    for fit in fits:
        assert fit["type"] == "morse"
        assert 1e-3 < fit["D"] < 10.0
        assert 1.5 < fit["r0"] < 3.5

    # Training set accumulates across iterations (combine_evaluator_outputs).
    assert fits[1]["n_train"] > fits[0]["n_train"]

    # The closed loop: iteration 0 bootstraps under EMT, iteration 1 samples
    # under the model fitted in iteration 0.
    it0_model = json.loads(
        (Path(calchome) / "0" / "sampler_input.json").read_text()
    )["model"]["type"]
    it1_model = json.loads(
        (Path(calchome) / "1" / "sampler_input.json").read_text()
    )["model"]["type"]
    assert it0_model == "emt"
    assert it1_model == "morse"

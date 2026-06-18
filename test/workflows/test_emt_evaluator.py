"""End-to-end execution of EMTEvaluator over LocalFileSystem + direct.

Unlike the VASP/ORCA evaluators (licensed binaries, ``remote`` tier), the EMT
evaluator uses a pure-ASE payload, so the whole ground-truth path -- ship the
input bundle, run the payload as a real detached subprocess via the ``direct``
executor, poll the sentinel, collect, and REF-label -- runs with no cluster and
no ML stack. This is the evaluator that genuinely executes in CI.
"""

import os
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows import EMTEvaluator


@pytest.fixture
def direct_python_on_path(monkeypatch):
    """Make the direct executor's bare ``python`` an interpreter with deps.

    The ``direct`` launch script invokes ``python`` with no conda activation
    (conda_env=""), so it resolves via PATH. Prepend the running interpreter's
    directory -- that interpreter is, by construction, the one running this
    test, hence has ``ase`` + ``remote_handler`` installed.
    """
    bindir = str(Path(sys.executable).parent)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([bindir, os.environ.get("PATH", "")])
    )


def test_emt_evaluator_runs_locally(tmp_path, direct_python_on_path):
    # Two tiny Cu structures (EMT supports Cu).
    configs = [
        Atoms(
            "Cu2",
            positions=[[0, 0, 0], [2.55, 0, 0]],
            cell=[8, 8, 8],
            pbc=True,
        ),
        Atoms("Cu", positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=True),
    ]
    traj = tmp_path / "configs.extxyz"
    ase.io.write(traj, configs)

    savepath = tmp_path / "save"
    savepath.mkdir()

    cluster_config = ClusterConfig(
        remote_machine="local",
        partition="p",
        conda_env="",  # no activation -> ambient python from PATH
        conda_source="",
        savepath=str(savepath),
        template="unused.sh.j2",
        executor="direct",
    )

    evaluator = EMTEvaluator(
        remote_fs=LocalFileSystem(),
        cluster_config=cluster_config,
        num_parallel=2,
        remote_savepath=savepath,
        loop_time=0.5,
    )

    calcdir = tmp_path / "iter0"
    calcdir.mkdir()

    evaluator.update(calcdir, traj)
    output_path = evaluator.get_evaluated_configs()

    labelled = ase.io.read(output_path, index=":")
    assert len(labelled) == 2
    for at in labelled:
        assert "REF_energy" in at.info
        assert np.isfinite(at.info["REF_energy"])
        assert at.arrays["REF_forces"].shape == (len(at), 3)


def test_emt_evaluator_stub(tmp_path):
    """Stub mode short-circuits compute and still produces labelled output."""
    cluster_config = ClusterConfig(
        remote_machine="local",
        partition="p",
        conda_env="",
        conda_source="",
        savepath=str(tmp_path),
        template="unused.sh.j2",
        executor="direct",
    )
    evaluator = EMTEvaluator(
        remote_fs=LocalFileSystem(),
        cluster_config=cluster_config,
        num_parallel=1,
        remote_savepath=tmp_path,
        stub=True,
    )
    calcdir = tmp_path / "iter0"
    calcdir.mkdir()
    evaluator.update(calcdir, tmp_path / "ignored.extxyz")
    output_path = evaluator.get_evaluated_configs()
    assert Path(output_path).exists()
    assert len(ase.io.read(output_path, index=":")) == 5

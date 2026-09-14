"""Example 3 -- Standalone submit/collect across two separate runs.

Long remote jobs do not need the launching process to stay alive: submit in
one invocation of this script, then come back later (a different process, a
different day) and collect. This shows the pattern for both a DFT array task
(:class:`~remote_handler.workflows.dft_runner.VASPHandler`, which exposes its
own ``collect_only=True`` constructor) and a MACE training array
(:class:`~remote_handler.workflows.trainers.MaceModelTrainer`, which
reconstructs its array task via
:meth:`~remote_handler.core.tasks.ArrayScriptTask.from_manifest` under the
hood).

Run ``python example_3_standalone.py submit`` to fire both jobs and exit
immediately, then ``python example_3_standalone.py collect`` (any time after,
even from a fresh shell) to check on them and fetch results once finished.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

import sys
from pathlib import Path

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.dft_runner import VASPHandler
from remote_handler.workflows.trainers import MaceModelTrainer

MACE_CALCDIR = Path("mace_run/0/training")


def submit(dft_kwargs: dict, mace_kwargs: dict) -> None:
    """Fire both array jobs and return at once -- no waiting."""
    dft_handler = VASPHandler(**dft_kwargs, collect_only=False)
    dft_manifest = dft_handler.detach()
    print(f"DFT array submitted; manifest at {dft_manifest}")

    mace_trainer = MaceModelTrainer(**mace_kwargs, collect_only=False)
    mace_trainer.update(MACE_CALCDIR, ["dft_labelled.extxyz"])
    mace_manifest = mace_trainer.training_task.submit_detached()
    print(f"MACE training submitted; manifest at {mace_manifest}")


def collect(dft_kwargs: dict, mace_kwargs: dict) -> None:
    """Reconstruct both tasks and collect whatever has finished."""
    dft_handler = VASPHandler(**dft_kwargs, collect_only=True)
    if dft_handler.finish_check():
        dft_handler.collect_and_harvest()
        print("DFT results collected.")
    else:
        print("DFT array still running.")

    mace_trainer = MaceModelTrainer(**mace_kwargs, collect_only=True)
    mace_trainer.update(MACE_CALCDIR, None)
    if mace_trainer.training_task.finish_check():
        model_paths = mace_trainer.get_trained_model()
        print(f"MACE models collected: {model_paths}")
    else:
        print("MACE training still running.")


if __name__ == "__main__":
    config = ClusterConfig.from_name("dirona")
    remote_fs = RemoteFileSystem(remote_alias="dirona")

    # Identical construction args for both runs -- only ``collect_only``
    # differs between the process that submits and the one that collects
    # later.
    dft_kwargs = dict(
        traj_files=["selected_structures.extxyz"],
        cluster_config=config,
        remote_fs=remote_fs,
        savepath="/scratch/me/vasp",
        num_parallel=10,
        n_processors=16,
        incar_kwargs={
            "xc": "PBE",
            "encut": 400,
            "ediff": 1e-6,
            "ismear": 0,
            "sigma": 0.05,
            "kpts": [3, 3, 1],
        },
    )
    mace_kwargs = dict(
        remote_fs=remote_fs,
        cluster_config=config,
        config_file="config/mace_config.yaml",
        num_models=3,
    )

    if sys.argv[1:] == ["submit"]:
        submit(dft_kwargs, mace_kwargs)
    elif sys.argv[1:] == ["collect"]:
        collect(dft_kwargs, mace_kwargs)
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} submit|collect")

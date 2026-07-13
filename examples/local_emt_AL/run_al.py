"""Self-contained active-learning loop that runs on *any* machine, no cluster.

A minimal but complete AL cycle for bulk Cu:

    sample (ASE MD)  ->  select (stride)  ->  label (EMT)  ->  train (Morse fit)

Every actor executes through the package's real task machinery, but over the
scheduler-free ``direct`` executor on a ``LocalFileSystem`` -- so there is no
SSH, no SLURM and no ML stack. Ground truth is ASE's EMT calculator (a shipped
preset); the "trained model" is a pairwise Morse potential fitted with scipy,
standing in for a heavy MLFF trainer. Dependencies: only ``ase``, ``numpy`` and
``scipy`` (the core install).

The point is orchestration, not accuracy: it demonstrates that *anything* can
be cast as a ``RemoteTask`` and that swapping the ``ClusterConfig`` is all it
takes to move the same loop onto a real cluster.

Run it from the environment the package is installed in (``direct`` launches a
bare ``python``, resolved from ``PATH``):

    python run_al.py
"""

from pathlib import Path

from actors import MorseMDSampler, MorseTrainer
from ase.build import bulk

from remote_handler import configure_logging
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows import EMTEvaluator
from remote_handler.workflows.selectors import StrideSelector
from remote_handler.workflows.staged_active_learning import (
    StagedActiveLearningHandler,
)

configure_logging()

HERE = Path(__file__).resolve().parent


def main(run_dir=None, num_iterations=3):
    """Run the local AL loop.

    Args:
        run_dir: Where to write the ``calchome/`` tree and scratch files.
            Defaults to ``al_run/`` next to this script. A test can point it
            at a tmp dir.
        num_iterations: Number of active-learning iterations.

    Returns:
        Path to the ``calchome/`` directory.
    """
    run_dir = Path(run_dir) if run_dir is not None else HERE / "al_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = run_dir / "scratch"
    scratch.mkdir(exist_ok=True)

    remote_fs = LocalFileSystem()

    # A local, scheduler-free "cluster": empty conda fields -> the ambient
    # python from PATH runs each payload. Swap executor="slurm" + a real
    # remote_machine/conda_env to move the identical loop onto a cluster.
    cluster_config = ClusterConfig(
        remote_machine="local",
        partition="none",
        conda_env="",
        conda_source="",
        savepath=str(scratch),
        template="unused.sh.j2",
        executor="direct",
    )

    # A small periodic Cu cell (EMT supports Cu); MD explores from here.
    seed_atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)

    sampler = MorseMDSampler(
        remote_fs=remote_fs,
        cluster_config=cluster_config,
        seed_atoms=seed_atoms,
        temperature_K=400.0,
        n_steps=200,
        sample_every=40,
    )
    selector = StrideSelector(n_configs=3)
    evaluator = EMTEvaluator(
        remote_fs=remote_fs,
        cluster_config=cluster_config,
        num_parallel=3,
        remote_savepath=scratch / "dft",
        loop_time=0.3,
    )
    trainer = MorseTrainer(remote_fs=remote_fs, cluster_config=cluster_config)

    handler = StagedActiveLearningHandler.from_actors(
        sampler=sampler,
        selector=selector,
        evaluator=evaluator,
        trainer=trainer,
        calchome=run_dir / "calchome",
        num_iterations=num_iterations,
        cycle_exit_point="training",  # every iteration ends with a fit
        initial_trainer_out=HERE / "inputs" / "seed_model.json",
        trainer_output_extension="json",
    )
    handler.run()

    calchome = run_dir / "calchome"
    print("\nActive learning finished. Fitted models per iteration:")
    for model in sorted(calchome.glob("trainer_output/trainer_it*_0.json")):
        print(f"  {model.name}: {model.read_text().strip()}")
    return calchome


if __name__ == "__main__":
    main()

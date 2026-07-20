"""Example-local AL actors: an MD sampler and a Morse trainer.

Both are thin wrappers that cast an ordinary Python payload as a
``SingleScriptTask`` and run it through whatever executor the ``ClusterConfig``
selects -- here the scheduler-free ``direct`` backend on a ``LocalFileSystem``,
so the whole loop runs on one machine with no cluster. Point the same
``ClusterConfig`` at a real cluster (``executor="slurm"``) and the identical
actor code submits to SLURM instead.

They implement the ``Sampler`` / ``ModelTrainer`` getter contract
(``get_sampled_configs`` / ``get_trained_model``) so they drop straight into
``StagedActiveLearningHandler.from_actors``.
"""

import json
import os
from pathlib import Path

from remote_handler.core.tasks import SingleScriptTask

_SCRIPTS = Path(__file__).resolve().parent / "scripts"


def _run_task(
    remote_fs,
    cluster_config,
    script,
    input_file,
    workdir,
    jobname,
    additional_files=(),
    loop_time=0.3,
):
    """Allocate + submit one ``SingleScriptTask`` and return its calcdir.

    Reads results in place (local disk), so no separate collection step is
    needed -- ``task.calcdir`` holds the payload's outputs once ``submit``
    returns.
    """
    remote_fs.makedirs(workdir, exist_ok=True)
    task = SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=Path(workdir) / "collected",
        cluster_config=cluster_config,
        remote_fs=remote_fs,
        remote_savepath=str(workdir),
        jobname=jobname,
        loop_time=loop_time,
        additional_files=list(additional_files),
        write_manifest_on_submit=False,
    )
    task.allocate_data()
    task.submit()
    if not task.finish_check():
        raise RuntimeError(
            f"payload {script} did not finish in {task.calcdir}"
        )
    return Path(task.calcdir)


class MorseMDSampler:
    """Run short ASE Langevin MD driven by the current model."""

    def __init__(
        self,
        remote_fs,
        cluster_config,
        seed_atoms,
        temperature_K=300.0,
        n_steps=200,
        sample_every=40,
        timestep_fs=1.0,
        rattle=0.05,
        loop_time=0.3,
    ):
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.seed_atoms = seed_atoms
        self.temperature_K = temperature_K
        self.n_steps = n_steps
        self.sample_every = sample_every
        self.timestep_fs = timestep_fs
        self.rattle = rattle
        self.loop_time = loop_time
        self._out = None

    def update(self, calcdir, sampler_input, **kwargs):
        calcdir = Path(calcdir)
        model_path = (
            sampler_input[0]
            if isinstance(sampler_input, (list, tuple))
            else sampler_input
        )
        model = json.loads(Path(model_path).read_text())

        at = self.seed_atoms
        payload_input = {
            "model": model,
            "structure": {
                "numbers": at.get_atomic_numbers().tolist(),
                "positions": at.get_positions().tolist(),
                "cell": at.get_cell().tolist(),
                "pbc": at.get_pbc().tolist(),
            },
            "temperature_K": self.temperature_K,
            "timestep_fs": self.timestep_fs,
            "n_steps": self.n_steps,
            "sample_every": self.sample_every,
            "rattle": self.rattle,
            "seed": int(Path(calcdir).name)
            if Path(calcdir).name.isdigit()
            else 0,
            "output": "sampled.extxyz",
        }
        input_file = calcdir / "sampler_input.json"
        input_file.write_text(json.dumps(payload_input, indent=2))

        result_dir = _run_task(
            self.remote_fs,
            self.cluster_config,
            script=_SCRIPTS / "md_sample.py",
            input_file=input_file,
            workdir=calcdir / "sampler_work",
            jobname="sample",
            loop_time=self.loop_time,
        )
        self._out = result_dir / "sampled.extxyz"

    def get_sampled_configs(self):
        return self._out


class MorseTrainer:
    """Fit a pairwise Morse potential to the accumulated labelled data."""

    def __init__(
        self,
        remote_fs,
        cluster_config,
        init_params=None,
        force_weight=1.0,
        rc=6.0,
        loop_time=0.3,
    ):
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.init_params = init_params or {"D": 0.5, "alpha": 1.5, "r0": 2.5}
        self.force_weight = force_weight
        self.rc = rc
        self.loop_time = loop_time
        self._out = None

    def update(self, calcdir, trainer_input, **kwargs):
        calcdir = Path(calcdir)
        data = (
            list(trainer_input)
            if isinstance(trainer_input, (list, tuple))
            else [trainer_input]
        )
        data = [Path(p) for p in data]

        payload_input = {
            "data": [f"additional_files/{p.name}" for p in data],
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "force_weight": self.force_weight,
            "rc": self.rc,
            "init": self.init_params,
            "output": "model.json",
        }
        input_file = calcdir / "trainer_input.json"
        input_file.write_text(json.dumps(payload_input, indent=2))

        result_dir = _run_task(
            self.remote_fs,
            self.cluster_config,
            script=_SCRIPTS / "fit_morse.py",
            input_file=input_file,
            workdir=calcdir / "trainer_work",
            jobname="train",
            additional_files=data,
            loop_time=self.loop_time,
        )
        self._out = result_dir / "model.json"

    def get_trained_model(self):
        return self._out

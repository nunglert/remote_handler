# Local EMT active-learning example

A **complete active-learning loop that runs on any machine, with no cluster**:

```
sample (ASE MD)  →  select (stride)  →  label (EMT)  →  train (Morse fit)  →  ↺
```

Everything runs through the package's real task machinery, but over the
scheduler-free `direct` executor on a `LocalFileSystem` — so there is **no SSH,
no SLURM and no ML stack**. The only dependencies are `ase`, `numpy` and
`scipy` (the core install).

- **Ground truth:** ASE's EMT calculator (the shipped `EMTEvaluator` preset).
- **"Trained model":** a pairwise Morse potential fitted with `scipy`
  (`MorseTrainer`), standing in for a heavy MLFF trainer. It is the simplest
  thing that is genuinely *fitted to the data* and then *drives the next
  sampler* — the point is the orchestration, not the accuracy.
- **Sampler:** short ASE Langevin MD (`MorseMDSampler`), the deliberately
  external sampling step cast as a `RemoteTask`.

Iteration 0 bootstraps under EMT; from iteration 1 on, the MD is driven by the
Morse potential fitted in the previous iteration, while the training set
accumulates (3 → 6 → 9 configurations).

## Run it

From the environment the package is installed in (the `direct` executor
launches a bare `python`, resolved from `PATH`):

```bash
cd examples/local_emt_AL
python run_al.py
```

It finishes in a few seconds and prints the Morse parameters fitted at each
iteration. Outputs land in `al_run/calchome/` — the standard
`StagedActiveLearningHandler` tree (`0/ 1/ 2/`, per-stage `*_output/` dirs, and
`AL_progress.json` for crash-safe resume).

## Why this matters

It shows the two claims of the framework concretely:

1. **Anything can be a `RemoteTask`.** The sampler and trainer here are ~40-line
   Python payloads (`scripts/md_sample.py`, `scripts/fit_morse.py`); the loop
   treats them exactly like a MACE trainer or a VASP evaluator.
2. **Local ⇄ cluster is one config change.** Point the same `ClusterConfig` at a
   real machine (`executor="slurm"`, a `remote_machine` SSH alias and a
   `conda_env`) and the identical actor code submits to SLURM instead.

## Files

| File | Role |
|------|------|
| `run_al.py` | Builds the actors + `StagedActiveLearningHandler` and runs the loop |
| `actors.py` | `MorseMDSampler` / `MorseTrainer` — wrap a payload as a `SingleScriptTask` |
| `scripts/md_sample.py` | Sampler payload: ASE Langevin MD under the current model |
| `scripts/fit_morse.py` | Trainer payload: `scipy` fit of a Morse potential |
| `inputs/seed_model.json` | Bootstrap model for iteration 0 (`{"type": "emt"}`) |

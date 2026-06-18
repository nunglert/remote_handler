# Changelog

All notable changes to `remote_handler` are documented here. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/). Versions
follow semantic versioning; while in `0.x`, **minor** releases may include
breaking changes and **patch** releases are bug fixes only.

The version is derived from git tags by `setuptools-scm` (tag `v0.2.0` →
version `0.2.0`). To cut a release: add a dated section below, merge to `main`,
then `git tag -a vX.Y.Z`.

## [0.2.0] — unreleased

First deliberately versioned release. Prior `0.1.0` was never tracked; this
draws the line.

### Added
- Pluggable execution backends: a scheduler-free `direct` executor alongside
  SLURM, selected via `ClusterConfig.executor`; `run_command` on the
  filesystem handlers.
- `remote_handler` command-line tool (`configure` / `check` / `show`) and a
  persisted global config — `REMOTE_HANDLER_CONFIG_DIR` is now optional.
- `ClusterConfig.from_name()` / `ClusterConfig.available_names()`.
- `EMTEvaluator` and a generic `ase_singlepoint.py` payload (shared
  `AseCalculatorEvaluator` base) — pure-ASE ground truth that runs with no
  cluster and no licensed software.
- Opt-in connection resilience (`resilient_connection`), useful for MFA authentication via certificates which are only valid for a certain amount of time.
- `NeuralilILTrainer`; `MaceFinetuningTrainer(convert_to_jax=True)`.
- A custom SLURM header template may be given as an explicit path, not only a
  registered name.
- Docs: Core Concepts, Contributing, expanded Installation/CLI, API-checked
  examples, mermaid diagrams; a weekly scheduled CI run.

### Changed
- Unified the DFT layer: `SinglePointHandler` base with `VASPHandler` /
  `ORCAHandler` presets; `AseCalculatorEvaluator` base with `VASPEvaluator` /
  `ORCAEvaluator` presets. Public signatures preserved.
- SLURM templates now render conda activation from the configured
  `conda_source` (micromamba works on SLURM too) instead of a hardcoded
  `~/miniconda3`.
- Core dependencies relaxed from exact pins to floors.
- The package version is now derived from git tags (setuptools-scm).
- `ClusterConfig` now lives in `core.cluster_config` (re-exported from
  `core.cluster_handler`, so the old import path still works — non-breaking).
- Test harness: the suite runs without any configuration (config state is
  isolated per-test); CI exercises a real SSH/SFTP + scheduler-free round-trip
  via the `sshlive` tier.

### Removed
- The `heavy` pytest marker.
- The driver-side `input_models` validation layer — each script now validates
  its own input via `validate(calcpath)`.

### Fixed
- `conda_source` now actually drives SLURM job activation (was ignored).
- Auto-patch a paramiko 3.5.1 crash (`AttributeError: public_blob`) with
  agent-provided SSH certificates, applied at import — no action needed.
- `ORCAEvaluator` gained stub mode; cleaned bare-excepts, typos, and a batch of
  docstring/lint issues.

### Breaking
- Several modules moved out of the package root: `atoms_handler` →
  `core.atoms_handler` / `core.atoms_shard`; `dft_runner` / `optimizer` /
  `run_optimizer` → `workflows.*`. Update imports accordingly.

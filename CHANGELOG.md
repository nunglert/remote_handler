# Changelog

All notable changes to `remote_handler` are documented here. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/). Versions
follow semantic versioning; while in `0.x`, **minor** releases may include
breaking changes and **patch** releases are bug fixes only.

The version is derived from git tags by `setuptools-scm` (tag `v0.2.0` →
version `0.2.0`). To cut a release: add a dated section below, merge to `main`,
then `git tag -a vX.Y.Z`.

## [Unreleased]

### Added
- **Durable task handles (detach-on-submit + crash resume).** A submitted
  `SingleScriptTask` / `ArrayScriptTask` can now persist everything needed to
  await and collect its job to a small JSON `TaskManifest`
  (`core/manifest.py`), written both locally and into the remote job
  directory. `task.detach()` allocates, starts the job, writes the manifest,
  and returns immediately — so submitting a remote job feels like a local
  backgrounded command. `TaskHandle.from_file(<manifest>)` rebuilds the
  filesystem + execution backend in a *different* process (re-binding to the
  running job via the new `Executor.attach`) to poll (`is_finished`), `wait`,
  and `collect` the results. Recovers work that was previously lost if the
  driver crashed while a job was in flight.
- `ClusterConfig.to_dict()` — JSON-round-trippable snapshot, embedded in the
  manifest so a resume does not depend on the original cluster JSON.
- `Executor.is_finished()` — non-blocking completion poll (SLURM `squeue`;
  `direct` sentinel file); `ArrayScriptTask.submit(only_start=…)` for parity
  with `SingleScriptTask`.
- **CLI `status` / `collect`.** `remote_handler status [PATH]` reports whether
  a detached task has finished (exit `0` finished / `2` running, for shell
  loops); `remote_handler collect [PATH] [--wait] [--force]` fetches its
  results. `PATH` is the manifest file or a directory containing one (defaults
  to the cwd) — so the workflow is: launch, walk away, and the next day `cd`
  back and run `remote_handler collect --wait`.

## [0.4.1] — 2026-07-09

A patch/minor increment on top of `0.4.0`: two optional per-cluster config
knobs, a bug fix in trajectory sharding, and the SLURM-availability probe moved
behind the connection-resilience wrapper. No breaking changes.

### Added
- **`conda_env_dir`** — an optional `ClusterConfig` field naming the directory
  on the remote that holds the named conda environments, so activation no
  longer assumes a default install location. Prompted for by
  `remote_handler configure cluster` and threaded through `remote_env`.
- **`slurm_header_add`** — reintroduced as an optional `ClusterConfig` field for
  extra lines injected verbatim into the generated SLURM job header (e.g.
  account/partition/QoS directives). Exposed via the CLI and rendered by the
  SLURM handler.

### Changed
- `validate_slurm_available` moved from `cluster_handler` to `slurm_utils` and
  now runs through the same resilient wrapper as other remote commands, so a
  transient credential/connection failure while probing `sbatch` pauses and
  retries instead of crashing.

### Fixed
- **`AtomsShardManager` crash on a default slice.** When `every`/`start`/`end`
  were left at their `None` defaults (as `VASPHandler` and the other DFT
  handlers do), `combine_trajs` computed `iteration_idx * indices.step` —
  `int * None` — and raised `TypeError` before any job was submitted. The
  slice fields are now normalized (`step→1`, `start→0`) and `iteration_index`
  is computed as `start + iteration_idx * step`, which also fixes a latent
  offset bug where a non-`None` `start` was ignored. Regression tests added.

## [0.4.0] — 2026-07-02

A feature release on top of `0.3.0`. The headline is a rewrite of trajectory
post-processing into a self-contained plugin framework
(`remote_handler.processing`), plus a mace-jax ASE calculator, a staged
active-learning handler, and connection resilience for the subprocess-ssh path.
Breaking: the old optimizer entry points are gone, replaced by
`remote_handler.processing` (no compatibility shims).

### Added
- **`remote_handler.processing`** — a self-contained, extensible trajectory-
  processing framework, organized by feature rather than by layer. A
  `RemoteTrajectoryProcessor` shards a trajectory and runs an ordered list of
  `Processor` steps over each shard (via `processing/driver.py` on the cluster);
  `RemoteOptimizationHandler` is a thin factory for the optimize step.
  Structural analysis is just the core with the processors you want, e.g.
  `processors=[RDFProcessor(), SteinhardtProcessor()]`.
- **Processor plugin contract** — each processor keeps its whole trio in one
  module: an `Args` schema (single source of truth), a cluster-side `run`, and
  a host-side `Processor` spec, wired via `register(name, run, schema=...)`.
  Downstream packages self-register the same way; the driver imports each
  processor's `registration_module` before resolving steps, so a downstream
  processor works with no custom driver. Built-ins: `optimize`, `rdf`.
- **Per-processor argument validation** — the driver validates every step's
  args against its schema at job start (`schema.from_dict(args).validate()`),
  so a malformed composed payload fails fast instead of deep inside an array
  task. The host `build_args` fills the same schema, so emit and check can't
  drift.
- **mace-jax ASE calculator** (`scripts/mace/mace_jax_ase.py`,
  `MaceJaxEnsembleCalculator`) — single model or stacked-params ensemble, with
  a `max_neighbors` bucket ladder to bound JIT recompiles and per-structure
  energy/force uncertainty from the ensemble spread.
- **`StagedActiveLearningHandler`** (+ `Stage`, `InputSource`, `SelectorActor`)
  — a staged active-learning loop with explicit per-stage inputs and selection.
- **Selector as an actor** — `SelectorActor` plus novelty/uncertainty-based
  selection wired through the loop; selectors take an explicit `reference`
  argument rather than mutating shared state.
- **Connection resilience** — an opt-in `ConnectionPolicy` (per-cluster
  `resilient_connection`) that pauses-and-retries SSH operations when a
  credential expires instead of crashing, resuming once it is refreshed.
- **`configure_logging`** — a utility for timestamped logging.
- `remote_handler show extras` now also reports the installed `remote_handler`
  version and conda env, per configured machine.

### Changed
- `UniformSelector` → **`StrideSelector`**. The old name remains as a subclass
  alias that emits a `DeprecationWarning`; it will be removed in a future
  release.
- Every subprocess `ssh` invocation is now non-interactive by construction
  (`stdin` closed, `-o BatchMode=yes -o ConnectTimeout=…`, detached session)
  and bounded by `ConnectionPolicy.command_timeout` (default 120s, applies even
  when resilience is off), so a stalled connection surfaces as an error rather
  than an indefinite hang.

### Fixed
- Subprocess `ssh` no longer **hangs on an interactive QR/MFA prompt** when a
  `step` certificate expires: the prompt can't grab the terminal, and a bounded
  timeout turns the stall into a `TimeoutExpired` that is classified transient
  and retried under an enabled policy.
- `squeue` status polling treats a **purged (completed) job** — `squeue -j`
  exiting with "Invalid job id specified" — as finished, instead of crashing
  the driver (the failure seen when a job completes while the connection is
  briefly down).
- Collection of finetuned MACE models.
- `lock.sh` now includes `torch-sim`; the uv lock was refreshed.

### Removed
- The old optimizer entry points, replaced by `remote_handler.processing` with
  **no compatibility shims**: `workflows/optimizer.py`,
  `workflows/processors.py`, `workflows/run_optimizer.py`, and the deprecated
  `scripts/jaxnest/optimize_traj.py`. Import handlers and processors from
  `remote_handler.processing` instead.

## [0.3.0] — 2026-06-22

A feature release on top of `0.2.0`: optional-dependency ergonomics, YAML
config inputs for the DFT and NeuralIL paths, a choice of mace-jax artifact
format, and a batch of SLURM/template fixes.

### Added
- Every actor's remote ``script`` is now optional: when omitted, the
  sampler/evaluator/trainer defaults it to the payload shipped under
  `scripts/` (e.g. `VASPEvaluator` → `dft_singlepoint.py`,
  `MaceModelTrainer` → `mace/mace_training.py`). Pass one only to override.
- `MaceFinetuningTrainer(jax_output_format=...)` — choose the converted
  mace-jax artifact: the opaque `.pkl` bundle (`"pkl"`, default — unchanged
  behaviour) or a directory bundle (`"bundle"`: a human-readable `config.json`
  plus `params.msgpack`) that mace-jax's own `load_model_bundle` consumes.
- mace-jax conversion utilities: `convert_torch_model_to_jax_bundle()` (torch
  `.model` → directory bundle) and `generate_jax_foundation_model()` (download
  a MACE foundation model and emit a mace-jax bundle).
- `remote_handler show extras` — probes (by actually importing) which optional
  dependency groups are installed.
- INCAR via YAML: `incar_config_file` on `VASPHandler` / `VASPEvaluator`,
  layered under any explicit `incar_kwargs`.
- `NeuralilILTrainer(config_file=...)` — training hyperparameters from a YAML
  file, layered defaults < `config_file` < `training_kwargs`.

### Changed
- `remote_mace_train_runner` is now optional on the MACE trainers: when unset
  it is auto-discovered from the remote conda env (`mace.cli.run_train`).
- The `torch_sim` extra no longer depends on mace — torch_sim runs without it.
  The redundant `torch_sim_freeze` extra was removed (a runtime guard reports
  the missing piece instead).
- `~` is accepted again in `conda_source` and remote paths, expanding to the
  remote `$HOME` (good defaults without hardcoding a home directory).

### Fixed
- `map_indices` is now honoured on the SLURM array path (the rendered
  `body_array.sh.j2` `MAP=(...)` lookup was dropped).
- `MaceFinetuningTrainer` no longer requests 2 GPUs by default —
  `n_processors` is threaded through to the SLURM request.
- `NeuralilILTrainer` no longer mutates its class-level default training kwargs
  across instances (defaults are deep-copied).
- Corrected the `musica` SLURM template.

### Removed
- `NSTrajectorySampler` and the `nested_sampling_AL` example. Its sampling
  payload lives in a separate downstream repository (it always pointed at an
  external jaxnest `ns_run.py`, never a shipped script), so the in-package
  actor is dropped in line with the "sampling code is a separate dependency"
  model.

### Breaking
- Actor constructors moved the remote ``script`` argument to *after* the
  required positionals (so it can default). Callers that passed it **by
  keyword** (the documented style — `sampling_script=`, `singlepoint_script=`,
  `training_script=`) are unaffected; callers that relied on positional order
  must update.

## [0.2.0] — 2026-06-18

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

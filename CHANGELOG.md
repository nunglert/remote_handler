# Changelog

All notable changes to `remote_handler` are documented here. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/). Versions
follow semantic versioning; while in `0.x`, **minor** releases may include
breaking changes and **patch** releases are bug fixes only.

The version is derived from git tags by `setuptools-scm` (tag `v0.2.0` →
version `0.2.0`). To cut a release: add a dated section below, merge to `main`,
then `git tag -a vX.Y.Z`.

## [0.8.0] — 2026-10-30

### Added
- **`on_stage_complete` hooks for `StagedActiveLearningHandler`.** Each hook
  is called with a `StageCompleted` event (stage, iteration, `calchome`,
  recorded outputs, `is_last`) once a stage's outputs are recorded. Use it
  for side work such as rendering a dashboard. Hooks are not recorded in the
  progress file, so a resume doesn't re-run them for finished steps. An
  exception in a hook is logged and does not stop the loop.
- **ACE warm start.** `AceModelTrainer.update()` takes `initial_potentials`
  (e.g. the previous iteration's committee, wired via a `Stage`'s
  `extra_inputs`). Member `i` continues fitting from
  `initial_potentials[i % len(...)]` (pacemaker's `-p`) instead of starting
  from scratch. Ladder fitting (`fit.ladder_step`/`ladder_type`) is
  supported for fits from scratch; on a warm start those keys are dropped,
  since a ladder would restart from an empty basis.
- **`keep_best` for ensemble trainers.** Train `num_models` members but hand
  only the `keep_best` with the lowest final test loss to the next stage,
  best first. Implemented for ACE (pacemaker's `test_metrics.txt`); other
  trainers reject the option. Ensemble trainers also take `num_parallel`
  (default unchanged: `min(num_models, 5)`).
- **Relative high-energy penalty for NeuralIL.** `high_energy_penalty_relative`
  sets the threshold in eV/atom above the training set's lowest energy per
  atom, so it no longer depends on the system, POTCARs or functional. It
  cannot be combined with the absolute `high_energy_penalty`.
- **`ClusterConfig.remote_fs()`** opens a `RemoteFileSystem` to the cluster
  with its `connection_policy` already applied.

### Changed
- **NeuralIL's default high-energy penalty is now relative:**
  `high_energy_penalty_relative = 5.0` replaces the absolute
  `high_energy_penalty = -5.0`. Setting `high_energy_penalty` explicitly
  restores the old behaviour.
- **The VASP single-point payload rejects unconverged SCF runs.** A run whose
  last electronic loop stopped at NELM without reaching EDIFF still produced
  energies and forces, which were used as labels. It now writes
  `scf_unconverged.info`, no labelled output and no success marker, and
  exits non-zero, so the evaluator's `required_fraction` decides whether the
  loop continues. Convergence is read from VASP 6's OUTCAR message, falling
  back to comparing OSZICAR's step count against NELM for VASP 5.

### Fixed
- **Schedule values now reach the actors.** The handlers passed
  `Schedule.get_kwargs()` to `update(**kwargs)`, but most actors never read
  those kwargs, so a scheduled `config_file`, `incar_kwargs`, `n_configs`,
  ... was silently ignored and every iteration used the initial value. Both
  `StagedActiveLearningHandler` and the legacy `ActiveLearningHandler` now
  apply each iteration's schedule to the actor before `update()` (as
  attributes, via the new `apply_schedule_kwargs`) and still forward the
  values as keyword arguments. Actors that bake a parameter in at
  construction handle it themselves: the VASP/ORCA/EMT evaluators rebuild
  `calc_config`, `NeuralilILTrainer` re-merges its hyperparameters (and
  accepts single hyperparameter keys), `SelectorActor` forwards keys to the
  wrapped selector, and `RemoteSelector` forwards them to the selector
  constructor as before.
- **Misspelt schedule keys are flagged at handler construction.** A key that
  names no attribute of the actor is still only passed to `update()`, as
  before, so actors that read schedule values from `update(**kwargs)` keep
  working unchanged. The handler now also emits a `ScheduleKeyWarning` for
  each such key, once, before anything is submitted. Turn it into an error
  with `warnings.simplefilter("error", ScheduleKeyWarning)`. An actor can
  declare the keys its `update()` reads by defining
  `unknown_schedule_keys(kwargs)`, which silences the warning (see
  `docs/source/active_learning.rst`).
  Downstream `Scheduled*` wrapper classes that copy kwargs onto the actor
  are no longer needed. They still work, since the values are just set
  twice.
- **Reattached array tasks are no longer collected while still running.**
  `from_manifest()` never re-attached the executor to the recorded job, so
  `wait_if_needed` polled the success markers instead of the queue. With a
  fraction-based `finish_check` (e.g. `RemoteAtomsHandler`, 80% by default),
  a reattached array was collected as soon as enough tasks had finished,
  with the rest still running. Reattached tasks now wait for the job itself
  to finish, then apply `finish_check` once. Old SLURM manifests without a
  `job_id` fall back to polling the markers, with a warning.
- **Connection policy applies from a task's first SFTP call.** Tasks set the
  cluster's `connection_policy` on the filesystem only after their first
  `exists()`/`makedirs()`, and unconditionally replaced it, so a filesystem
  shared between tasks could lose an enabled policy. It is now set before
  the first call and only ever upgraded, never switched off.
- **NeuralIL OneCycle schedule is reset every epoch.** It spans one epoch
  but was never reset, so after the first epoch the learning rate stayed at
  the schedule's final value.
- **NeuralIL high-energy penalty used the padded atom count.** The energy
  was divided by the padded number of atoms, which pulls the energy per atom
  of structures smaller than the largest in the set towards zero, so they
  could be down-weighted as high-energy by mistake. It now counts only real
  atoms.
- **Remote commands can no longer hang a submit forever.**
  `RemoteFileSystem.run_command` waited for the remote exit status with no
  limit, so a command stalled on an overloaded login node (e.g. the
  `run_train.py` lookup a MACE trainer does when no
  `remote_mace_train_runner` is given) blocked the submit indefinitely
  without logging anything. `run_command` on both filesystems now takes a
  `timeout` (default `COMMAND_TIMEOUT = 120` s) and raises the new
  `CommandTimeoutError` (a `RemoteConnectionError`), naming the host and the
  command, once it expires. The env probes used to locate training runners
  get a longer limit (`remote_env.PROBE_TIMEOUT = 600` s), since importing
  torch from a shared filesystem can take minutes. The trainers log the
  lookup before it starts and say how to skip it.
- **A failed runner lookup is no longer reported as a missing package.**
  `resolve_remote_module_file` / `resolve_remote_executable` turned every
  error, including connection failures, into "not found", so the trainer
  told the user to install mace/pacemaker in the remote env. Timeouts and
  connection errors now propagate; only a probe that runs and finds nothing
  returns "not found". `remote_home` also raises on a timeout instead of
  silently falling back to the local home directory.
  Behaviour change: every `run_command` call now has a limit by default, so
  pass `timeout=None` to keep the old unlimited wait.

## [0.7.0] — 2026-09-12

### Fixed
- **Standalone collection of a `collect_only=True` array task no longer
  crashes with `AttributeError`.** `ArrayScriptTask.all_calcpaths` was only
  ever set as a side effect of `allocate_data()`, so `finish_check()` broke
  for any `collect_only=True` task built in a fresh process that skipped
  allocation — including `RemoteAtomsHandler`/`SinglePointHandler`/
  `VASPHandler`/`ORCAHandler`. `all_calcpaths` is now a computed property
  derived from `num_array_tasks` (explicit, or inferred from
  `individual_ressources`), so it works identically whether or not
  allocation happened.
- **MACE trainers (`MaceModelTrainer`/`MaceFinetuningTrainer`) gained a
  `collect_only` mode.** Previously `update()` always allocated a new array
  and `get_trained_model()` always (re-)submitted, so a MACE training run
  detached in one process had no way to be collected from another.

### Added
- **`ArrayScriptTask.from_manifest()` / `SingleScriptTask.from_manifest()`**
  — reconstruct a full, `collect_only=True` task from the manifest written by
  `detach()`/`write_manifest()`, without re-allocating or re-submitting.
  Distinct from `TaskHandle` (which only rebuilds an executor+filesystem):
  this rebuilds the actual task object, so `finish_check()`/
  `collect_results()` work exactly as they did in the original process.
- **`ArrayScriptTask(num_array_tasks=...)`** constructor argument — lets a
  `collect_only=True` array task be built without `individual_ressources`
  (which `from_manifest()` relies on). Passing it alongside
  `individual_ressources` that disagree raises `TaskError` instead of
  silently picking one.
- **`submit_detached()`** on both task classes — the second half of
  `detach()` (start the job, persist the manifest, return at once) without
  the allocation step, for a caller that already allocated separately (e.g.
  a MACE trainer's `update()`).
- **`docs/source/examples/example_3_standalone.py`** — submit a DFT array and
  a MACE training array in one run, collect both from a later, separate run.
- **`AceModelTrainer` / `GraceModelTrainer`** — ensemble-array trainers
  fitting ACE (`pacemaker`) / GRACE (`gracemaker`) potentials, the same
  shape as `MaceModelTrainer`. New `ace` / `grace` install extras (they
  cannot coexist in one environment — both resolve to a top-level
  `tensorpotential` package, but from two unrelated projects; see the
  comments in `pyproject.toml`), plus `ace_tensorpot_gpu` for the
  GPU-accelerated `pacemaker` backend.
- **Crash-resume now reattaches to an already-submitted job instead of
  resubmitting a duplicate**, closing the gap where a *driver* restart
  mid-iteration (not just between iterations) orphaned a still-running job
  on the cluster. `Sampler` / `Selector` / `GroundTruthEvaluator` /
  `ModelTrainer` gained a `pending_manifest()` hook (default: reattach
  unsupported, so every existing actor keeps today's behaviour);
  `ActiveLearningHandler` and `StagedActiveLearningHandler` both gained a
  `_reattach_if_pending()` helper that flips a pending actor to
  `collect_only` for one step when it finds one (duck-typed for the
  latter, so a stage actor need not subclass the actor ABCs at all — see
  `StagedActiveLearningHandler`'s module docstring). `DetachableTaskMixin`
  gained `submit_if_needed()` / `wait_if_needed()` so an actor's `update()`
  waits correctly either way — a single immediate `finish_check()` when
  freshly submitted, or polling until done (or an optional `max_wait`) when
  reattached — and every `DetachableTaskMixin` actor (the MACE/ACE/GRACE/
  NeuralilIL trainers, the VASP/ORCA/EMT evaluators) gained a matching
  `reattach_max_wait` constructor kwarg.
- **`docs/source/active_learning.rst`** — a dedicated page for the
  active-learning loop: the four actor interfaces, `StagedActiveLearningHandler`'s
  stage/ring/DAG model and resume, and every shipped evaluator/trainer/
  selector with an unfoldable, signature-checked usage example (a real
  minimal config file for the trainers that need one). `docs/source/examples.rst`
  gained matching Examples 4–7 (ORCA, NeuralilIL, ACE, GRACE).

### Changed
- **`MaceModelTrainer` / `MaceFinetuningTrainer` / `AceModelTrainer` /
  `GraceModelTrainer` now share one `_EnsembleArrayTrainer` base** (was a
  MACE-only `_MaceArrayTrainer`), so the three ensemble-array trainers
  duplicate only their tool-specific hooks (train-runner resolution,
  artifact naming, stub scaffolding) instead of the whole submit/poll/
  collect machinery. Public signatures unaffected.
- **`docs/source/concepts.rst`'s active-learning section is now a short
  teaser** pointing at the new `active_learning.rst`, instead of the only
  place the loop was documented.

### Deprecated
- **`ActiveLearningHandler`** — raises a `DeprecationWarning` on
  construction and will be removed in a future release. It hardcodes the
  fixed sampler/evaluator/trainer triple with no selector stage; use
  `StagedActiveLearningHandler` instead (`from_actors(...)` is a drop-in
  replacement for this constructor).

## [0.6.0] — 2026-08-26

Sampling moves out of `remote_handler`. Every concrete `Sampler` and the
remote payload behind it now lives in its own downstream project repository;
the package keeps the `Sampler` interface and the orchestration around it.
This is a breaking change — see **Removed** / **Breaking** for the import
paths and extras that are gone. In exchange the host install and the test
matrix shrink substantially, and the Steinhardt processor moves *in*.

### Added
- **`SteinhardtProcessor` / `SteinhardtArgs`** — per-atom Steinhardt
  bond-orientational order parameters (`q<l>` / `w<l>`, plus the per-frame
  means `q<l>_mean` / `w<l>_mean` in `atoms.info`) as a built-in `processing`
  step, re-exported from `remote_handler.processing`. The compute is a JAX
  port of QUIP's `steinhardt_nelson_qw_module::calc_qw` together with the
  angular machinery it relies on (`SphericalYCartesian` / `SolidRCartesian`,
  `wigner3j`), validated against QUIP's `get_qw` binary to machine precision
  with the reference data checked in under `test/data/steinhardt/`. The
  `jit`-compiled core runs in float64 / complex128 (the spherical-harmonic
  sums lose all signal in float32), opted in for the scope of the computation
  rather than by flipping the global `jax_enable_x64` flag, so a surrounding
  float32 pipeline is unaffected. Previously this lived downstream and was
  cited as the example of an out-of-package processor.
- **`[steinhardt]` install extra** (`jax>=0.9.0`). `jax` is imported lazily
  inside the cluster-side `run`, so the module stays importable — and cheap —
  on a jax-free host, like the other built-in processors.

### Changed
- **`convert_to_jax` / `jax_output_format` moved to the shared MACE trainer
  base**, so `MaceModelTrainer` (train from scratch) supports the torch →
  mace-jax conversion too, not just `MaceFinetuningTrainer`. Conversion is a
  property of training a MACE model, not of finetuning specifically. The
  cluster-side step is now the shared `mace_utils.maybe_convert_to_jax`, called
  by both the `mace_training.py` and `mace_finetuning.py` payloads, and
  `MaceTrainingInput` gained the two matching fields. Existing
  `MaceFinetuningTrainer` usage is unchanged.
- **The `[all]` extra is recomposed** to
  `dev,mace_freeze,quests,neuralil,mace_jax,steinhardt`: `torch_sim` and
  `clinamen` are gone with the samplers, `steinhardt` is new. CI still
  installs `[all]`, so every optional tier keeps un-skipping.
- **Docs and README follow the split.** The installation table drops the
  removed extras and the "`torch_sim` needs a mace source" warning, the
  examples page covers DFT labelling and MACE training, and the README gains
  an *Example projects using remote_handler* section linking the downstream
  sampler repositories.

### Fixed
- **`FixAtoms` constraints no longer corrupt forces on the *aggregation*
  paths either.** `scripts/mace/mace_utils.write_with_keys` and
  `AseCalculatorEvaluator._collect_results` now call
  `get_forces(apply_constraint=False)`. Both re-derive forces from an `Atoms`
  deserialized from extxyz *after* the singlepoint already stored the correct
  values, so a `FixAtoms` constraint carried in the file silently re-zeroed
  the frozen atoms' forces on the way into `REF_forces` — the training labels
  — even though the payload that wrote them was already fixed. The 0.5.0 entry
  covered only `dft_singlepoint.py` / `orca_singlepoint.py` /
  `ase_singlepoint.py`; this completes it. Regression tests added for both
  call sites.

### Removed
- **Every concrete sampler and its remote payload.** `workflows/samplers.py`
  is gone in full — `LammpsSampler`, `TorchSimSampler`, `CMAESSampler`,
  `BatchCMAESSampler`, `SSPDSampler`, `BatchSSPDSampler` — along with the
  payloads they shipped (`scripts/lammps/`, `scripts/torch_sim/` including
  `mace_ensemble.py`, `parallel_tempering.py`, `create_interface.py` and
  `simulation_pipelines.py`, `scripts/clinamen2/`, `scripts/sspd/`) and their
  input schemas (`LammpsSamplingInput`, `TorchSimSamplingInput`,
  `CmaesSamplingInput`, `CmaesOptimizeInput`). `remote_handler` now provides
  only the `Sampler` interface, in `workflows/active_learning_base.py`.
- **The `[torch_sim]` and `[clinamen]` install extras.**
- **The samplers' examples** — `examples/mace_md_AL/`,
  `examples/torch_sim_AL/`, `examples/cmaes_evolution_AL/`,
  `examples/mace_sspd_AL/`, the stale `examples/cluster_handler.py`, and the
  torch-sim / active-learning docs example pages. `examples/local_emt_AL/`
  (no cluster, no licensed software) remains the runnable reference.
- **The samplers' tests** — `test_cmaes_sampling_integration.py`,
  `test_torch_sim_integration.py`, `test_mace_ensemble.py` and
  `test_active_learning_workflows.py`. The `integration` tier now means
  "needs mace", not "needs mace/torch_sim/clinamen".

### Breaking
- `from remote_handler.workflows import LammpsSampler, TorchSimSampler` (both
  were in `workflows.__all__`) and any import from
  `remote_handler.workflows.samplers` now raise `ImportError`. Depend on the
  relevant downstream project instead — the actors keep their constructor
  signatures there, so only the import path moves.
- `pip install ".[torch_sim]"` / `".[clinamen]"` no longer resolve; pip warns
  that the extra is not provided and installs nothing for it. `".[all]"` still
  resolves but no longer brings torch-sim or Clinamen2 into the environment.

## [0.5.0] — 2026-07-26

### Added
- **Durable task handles (crash resume + detach-on-submit).** A submitted
  `SingleScriptTask` / `ArrayScriptTask` now persists everything needed to
  await and collect its job to a small JSON `TaskManifest` (`core/manifest.py`),
  written both locally (`<local_savepath>.manifest.json`) and into the remote
  job directory. **This happens by default** on every `submit`/`run` — right
  after submission and *before* the blocking wait — so if the driver process
  dies mid-wait the running job is no longer orphaned (the original motivation).
  Opt out per task with `write_manifest_on_submit=False`. The write is
  best-effort (the local copy is the contract; a remote-copy failure only
  warns) so persisting the safety net can never take down the job it protects.
  `task.detach()` additionally *returns without waiting*, so submitting a
  remote job feels like a local backgrounded command.
  `TaskHandle.from_file(<manifest>)` rebuilds the filesystem + execution backend
  in a *different* process (re-binding to the running job via the new
  `Executor.attach`) to poll (`is_finished`), `wait`, and `collect` the results.
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
- **`[all]` install extra** — `pip install ".[all]"` pulls the full stack in
  one shot (`dev,mace_freeze,torch_sim,clinamen,quests,neuralil,mace_jax`).
  It deliberately uses the `mace_freeze` fork as the sole `mace-torch` source
  (not `[mace]`), so it stays internally consistent. CI's install step now
  uses it, so every optional test tier un-skips.
- **Runnable local MD→EMT→Morse AL example** (`examples/local_emt_AL/`) — a
  self-contained active-learning loop (MD sampling → EMT ground truth → Morse
  MLFF) that runs locally with no cluster or licensed software, plus the
  accompanying docs.
- **`memory` job parameter** on `SlurmHandlerBase`, threaded through the tasks
  layer and wired to the DFT engines (`dft_runner`) and the SLURM templates
  (`datalab`/`dirona`/`musica`), so a per-job memory request can be set
  centrally instead of hardcoded in a header.

### Changed
- **`shard_size` is now optional.** `AtomsShardManager`,
  `RemoteTrajectoryProcessor`, and `RemoteOptimizationHandler` accept
  `shard_size=None` (or a negative value such as `-1`), meaning "no sharding":
  the entire configuration set is processed as a single shard, resolved once
  the total count is known. Previously a shard size was mandatory.

### Fixed
- **Conda env no longer shadowed by module-loaded interpreters.** The generated
  activation now appends `[ -n "$CONDA_PREFIX" ] && export
  PATH="$CONDA_PREFIX/bin:$PATH"` after `conda activate`. On clusters where a
  job's `environment`/`slurm_header_add` runs `module load` (e.g. EESSI/Lmod,
  which prepend a compat-layer `python` to `PATH`), conda would re-insert the
  env's `bin` *behind* that entry, so a bare `python` ran the module's
  interpreter and the env's packages (`ase`, …) were invisible —
  `ModuleNotFoundError` despite a correct `conda_env`. Forcing the env's `bin`
  to the front fixes it centrally, so users never have to hardcode the env path
  in their header (it is already declared once, in the cluster config). Applies
  to both the SLURM and `direct` backends; guarded so a blank activation is a
  no-op.
- **`FixAtoms` constraints no longer corrupt written forces.** The singlepoint
  payloads (`dft_singlepoint.py`, `orca_singlepoint.py`, `ase_singlepoint.py`)
  read `atoms.calc.results["forces"]` instead of `atoms.get_forces()`, which
  applies `FixAtoms` constraints deserialized from the extxyz and silently
  zeroes the forces on frozen atoms. The calculator's stored results are
  unaffected, so the real forces are now recorded.

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

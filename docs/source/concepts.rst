Core Concepts
=============

This page introduces the main abstractions in ``remote_handler`` and the
boundaries between them, independently of any particular cluster or workflow.
It is the mental model the rest of the documentation assumes; the
:doc:`api/index` fills in the concrete classes.

The big picture
---------------

``remote_handler`` orchestrates **active-learning workflows for machine-learned
force fields** across remote HPC clusters. The guiding idea is a strict split
between the **host** (your laptop / login node, where orchestration logic
lives) and one or more **remotes** (clusters, where the actual compute runs).
The host never does science; it moves files, launches jobs, and collects
results. The science runs on the remote, inside its own environment.

The package is organised in three layers:

``core/``
    Transport and execution primitives: SSH/SFTP connections, filesystem
    handlers, the executor abstraction, and the task definitions.
    Dependency-light and host-side.

``workflows/``
    The active-learning loop and its actors. Pure orchestration built on top of
    ``core``.

``scripts/``
    **Remote payloads** -- shipped to the cluster and executed in its conda
    environment, never imported on the host. This is the only layer that
    depends on the heavy ML stacks (MACE, torch-sim, jax, ...).

A useful test for *where new code belongs*: if it needs an ML framework at
import time, it is a ``scripts/`` payload; if it decides *what* to run and
*where*, it is ``core``/``workflows``.

The host / remote boundary
--------------------------

Every file operation goes through a single interface,
:class:`~remote_handler.core.file_handler.FileSystemHandlerBase`, with two
interchangeable implementations. The same task code therefore runs unchanged
whether the "remote" is a cluster over SFTP or just the local disk:

.. mermaid::

   flowchart LR
       T["task / workflow code"] --> FS{"FileSystemHandlerBase"}
       FS --> L["LocalFileSystem<br/>(subprocess on this machine)"]
       FS --> R["RemoteFileSystem<br/>(paramiko SFTP over SSH)"]

Both implementations expose the same verbs (``put`` / ``get`` / ``makedirs`` /
``read_ase`` / ``run_command`` / ...). ``run_command`` runs a shell command
*where the filesystem lives* -- a local subprocess, or one exec over the
same authenticated SSH connection -- which is what lets execution happen
without a scheduler (see below). Remote hosts are resolved through your
``~/.ssh/config`` (key-based auth, ``ProxyJump`` supported).

Tasks: the unit of remote work
------------------------------

A :class:`~remote_handler.core.tasks.RemoteTask` is the central execution
abstraction. It answers three questions and an optional fourth, and nothing
else:

- **What to transfer?** ``allocate_data`` -- stage inputs onto the remote.
- **How to start it?** ``submit`` -- launch the job(s).
- **How to collect?** ``collect_results`` -- bring outputs back.
- *(optional)* **Did it really finish?** ``finish_check`` -- e.g. look for
  a sentinel file, independent of what the scheduler thinks.

The lifecycle of a run is always the same shape:

.. mermaid::

   sequenceDiagram
       participant Task as RemoteTask
       participant FS as remote_fs
       participant Ex as Executor
       Task->>FS: allocate_data()  -- transfer inputs
       Task->>Ex: submit()
       Ex-->>Task: (job running on target)
       Task->>Ex: wait_until_finished()
       Task->>FS: collect_results()  -- retrieve outputs

The two concrete tasks are
:class:`~remote_handler.core.tasks.SingleScriptTask` (one script, one input)
and :class:`~remote_handler.core.tasks.ArrayScriptTask` (the same script across
many shards, one per sub-directory). Crucially, a task describes *what* to run
-- it does **not** know *how* the job is actually launched. That is the
executor's job.

Executors: the pluggable backend seam
-------------------------------------

How a task's scripts actually run is decided by the ``executor`` field of its
:class:`~remote_handler.core.cluster_config.ClusterConfig`. A task builds its
backend through a factory rather than hard-wiring a scheduler, so the same task
runs with or without one:

.. mermaid::

   flowchart TB
       Task["SingleScriptTask / ArrayScriptTask"]
       Task --> SF["make_single_executor()"]
       Task --> AF["make_array_executor()"]
       SF -->|"'slurm'"| SJH["SlurmJobHandler"]
       SF -->|"'direct'"| DSE["DirectSingleExecutor"]
       AF -->|"'slurm'"| SAH["SlurmArrayHandler"]
       AF -->|"'direct'"| DAE["DirectArrayExecutor"]

Both backends satisfy the same small contract
(:class:`~remote_handler.core.executors.Executor`: ``submit`` /
``wait_until_finished``, plus ``write_jobfile`` and a ``calcdir``). The SLURM
backend submits with ``sbatch`` and polls the queue; the ``direct`` backend
writes a launch script, runs it detached over ``run_command``, and polls for a
completion sentinel -- no scheduler required. Cluster JSONs without an
``executor`` key default to ``slurm``, so existing setups are unaffected.

The active-learning loop
------------------------

The ``workflows`` layer composes tasks into the active-learning cycle. Three
roles, each an abstract actor, hand artifacts to the next:

.. mermaid::

   flowchart TB
       Sampler["Sampler"] -->|"candidate configs (.extxyz)"| Eval["GroundTruthEvaluator"]
       Eval -->|"labelled configs (DFT)"| Trainer["ModelTrainer"]
       Trainer -->|"retrained MLFF model"| Sampler

- :class:`~remote_handler.workflows.active_learning_base.Sampler` proposes new
  candidate configurations.
- :class:`~remote_handler.workflows.active_learning_base.GroundTruthEvaluator`
  labels them with ground truth (DFT via VASP/ORCA).
- :class:`~remote_handler.workflows.active_learning_base.ModelTrainer` retrains
  the force field on the accumulated data; the new model feeds the next
  sampler.

Each actor can run on a *different* machine, and each typically does its work by
constructing and running a ``RemoteTask``. The loop itself is driven by
:class:`~remote_handler.workflows.active_learning_base.ActiveLearningHandler`,
which owns the **bookkeeping**: it writes a ``calchome/`` tree (one numbered
directory per iteration) and an ``AL_progress.json`` cache recording every
output path. That cache is what makes the loop **resumable** -- a crashed
run picks up at the actor it left off, rather than recomputing. A
:class:`~remote_handler.workflows.active_learning_base.Schedule` can vary each
actor's behaviour from one iteration to the next.

Configuration
-------------

Two configuration objects, at two scopes, keep the above wired together:

:class:`~remote_handler.core.cluster_config.ClusterConfig`
    *Per remote target.* Where and as whom work runs (SSH alias, conda env,
    save path) plus policy (connection resilience, which ``executor`` to use).
    Backend-agnostic -- consumed by every executor. Loaded from a JSON with
    ``ClusterConfig.from_name("lugus")`` or ``from_file(...)``.

``RemoteHandlerConfig`` (via ``get_config()``)
    *Global.* Where cluster JSONs and templates live, and the defaults offered
    when creating new ones. Resolved once, with precedence
    *environment variables > persisted config file > built-in defaults*.

Both are normally managed for you by the ``remote_handler`` command-line tool
(see :doc:`cli`), so day-to-day you refer to clusters by name and never touch
the underlying files by hand.

One repository per project
--------------------------

``remote_handler`` is a **library, not an application**. It is not where an
active-learning campaign lives; it is the reusable machinery a campaign builds
on. The intended layout splits across **separate repositories, each a
dependency of the next** -- so the parts that churn fast never sit in the same
repo as the parts that must stay stable:

.. mermaid::

   flowchart TB
       subgraph P["AL project (repo)"]
           RUN["Sampler adapter + run script<br/>+ cluster configs"]
       end
       RUN -->|depends on| SAMP["sampling code<br/>(library — separate repo)<br/>the sampling method, standalone"]
       RUN -->|depends on| RH["remote_handler<br/>(library — separate repo)<br/>transport · DFT/ORCA evaluation · MACE training · AL loop"]

The two libraries are independent -- only the project ties them together.
Three roles, ordered by how fast the code changes:

- **``remote_handler`` -- stable core.** Transport, the executor backends, the
  DFT/ORCA evaluators, MACE/NeuralIL training, and the active-learning loop
  itself change slowly. It also *defines* the actor interfaces (the
  :class:`~remote_handler.workflows.active_learning_base.Sampler` /
  ``GroundTruthEvaluator`` / ``ModelTrainer`` ABCs). Tested, released, and
  versioned here, and reused by every project.
- **Sampling code -- its own library, its own repo.** The actual sampling
  method, where most development happens and the code churns the most. It is a
  *standalone* package: it knows nothing about ``remote_handler`` or active
  learning, so it can evolve on its own schedule and be reused outside this
  context entirely.
- **The AL project -- thin glue.** A per-campaign repo that pins both libraries
  as dependencies and provides the bridge between them: a small
  :class:`~remote_handler.workflows.active_learning_base.Sampler` adapter that
  drives the sampling library, a run script wiring it to the shipped
  evaluators/trainers, and the cluster configs and inputs for *this* run.

This keeps fast-moving, half-finished sampling experiments out of the core
entirely -- they are not even in the same repository -- so ``remote_handler``
(and the reusable training / DFT evaluation it provides) stays clean, broadly
tested, and shared. The sampling library, owing nothing to ``remote_handler``,
stays independently useful too. Pinning to tags
(``... @ git+<repo>@v0.2.0``) means a project only picks up changes from either
library when it deliberately bumps the pin.

Where to go next
----------------

- :doc:`installation` -- install the host driver and the right extras.
- :doc:`cli` -- configure clusters and diagnose your setup.
- :doc:`quickstart` -- a first end-to-end run.
- :doc:`api/index` -- the concrete classes behind every concept above.
- :doc:`contributing` -- how the layers map onto where code should live.

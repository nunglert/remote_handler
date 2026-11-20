Trajectory Processing
=====================

``remote_handler.processing`` runs per-configuration work over large
trajectories on a remote cluster, such as relaxing every structure or
computing structural descriptors for every frame. You give it trajectory
files and an ordered list of **processors**. It shards the configurations into
an array job, runs the processors on each shard, and writes the annotated
configurations back next to your input files.

.. contents:: On this page
   :local:
   :depth: 1

How it works
------------

.. mermaid::

   flowchart LR
       T["traj files<br/>(.extxyz)"] --> S["shard"]
       S --> A0["shard 0"]
       S --> A1["shard 1"]
       S --> An["shard n"]
       subgraph cluster["array job on the cluster (driver.py per shard)"]
           A0 --> P0["step 1 → step 2 → …"]
           A1 --> P1["step 1 → step 2 → …"]
           An --> Pn["step 1 → step 2 → …"]
       end
       P0 --> C["collect + regroup"]
       P1 --> C
       Pn --> C
       C --> R["&lt;tag&gt;_&lt;traj&gt;.extxyz<br/>next to each input"]

Two concerns are kept apart:

* **Data movement** is done once, by
  :class:`~remote_handler.processing.handler.RemoteTrajectoryProcessor`. It
  reads the trajectories, splits them into shards of ``shard_size``
  configurations, uploads each shard with a small input JSON, submits one
  array task per shard, and afterwards collects the results and regroups them
  into the layout of the original files.
* **Computation** is a list of
  :class:`~remote_handler.processing.base.Processor` steps. On the cluster,
  the shipped driver (``processing/driver.py``) reads its shard once, runs the
  steps in the order you listed them, and writes the shard once. So a single
  job can relax structures *and then* characterize the relaxed result, and
  the trajectory is uploaded only once however many steps you run.

Each processor annotates the ``ase.Atoms`` objects in place. By convention,
per-frame results go in ``atoms.info`` and per-atom results in
``atoms.arrays``, so they survive the round trip through ``.extxyz``.

The array job goes through the normal task machinery (see :doc:`concepts`), so
it runs on SLURM or the scheduler-free ``direct`` executor, whichever the
:class:`~remote_handler.core.cluster_config.ClusterConfig` selects.

.. note::

   The driver and the processors run **in the cluster's conda env**, so
   ``remote_handler`` must be installed there, together with the extras of
   the processors you use (``steinhardt`` for Steinhardt; neuralil-jaxrens for
   the ``NNFF``/``MACE`` optimize backends). The host only needs the core
   install.

Built-in processors
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Processor
     - What it does
     - Results
   * - :class:`~remote_handler.processing.OptimizeProcessor`
       (``"optimize"``)
     - LBFGS relaxation. Backends: ``NNFF`` / ``MACE`` (a NeuralIL ensemble
       from ``pickle_file``, staged once and shared by all shards) or ``LJ``
       (Lennard-Jones, for smoke tests).
     - Relaxed positions; ``info["is_converged"]``. Already converged
       structures are skipped on a rerun.
   * - :class:`~remote_handler.processing.RDFProcessor`
       (``"rdf"``)
     - Total radial distribution function g(r), PBC-aware, via an ASE
       neighbour list. No extra dependencies.
     - ``info["r_rdf"]`` (bin centres), ``info["rdf_total"]`` (g(r)).
   * - :class:`~remote_handler.processing.SteinhardtProcessor`
       (``"steinhardt"``)
     - Steinhardt bond-orientational order parameters q\ :sub:`l` and
       w\ :sub:`l` for each ``l`` in ``l_values`` (default 4, 6), over
       neighbours within ``[r_cut_min, r_cut]``. A JAX port of QUIP's
       ``get_qw``, validated against it. Needs the ``steinhardt`` extra.
     - ``arrays["q<l>"]``, ``arrays["w<l>"]`` (per atom);
       ``info["q<l>_mean"]``, ``info["w<l>_mean"]`` (per frame).

Each processor's arguments are checked twice: by its constructor on the host,
and by its arg schema (``OptimizeArgs``, ``RdfArgs``, ``SteinhardtArgs``) in
the driver *before* any compute starts. A bad argument fails at job start,
not halfway through an array task.

Running an analysis
-------------------

Pass the processors you want to
:class:`~remote_handler.processing.handler.RemoteTrajectoryProcessor` and call
``run()``. This example computes g(r) and q4/q6 for two MD trajectories:

.. literalinclude:: examples/example_8_processing.py
   :language: python
   :start-after: if __name__ == "__main__":
   :dedent: 4

Things to know:

* **Output files.** Results are written next to each input file, prefixed with
  ``tag``: ``md_300K.extxyz`` → ``ana_md_300K.extxyz``. The order of the
  configurations is the same as in the input.
* **Subsets.** ``every`` / ``start`` / ``end`` slice each trajectory before
  sharding, e.g. ``every=10`` to analyse every tenth frame.
* **Shard size.** ``shard_size=None`` (or ``-1``) puts everything into one
  task. Otherwise there are ``ceil(n_configs / shard_size)`` tasks, of which
  at most ``num_parallel`` run at once.
* **Environment modules.** ``slurm_header_add`` defaults to
  ``"module load openblas"``. Pass your cluster's equivalent, or ``""`` if it
  has no such module.
* **Remote layout.** Each shard works in ``<savepath>/calcdirs/<i>/``, which
  holds its input JSON, its configurations and the processed output.

For relaxation alone there is a shortcut,
:class:`~remote_handler.processing.handler.RemoteOptimizationHandler`, which
takes the optimize arguments directly. It is equivalent to
``RemoteTrajectoryProcessor(processors=[OptimizeProcessor(...)], ...)``. To
relax *and* analyse in one job, list the steps together:

.. code-block:: python

   processors=[
       OptimizeProcessor(pickle_file="ensemble.pkl", fmax=0.05),
       RDFProcessor(),
       SteinhardtProcessor(r_cut=3.0),
   ]

Failures and resuming
---------------------

``run()`` raises if any shard failed or fewer configurations came back than
went in, instead of writing partial results. To retry, construct the handler
again with the same arguments and ``collect_only=True``. Nothing is restaged.
Then call:

.. code-block:: python

   handler.recompute_failed()

This resubmits only the failed shards, waits for them, and collects and writes
the results of all shards.

A shard writes its output only when all its steps have finished. A shard that
was killed partway, for example by the time limit, therefore starts over when
it is rerun. If your steps are slow, use a smaller ``shard_size``.

Writing your own processor
--------------------------

Processors are plugins. A processor that lives in *your* package works with
the stock driver, with no change to ``remote_handler``. It consists of four
parts, which belong in **one module**:

1. an **arg schema**, a dataclass that subclasses ``ScriptInput`` (from
   ``remote_handler.scripts.input_models``) and has a ``validate`` method;
2. a cluster-side **run** function,
   ``run(atoms_list, args, calcpath=None) -> atoms_list``;
3. a host-side :class:`~remote_handler.processing.base.Processor` subclass
   with a class-level ``name`` and a ``build_args`` method;
4. a module-level :func:`~remote_handler.processing.registry.register` call
   that ties the ``name`` to ``run`` and the schema.

Here is a complete processor that computes per-atom coordination numbers:

.. literalinclude:: examples/custom_processor.py
   :language: python
   :start-after: from remote_handler.scripts.input_models import ScriptInput

Use it like a built-in:

.. code-block:: python

   from my_project.processors.coordination import CoordinationProcessor

   processors = [RDFProcessor(), CoordinationProcessor(cutoff=2.8)]

**How it reaches the cluster.** The cluster never receives your class, only
the step names, their arguments, and a list of modules to import. Every
``Processor`` has a ``registration_module``, which defaults to the module the
class is defined in. The driver imports those modules before it resolves the
steps, and importing yours runs its ``register(...)`` call. This is why the
four parts belong in one module, and why your package must be installed in the
cluster's conda env. If it is not, the job fails at start with *"Could not
import processor registration module … Is the downstream package installed in
this env?"*.

Further rules:

* ``name`` must be unique within one pipeline and must match the name passed
  to ``register``. A subclass without ``name`` raises ``TypeError`` when the
  class is defined.
* ``build_args`` must return JSON-serializable values. It receives the remote
  working directory, ``savepath``, for building the remote paths of staged
  files.
* To ship a file the step needs on the cluster (a model, a reference
  structure), override ``shared_files()`` to return its local path. It is
  uploaded once into ``savepath`` and shared by all shards. See
  ``OptimizeProcessor`` for the pattern.
* Registering a name that already exists replaces it. This also lets you
  override a built-in.
* ``register(..., schema=None)`` is allowed, but then the arguments are not
  checked before the job starts.

API reference
-------------

The full class and function reference is in the
:ref:`Trajectory Processing section <api-processing>` of the API docs.

Examples
========

Practical, end-to-end examples for the most common workflows. Every script
shown here is a real file in the repository, checked by the test suite so it
cannot silently drift out of sync with the API.

They come in two flavours:

* The **runnable local example** below needs no cluster and no ML stack -- it
  runs on any machine with just the core install and is *executed* end-to-end
  in CI (``test/test_local_emt_example.py``). Start here.
* The **cluster workflows** (Examples 1--2) orchestrate remote work (SSH +
  SLURM, GPUs, VASP), so they are **not executed** in CI -- only checked for
  API-currency (``test/test_examples.py``). They assume you have registered the
  referenced clusters with ``remote_handler configure cluster`` (see
  :doc:`cli`); adapt the aliases, paths and resources to your own setup before
  running.

The conceptual roles (sampler / selector / evaluator / trainer) and the
:class:`~remote_handler.workflows.staged_active_learning.StagedActiveLearningHandler`
that wires them together are explained in :doc:`active_learning`.

Run it now: a full loop with no cluster
---------------------------------------

A complete active-learning cycle for bulk Cu that runs on your laptop in a few
seconds::

    sample (ASE MD)  ->  select (stride)  ->  label (EMT)  ->  train (Morse fit)  ->  loop

Everything goes through the package's real task machinery, but over the
scheduler-free ``direct`` executor on a
:class:`~remote_handler.core.file_handler.LocalFileSystem` -- so there is no
SSH, no SLURM and no ML framework. Ground truth is ASE's EMT calculator (the
shipped :class:`~remote_handler.workflows.evaluators.EMTEvaluator`); the
"trained model" is a pairwise Morse potential fitted with ``scipy``, standing
in for a heavy MLFF trainer. Iteration 0 bootstraps under EMT; from then on the
MD is driven by the potential fitted in the previous iteration.

The point is orchestration, not accuracy: it shows that *anything* can be cast
as a :class:`~remote_handler.core.tasks.RemoteTask`, and that moving the same
loop onto a real cluster is a single change to the ``ClusterConfig``
(``executor="slurm"`` with a ``remote_machine`` and ``conda_env``).

Run it from the environment the package is installed in:

.. code-block:: bash

   cd examples/local_emt_AL
   python run_al.py

**The driver** wires four actors into the staged handler and runs the loop:

.. literalinclude:: ../../examples/local_emt_AL/run_al.py
   :language: python

**The custom actors** are thin wrappers that cast an ordinary Python payload as
a :class:`~remote_handler.core.tasks.SingleScriptTask`. The sampler and trainer
here are *external* code plugged into the loop -- exactly how a production
sampler or trainer is integrated:

.. literalinclude:: ../../examples/local_emt_AL/actors.py
   :language: python

**The payloads** run on the target (here, locally). The sampler payload is a
short ASE Langevin-MD run under the current model; the trainer payload fits the
Morse parameters to the labelled data. Neither imports ``remote_handler`` -- a
payload is just a script the executor ships and runs:

.. literalinclude:: ../../examples/local_emt_AL/scripts/md_sample.py
   :language: python

.. literalinclude:: ../../examples/local_emt_AL/scripts/fit_morse.py
   :language: python

Cluster workflows
-----------------

The remaining examples orchestrate work on remote clusters. Adapt the cluster
aliases and paths to your own setup before running.

Example 1: DFT single-point labelling
-------------------------------------

Run VASP single-point calculations on a set of structures as a SLURM array job,
then collect the labelled configurations back to the host:

.. literalinclude:: examples/example_1_dft.py
   :language: python

Example 2: Training a MACE force field
--------------------------------------

Configure a :class:`~remote_handler.workflows.trainers.MaceModelTrainer` and run
a training step. The trainer is an active-learning *actor*: in a full run an
:class:`~remote_handler.workflows.active_learning_base.ActiveLearningHandler`
loop drives it, but you can also call ``update`` / ``get_trained_model``
directly:

.. literalinclude:: examples/example_2_mace_training.py
   :language: python

Example 3: Standalone submit/collect
-------------------------------------

Submit a DFT array and a MACE training array in one run and exit at once
(``detach``/``submit_detached``), then reconstruct and collect both from a
later, separate run -- e.g. an active-learning driver that submits before a
laptop goes to sleep and collects the next morning. Like Examples 1--2, this
is statically validated only (API-currency, no live cluster):

.. literalinclude:: examples/example_3_standalone.py
   :language: python

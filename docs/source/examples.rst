Examples
========

Practical, end-to-end examples for the most common workflows. Each example is a
real, importable script under ``docs/source/examples/`` -- the very same files
shown here are checked by the test suite (``test/test_examples.py``), so they
cannot silently drift out of sync with the API.

.. note::

   These scripts orchestrate work on remote clusters (SSH + SLURM, GPUs, VASP),
   so they are **not executed** in CI -- only checked for API-currency. They
   assume you have registered the referenced clusters with
   ``remote_handler configure cluster`` (see :doc:`cli`); adapt the aliases,
   paths and resources to your own setup before running. The conceptual roles
   (sampler / evaluator / trainer) are explained in :doc:`concepts`.

Example 1: DFT single-point labelling
-------------------------------------

Run VASP single-point calculations on a set of structures as a SLURM array job,
then collect the labelled configurations back to the host:

.. literalinclude:: examples/example_1_dft.py
   :language: python

Example 2: Training a MACE force field
--------------------------------------

Configure a :class:`~remote_handler.workflows.trainers.MaceModelTrainer` and run
a training step. The trainer is an active-learning *actor*: in a full run the
loop (Example 4) drives it, but you can also call ``update`` /
``get_trained_model`` directly:

.. literalinclude:: examples/example_2_mace_training.py
   :language: python

Example 3: Sampling with torch-sim MD
-------------------------------------

Configure a :class:`~remote_handler.workflows.samplers.TorchSimSampler` to run
parallel-tempering MD from the current model and propose new candidate
configurations:

.. literalinclude:: examples/example_3_torch_sim.py
   :language: python

Example 4: A full active-learning loop
--------------------------------------

Wire a sampler, a ground-truth evaluator and a trainer into the
:class:`~remote_handler.workflows.active_learning_base.ActiveLearningHandler`
and run the Sampler → Evaluator → Trainer loop. Note how each actor can target a
different cluster:

.. literalinclude:: examples/example_4_active_learning.py
   :language: python

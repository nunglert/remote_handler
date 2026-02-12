Examples
========

This section provides practical examples for common use cases.

Example 1: Batch Optimization
------------------------------

Optimize a trajectory of atomic structures on a remote cluster:

.. code-block:: python

   from remote_handler import RemoteOptimizationHandler
   from remote_handler.core import ClusterConfig
   from remote_handler.core.file_handler import RemoteFileSystem

   # Load cluster configuration
   config = ClusterConfig.from_file("datalab.json")
   remote_fs = RemoteFileSystem(remote_alias="datalab")

   # Create handler
   handler = RemoteOptimizationHandler(
       traj_files=["md_trajectory.extxyz"],
       cluster_config=config,
       remote_fs=remote_fs,
       n_processors=4,
       shard_size=50,  # 50 structures per job
       num_parallel=20,  # Run 20 jobs in parallel
       every=5,  # Process every 5th frame
       optimization_backend="NNFF",
       pickle_file="model.pkl",
       do_steinhardt=True,
   )

   # Run the workflow
   handler.allocate_data()
   handler.submit()
   handler.collect_and_harvest()

Example 2: Active Learning
---------------------------

Run an active learning loop for training a neural network potential:

.. code-block:: python

   from remote_handler.workflows import ActiveLearningWorkflow
   from remote_handler.workflows.evaluators import ModelEvaluator
   from remote_handler.workflows.samplers import UncertaintySampler
   from remote_handler.workflows.trainers import NNFFTrainer

   # Configure active learning components
   evaluator = ModelEvaluator(model_path="initial_model.pkl")
   sampler = UncertaintySampler(threshold=0.1, max_samples=100)
   trainer = NNFFTrainer(epochs=1000, learning_rate=0.001)

   # Create workflow
   workflow = ActiveLearningWorkflow(
       initial_data="seed_data.extxyz",
       cluster_config=config,
       evaluator=evaluator,
       sampler=sampler,
       trainer=trainer,
       num_iterations=10,
   )

   # Run active learning
   workflow.run()

Example 3: DFT Calculations
----------------------------

Run VASP single-point calculations on selected structures:

.. code-block:: python

   from remote_handler.dft_runner import VASPHandler

   handler = VASPHandler(
       traj_files=["selected_structures.extxyz"],
       cluster_config=config,
       remote_fs=remote_fs,
       savepath="/scratch/username/vasp_calc",
       num_parallel=10,
       every=1,
       incar_kwargs={
           "xc": "PBE",
           "encut": 400,
           "prec": "Accurate",
           "ismear": 0,
           "sigma": 0.05,
           "ediff": 1e-6,
       },
   )

   handler.run()

Example 4: Collecting Results from Previous Run
------------------------------------------------

Resume and collect results from a previous calculation:

.. code-block:: python

   handler = RemoteOptimizationHandler(
       traj_files=["trajectory.extxyz"],
       cluster_config=config,
       remote_fs=remote_fs,
       savepath="/scratch/username/previous_run",
       collect_only=True,  # Don't submit new jobs
   )

   # Just collect and process results
   handler.collect_and_harvest()

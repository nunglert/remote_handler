Quick Start
===========

This guide will get you started with using remote_handler for your computational workflows.

.. tip::

   Want to see a complete active-learning loop before wiring up a cluster? The
   bundled ``examples/local_emt_AL`` example runs the whole
   sample → select → label → train cycle on your laptop in a few seconds, with
   no SSH, no SLURM and no ML stack. See :doc:`examples` (*Run it now: a full
   loop with no cluster*).

Basic Workflow
--------------

The simplest way to use remote_handler is with a basic Python script that you want
to run on a remote cluster. This example shows how to submit a single script task.

1. **Configure your cluster**

   Create a configuration file or use ``ClusterConfig``:

   .. code-block:: python

      from remote_handler.core.cluster_handler import ClusterConfig

      config = ClusterConfig(
          remote_machine="datalab",
          partition="normal",
          conda_env="myenv",
          conda_source="~/miniconda3/bin/activate",
          savepath="/scratch/username",
          template="datalab.sh.j2"
      )

2. **Set up file system handler**

   .. code-block:: python

      from remote_handler.core.file_handler import RemoteFileSystem

      remote_fs = RemoteFileSystem(remote_alias="datalab")

3. **Create a simple Python script to run remotely**

   Save this as ``my_script.py``:

   .. code-block:: python

      # my_script.py
      import sys
      import json

      def main():
          # Read input data
          with open("input.json", "r") as f:
              data = json.load(f)

          # Do some computation
          result = data["value"] ** 2

          # Write output
          with open("output.json", "w") as f:
              json.dump({"result": result}, f)

          print(f"Computed result: {result}")

      if __name__ == "__main__":
          main()

4. **Submit the task**

   .. code-block:: python

      from remote_handler.core.tasks import SingleScriptTask

      # Create the task
      task = SingleScriptTask(
          cluster_config=config,
          filesystem=remote_fs,
          script="my_script.py",
          jobname="my_job",
          n_processors=1,
      )

      # Upload script and input files
      task.allocate_data()

      # Submit to SLURM
      task.submit()

      # Wait for completion and collect results
      task.collect_and_harvest()

Advanced Usage
--------------

DFT Calculations with VASP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run VASP DFT calculations on structures:

.. code-block:: python

   from remote_handler.dft_runner import VASPHandler
   from remote_handler.core.cluster_handler import ClusterConfig
   from remote_handler.core.file_handler import RemoteFileSystem

   # Set up configuration
   config = ClusterConfig.from_file("my_cluster.json")
   remote_fs = RemoteFileSystem(remote_alias="datalab")

   # Create VASP handler
   vasp_handler = VASPHandler(
       traj_files=["structures.extxyz"],
       cluster_config=config,
       remote_fs=remote_fs,
       shard_size=50,  # Process 50 structures per job
       num_parallel=10,  # Submit 10 parallel jobs
       incar_kwargs={
           "xc": "PBE",
           "encut": 400,
           "prec": "Accurate",
           "ediff": 1e-5,
       },
   )

   # Run the calculations
   vasp_handler.allocate_data()
   vasp_handler.submit()
   vasp_handler.collect_and_harvest()

MACE Model Training
~~~~~~~~~~~~~~~~~~~~

Train MACE machine learning potentials:

.. code-block:: python

   from remote_handler.workflows.trainers import MaceModelTrainer
   from remote_handler.core.cluster_handler import ClusterConfig
   from remote_handler.core.file_handler import RemoteFileSystem

   # Set up configuration
   config = ClusterConfig.from_file("my_cluster.json")
   remote_fs = RemoteFileSystem(remote_alias="datalab")

   # Create MACE trainer
   mace_trainer = MaceModelTrainer(
       training_script="train_mace.py",
       config_file="mace_config_template.yaml",
       remote_mace_train_runner="/path/to/mace/run_train.py",
       cluster_config=config,
       filesystem=remote_fs,
       num_models=5,  # Train 5 models with different seeds
       results_dir="mace_models",
   )

   # Train models
   mace_trainer.allocate_data()
   mace_trainer.train()

LAMMPS Molecular Dynamics Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run LAMMPS MD simulations with uncertainty-based sampling:

.. code-block:: python

   from remote_handler.workflows.samplers import LammpsSampler
   from remote_handler.core.cluster_handler import ClusterConfig
   from remote_handler.core.file_handler import RemoteFileSystem

   # Set up configuration
   config = ClusterConfig.from_file("my_cluster.json")
   remote_fs = RemoteFileSystem(remote_alias="datalab")

   # Create LAMMPS sampler
   lammps_sampler = LammpsSampler(
       sampling_script="lammps/lammps_sampling.py",
       template_path="lammps_template.in",
       data_path="initial_structure.data",
       remote_lammps_executable_path="/usr/local/bin/lmp",
       cluster_config=config,
       filesystem=remote_fs,
       num_samples=1000,  # Generate 1000 configurations
       results_dir="lammps_samples",
   )

   # Run sampling
   lammps_sampler.allocate_data()
   lammps_sampler.sample(model_ensemble=trained_models)

Active Learning Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine sampling, evaluation, and training into a complete active learning loop:

.. code-block:: python

   from remote_handler.workflows.active_learning_base import ActiveLearningHandler
   from remote_handler.workflows.samplers import LammpsSampler
   from remote_handler.workflows.evaluators import DFTEvaluator
   from remote_handler.workflows.trainers import MaceModelTrainer
   from remote_handler.core.cluster_handler import ClusterConfig
   from remote_handler.core.file_handler import RemoteFileSystem

   # Set up configuration
   config = ClusterConfig.from_file("my_cluster.json")
   remote_fs = RemoteFileSystem(remote_alias="datalab")

   # Configure components
   sampler = LammpsSampler(
       sampling_script="lammps/lammps_sampling.py",
       template_path="lammps.in.j2",
       data_path="start.data",
       remote_lammps_executable_path="/usr/local/bin/lmp",
       cluster_config=config,
       filesystem=remote_fs,
   )

   evaluator = DFTEvaluator(
       cluster_config=config,
       filesystem=remote_fs,
       vasp_kwargs={"encut": 400, "xc": "PBE"},
   )

   trainer = MaceModelTrainer(
       training_script="train_mace.py",
       config_file="mace_config.yaml.j2",
       remote_mace_train_runner="/path/to/run_train.py",
       cluster_config=config,
       filesystem=remote_fs,
       num_models=5,
   )

   # Create active learning handler
   al_handler = ActiveLearningHandler(
       sampler=sampler,
       evaluator=evaluator,
       trainer=trainer,
       initial_training_data="initial_data.extxyz",
       max_iterations=10,
   )

   # Run the active learning loop
   al_handler.run()

This will automatically:

1. Train initial models on your starting data
2. Sample new configurations using LAMMPS with the current models
3. Evaluate high-uncertainty configurations with DFT
4. Retrain models with the augmented dataset
5. Repeat until convergence or max iterations

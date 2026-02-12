Quick Start
===========

This guide will get you started with using remote_handler for your computational workflows.

Basic Workflow
--------------

1. **Configure your cluster**

   Create a configuration file or use ``ClusterConfig``:

   .. code-block:: python

      from remote_handler.core import ClusterConfig

      config = ClusterConfig(
          remote_machine="datalab",
          partition="normal",
          conda_env="jaxnest",
          conda_source="~/miniconda3/bin/activate",
          savepath="/scratch/username",
          template="datalab.sh.j2"
      )

2. **Set up file system handler**

   .. code-block:: python

      from remote_handler.core.file_handler import RemoteFileSystem

      remote_fs = RemoteFileSystem(remote_alias="datalab")

3. **Create a handler for your task**

   .. code-block:: python

      from remote_handler import RemoteOptimizationHandler

      handler = RemoteOptimizationHandler(
          traj_files=["trajectory.extxyz"],
          cluster_config=config,
          remote_fs=remote_fs,
          shard_size=100,
          num_parallel=10,
          n_processors=4,
      )

4. **Execute the workflow**

   .. code-block:: python

      # Allocate resources
      handler.allocate_data()

      # Submit jobs
      handler.submit()

      # Collect results
      handler.collect_and_harvest()

Advanced Usage
--------------

Active Learning Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from remote_handler.workflows import ActiveLearningWorkflow

   workflow = ActiveLearningWorkflow(
       initial_data="training.extxyz",
       cluster_config=config,
       num_iterations=10,
   )

   workflow.run()

DFT Calculations
~~~~~~~~~~~~~~~~

.. code-block:: python

   from remote_handler.dft_runner import VASPHandler

   dft_handler = VASPHandler(
       traj_files=["structures.extxyz"],
       cluster_config=config,
       remote_fs=remote_fs,
       incar_kwargs={
           "xc": "PBE",
           "encut": 400,
           "prec": "Accurate",
       },
   )

   dft_handler.run()

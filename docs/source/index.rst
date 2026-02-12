Welcome to remote_handler's documentation!
==========================================

**remote_handler** is a Python package for managing SLURM array jobs on remote HPC clusters.
It provides a high-level interface for submitting, monitoring, and collecting results from
computational workflows involving atomic simulations, DFT calculations, and active learning.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples

Features
--------

- **Remote Execution**: Submit and manage SLURM jobs on remote clusters via SSH
- **Array Jobs**: Efficient batch processing using SLURM array jobs
- **File Management**: Automatic handling of input/output file transfers
- **Workflows**: Support for active learning and optimization workflows
- **Type Safety**: Full type hints for better IDE support

Quick Example
-------------

.. code-block:: python

   from remote_handler import RemoteOptimizationHandler
   from remote_handler.core import ClusterConfig

   # Configure your cluster
   config = ClusterConfig.from_file("my_cluster.json")

   # Set up optimization handler
   handler = RemoteOptimizationHandler(
       traj_files=["trajectory.extxyz"],
       cluster_config=config,
       shard_size=100,
       num_parallel=10,
   )

   # Run the optimization workflow
   handler.run()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

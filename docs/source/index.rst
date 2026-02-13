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

Quick Example
-------------

.. code-block:: python

   from remote_handler.dft_runner import VASPHandler
   from remote_handler.core import ClusterConfig, RemoteFileSystem

   # Configure your cluster
   config = ClusterConfig.from_file("my_cluster.json")
   remote_fs = RemoteFileSystem(remote_alias="datalab")

   # Set up VASP calculations
   vasp_handler = VASPHandler(
       traj_files=["structures.extxyz"],
       cluster_config=config,
       remote_fs=remote_fs,
       incar_kwargs={
           "xc": "PBE",
           "encut": 400,
           "prec": "Accurate",
       },
   )

   # Run DFT calculations
   vasp_handler.run()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

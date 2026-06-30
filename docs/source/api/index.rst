API Reference
=============

This section contains the complete API reference for remote_handler.

Core Modules
------------

Core modules provide the foundational functionality for cluster interaction,
file management, and task execution.

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: remote_handler.config
   :members:
   :undoc-members:
   :show-inheritance:

Cluster Configuration
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: remote_handler.core.cluster_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: remote_handler.core.cluster_handler
   :members:
   :undoc-members:
   :show-inheritance:

File System Handlers
~~~~~~~~~~~~~~~~~~~~

.. automodule:: remote_handler.core.file_handler
   :members:
   :undoc-members:
   :show-inheritance:

Task Management
~~~~~~~~~~~~~~~

.. automodule:: remote_handler.core.tasks
   :members:
   :undoc-members:
   :show-inheritance:

SLURM Utilities
~~~~~~~~~~~~~~~

.. automodule:: remote_handler.core.slurm_utils
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Environment

Handlers
--------

Atoms Handler
~~~~~~~~~~~~~

.. automodule:: remote_handler.core.atoms_handler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: remote_handler.core.atoms_shard
   :members:
   :undoc-members:
   :show-inheritance:

Trajectory Processing Handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimization and structural analysis are separate sibling handlers
(``RemoteOptimizationHandler`` / ``RemoteAnalysisHandler``) over a shared
sharding/collection base.

.. automodule:: remote_handler.workflows.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

DFT Runner
~~~~~~~~~~

.. automodule:: remote_handler.workflows.dft_runner
   :members:
   :undoc-members:
   :show-inheritance:

Workflows
---------

Active Learning
~~~~~~~~~~~~~~~

.. automodule:: remote_handler.workflows
   :members:
   :undoc-members:
   :show-inheritance:

Evaluators
~~~~~~~~~~

.. automodule:: remote_handler.workflows.evaluators
   :members:
   :undoc-members:
   :show-inheritance:

Samplers
~~~~~~~~

.. automodule:: remote_handler.workflows.samplers
   :members:
   :undoc-members:
   :show-inheritance:

Trainers
~~~~~~~~

.. automodule:: remote_handler.workflows.trainers
   :members:
   :undoc-members:
   :show-inheritance:

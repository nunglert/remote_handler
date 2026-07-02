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

Trajectory Processing
~~~~~~~~~~~~~~~~~~~~~~~

``remote_handler.processing`` is a self-contained plugin framework: a single
composable core (``RemoteTrajectoryProcessor``) runs an ordered list of
``Processor`` steps over a shared sharding/collection base, and
``RemoteOptimizationHandler`` is a thin factory for the optimize step.
Structural analysis is just the core with the processors you want (e.g.
``processors=[RDFProcessor(), SteinhardtProcessor()]``). Each built-in processor
keeps its arg schema, cluster-side ``run``, and host-side ``Processor`` spec in
one module under ``processing/processors/``.

.. automodule:: remote_handler.processing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: remote_handler.processing.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: remote_handler.processing.handler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: remote_handler.processing.registry
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

Installation
============

Requirements
------------

- Python 3.10 or higher
- SSH access to the remote machine you want to run on (resolved through your
  ``~/.ssh/config``; key-based auth and ``ProxyJump`` are supported)
- A scheduler is **optional**: the default ``slurm`` executor needs ``sbatch``
  on the remote, but the ``direct`` executor runs scripts scheduler-free (see
  :doc:`cli`)

The core install is lightweight -- it pulls in only ``ase``, ``numpy``,
``paramiko``, ``jinja2``, ``natsort`` and ``pyyaml``. Everything ML-specific
(MACE, NeuralIL, ...) lives behind the optional extras below, so the
host driver stays slim; the heavy stacks only need to be present in the *remote*
conda environment where the payload scripts actually run.

From Source
-----------

Clone the repository and install in editable (development) mode:

.. code-block:: bash

   git clone git@gitlab.tuwien.ac.at:e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler.git
   cd remote_handler
   pip install -e .

For development with testing and documentation:

.. code-block:: bash

   pip install -e ".[dev,docs]"

The ``docs`` extra also installs ``sphinx-autobuild`` for a live preview that
rebuilds and refreshes the browser on every save:

.. code-block:: bash

   sphinx-autobuild docs/source docs/build/html --open-browser

A one-off static build (no live reload) is ``cd docs && make html``.

Optional dependencies (extras)
------------------------------

Extras are selected with the ``.[extra1,extra2]`` syntax, e.g.
``pip install -e ".[mace,mace_jax]"``. Each one unlocks a particular workflow
actor or remote-side payload:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Extra
     - Enables
   * - ``dev``
     - ``pytest`` + ``pre-commit`` for running the test suite and hooks
   * - ``docs``
     - Building these Sphinx docs (``cd docs && make html``)
   * - ``mace``
     - ``MaceModelTrainer`` (train a MACE force field)
   * - ``mace_freeze``
     - ``MaceFinetuningTrainer`` (layer-freezing finetune)
   * - ``mace_jax``
     - ``MaceFinetuningTrainer(convert_to_jax=True)`` (torch ``.model`` →
       mace-jax ``.pkl``)
   * - ``neuralil``
     - ``NeuralilILTrainer`` (NeuralIL force-field training)
   * - ``clinamen``
     - CMA-ES structure sampling (``scripts/clinamen2``)
   * - ``quests``
     - QUESTS-based config selection (``workflows/selectors.py``)

.. warning::

   **Do not mix mace sources in one environment.** ``mace`` pulls
   ``mace-torch`` from PyPI and ``mace_freeze`` from the git fork -- the *same*
   package from *conflicting* sources. Combining them (``.[mace,mace_freeze]``)
   does not error cleanly (both report the same version), so it can silently
   install the fork; keep each mace source in a separate environment.

.. note::

   ``mace_freeze``, ``neuralil`` and ``mace_jax`` are installed straight from
   git, so those installs need network access to the respective repositories.

Common combinations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Host driver only (orchestrate jobs; ML stacks live on the remote)
   pip install -e .

   # Train MACE models
   pip install -e ".[mace]"

   # Finetune (freeze fork) and convert the result to mace-jax
   pip install -e ".[mace_freeze,mace_jax]"

   # NeuralIL training
   pip install -e ".[neuralil]"

.. tip::

   In a typical active-learning run the host only orchestrates; the MACE /
   NeuralIL packages need to be installed in the **remote** conda
   environment named by your cluster config's ``conda_env``, not necessarily on
   the host. Install the matching extra on the host only when you import those
   actors there (e.g. for ``stub=True`` dry-runs or local testing).

Configuration
-------------

.. tip::

   The quickest way to set this up is the interactive CLI:
   ``remote_handler configure cluster`` generates a valid cluster config and
   ``remote_handler configure`` persists global settings so the environment
   variables below become optional. See :doc:`cli`. The sections below
   describe the underlying files and variables it manages.

Cluster Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a cluster configuration file (e.g., ``my_cluster.json``):

.. code-block:: json

   {
       "remote_machine": "datalab",
       "partition": "normal",
       "conda_env": "jaxnest",
       "conda_source": "~/miniconda3",
       "savepath": "/home/username/scratch",
       "template": "datalab.sh.j2"
   }

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

remote_handler uses environment variables for configuration management:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Variable
     - Description
     - Default
   * - ``REMOTE_HANDLER_CONFIG_DIR``
     - Directory containing cluster configuration JSON files. Used when loading configs with ``ClusterConfig.from_file("name.json")``
     - None (optional)
   * - ``REMOTE_HANDLER_SLURM_TEMPLATES``
     - Directory containing custom SLURM job template files (Jinja2 ``.sh.j2`` files). Custom templates override built-in templates.
     - Built-in templates only

**Example setup:**

.. code-block:: bash

   # In your ~/.bashrc or ~/.zshrc
   export REMOTE_HANDLER_CONFIG_DIR="$HOME/.config/remote_handler"
   export REMOTE_HANDLER_SLURM_TEMPLATES="$HOME/.config/remote_handler/templates"

**Note:** Both variables are optional. If ``REMOTE_HANDLER_CONFIG_DIR`` is not set,
you'll need to provide absolute paths when loading configuration files.

SSH Setup
---------

Ensure you have SSH access configured to your cluster. Add your cluster
to ``~/.ssh/config`` for convenient access:

.. code-block:: text

   Host datalab
       HostName datalab.example.edu
       User username

**Note:** Having the cluster configured in your SSH config is mandatory. The ``IdentityFile`` directive is optional and will use your default
SSH key (~/.ssh/id_rsa) if not specified. Only add it if you need to use a
specific key file for this host.

For clusters with multi-factor authentication (MFA), you need a tool like
`smallstep <https://smallstep.com/docs/step-cli/>`_ to obtain short-lived SSH
credentials. Pair this with the opt-in ``resilient_connection`` setting on
:class:`~remote_handler.core.cluster_config.ClusterConfig`, which pauses and
retries instead of crashing when such a credential expires mid-run.

Installation
============

Requirements
------------

- Python 3.10 or higher
- SSH access to a SLURM cluster
- ASE (Atomic Simulation Environment)

From Source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone <your-gitlab-repo-url>
   cd remote_handler
   pip install -e .

For development with testing and documentation:

.. code-block:: bash

   pip install -e ".[dev,docs]"

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

**Note:** The ``IdentityFile`` directive is optional and will use your default
SSH key (~/.ssh/id_rsa) if not specified. Only add it if you need to use a
specific key file for this host.

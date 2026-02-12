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

Create a cluster configuration file (e.g., ``my_cluster.json``):

.. code-block:: json

   {
       "remote_machine": "datalab",
       "partition": "normal",
       "conda_env": "jaxnest",
       "conda_source": "~/miniconda3/etc/profile.d/conda.sh",
       "savepath": "/home/username/scratch",
       "template": "datalab.sh.j2"
   }

SSH Setup
---------

Ensure you have SSH keys configured for passwordless access:

.. code-block:: bash

   ssh-copy-id username@your-cluster.edu

Add your cluster to ``~/.ssh/config``:

.. code-block:: text

   Host datalab
       HostName datalab.example.edu
       User username
       IdentityFile ~/.ssh/id_rsa

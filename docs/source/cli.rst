Command-Line Interface
======================

Installing the package exposes a ``remote_handler`` console command that
configures and diagnoses your setup interactively. It is the recommended way
to get started — it writes a **persistent configuration file** so you no
longer need to export environment variables in every shell session.

.. code-block:: bash

   pip install -e .        # makes the `remote_handler` command available

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Purpose
   * - ``remote_handler configure``
     - Build the **global** configuration and persist it.
   * - ``remote_handler configure cluster [NAME]``
     - Create or edit a per-cluster ``ClusterConfig`` JSON.
   * - ``remote_handler check [--connect]``
     - Validate the configuration and diagnose remote setup.
   * - ``remote_handler show [--json]``
     - Print the resolved configuration and where it came from.
   * - ``remote_handler --version``
     - Print the package version.

Where configuration lives
-------------------------

The global config is stored at
``$XDG_CONFIG_HOME/remote_handler/config.json`` (falling back to
``~/.config/remote_handler/config.json``). Per-cluster JSON files live in the
same directory by default, so ``ClusterConfig.from_file("lugus.json")`` works
with no further setup.

Configuration is resolved with the following precedence (highest first):

#. Environment variables (``REMOTE_HANDLER_CONFIG_DIR``,
   ``REMOTE_HANDLER_SLURM_TEMPLATES``)
#. The persistent config file written by ``remote_handler configure``
#. Built-in defaults

Environment variables therefore still win when set, so existing setups keep
working — the persistent file simply removes the *requirement* to export them.

First-time setup
----------------

A typical first run needs only the per-cluster step:

.. code-block:: bash

   $ remote_handler configure cluster lugus
     1) datalab
     2) lugus
     3) vsc5
     (or type a custom value)
   Remote SSH alias (from ~/.ssh/config): 2
   SLURM partition [normal]: normal
   Conda environment [base]: jaxnest
   Conda activation source [~/miniconda3/bin/activate]:
   Remote savepath (where computations are stored): /home/me/scratch
     1) datalab.sh.j2
     2) dirona.sh.j2
     (or type a custom value)
   SLURM header template: 2
   Wrote /home/me/.config/remote_handler/lugus.json

The resulting file always contains all six required ``ClusterConfig`` fields
(``remote_machine``, ``partition``, ``conda_env``, ``conda_source``,
``savepath``, ``template``) and is validated before being written, so it is
guaranteed to load:

.. code-block:: python

   from remote_handler.core.cluster_handler import ClusterConfig

   config = ClusterConfig.from_file("lugus.json")   # no env var needed

``configure cluster``
---------------------

Prompts for the six per-cluster fields. The SSH alias is offered as a
pick-list read from ``~/.ssh/config`` (type a number, the literal value, or a
custom alias); the template is offered as a pick-list of the available SLURM
header templates. Pass a name as an argument
(``remote_handler configure cluster lugus``) or you will be prompted for one.
Re-running against an existing file pre-fills the answers and asks before
overwriting. Defaults for ``partition``, ``conda_env`` and ``conda_source``
come from the global configuration.

``configure`` (global)
----------------------

Prompts for the global defaults and writes the persistent config file:

* ``config_dir`` — directory holding the per-cluster JSON files
* ``slurm_templates_dir`` — optional directory of custom ``.sh.j2`` templates
* ``default_partition`` / ``default_conda_env`` / ``default_conda_source`` —
  defaults offered when creating new cluster configs

.. code-block:: bash

   $ remote_handler configure
   Config directory (where cluster JSONs live) [~/.config/remote_handler]:
   Custom SLURM templates dir (blank = built-in only) []:
   Default SLURM partition [normal]:
   Default conda environment [base]:
   Default conda activation source [~/miniconda3/bin/activate]:
   Wrote /home/me/.config/remote_handler/config.json
   Configuration persisted; REMOTE_HANDLER_CONFIG_DIR is no longer required.

``check``
---------

Validates the configuration and reports ``PASS`` / ``FAIL`` / ``WARN`` per
step. It answers the common question "why are my remote tests skipping?":

.. code-block:: bash

   $ remote_handler check
   [PASS] Configuration resolved
   [INFO] Persistent file: ~/.config/remote_handler/config.json (present)
   [PASS] Config dir: ~/.config/remote_handler
   [PASS] Found 1 cluster config(s)
   [PASS] lugus.json: loads
   [PASS] lugus.json: alias 'lugus' resolves

Local checks run offline: that the config resolves, the config directory
exists, each cluster JSON loads, the SSH alias resolves in ``~/.ssh/config``,
and the template is found. The command exits non-zero if any check fails.

Add ``--connect`` to also open a live SSH connection and probe the remote for
``sbatch`` and the configured conda environment:

.. code-block:: bash

   $ remote_handler check --connect
   ...
   [PASS] lugus.json: sbatch at /usr/bin/sbatch
   [PASS] lugus.json: conda env 'jaxnest' present

The live probe degrades gracefully: an unreachable host, an offline network,
or the known paramiko ssh-agent certificate issue are reported as ``WARN``
(with the ``ssh-add -D`` remediation) rather than failing the command.

``show``
--------

Prints the resolved configuration plus its provenance (persistent file path,
any overriding environment variables, template search directories, and the
discovered cluster configs). Use ``--json`` for machine-readable output:

.. code-block:: bash

   $ remote_handler show
   remote_handler 0.1.0
   persistent config file: ~/.config/remote_handler/config.json (present)
   config_dir: ~/.config/remote_handler
   default_partition: normal
   default_conda_env: base
   default_conda_source: ~/miniconda3/bin/activate
   template search dirs:
     - .../remote_handler/templates
   clusters: lugus

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
   * - ``remote_handler configure test [NAME]``
     - Set which cluster the ``remote`` test tier targets.
   * - ``remote_handler check [--connect]``
     - Validate the configuration and diagnose remote setup.
   * - ``remote_handler show [--json]``
     - Print the resolved configuration and where it came from.
   * - ``remote_handler show templates``
     - List the registered SLURM templates and their file paths.
   * - ``remote_handler show clusters``
     - List the registered cluster configs and their file paths.
   * - ``remote_handler show extras [MACHINE ...]``
     - Probe importable extras — locally, or per machine (``local`` / clusters).
   * - ``remote_handler status [PATH]``
     - Report whether a detached task (by its manifest) has finished.
   * - ``remote_handler collect [PATH] [--wait] [--force]``
     - Fetch a detached task's results from its manifest.
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
   config = ClusterConfig.from_name("lugus")        # same, by registered name

``ClusterConfig.from_name("lugus")`` looks up ``lugus.json`` in the config
directory written by ``configure cluster`` and raises a helpful error listing
the available clusters if it is not registered.

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

Two optional fields are not prompted for -- add them to the JSON by hand:

* ``executor`` -- ``"slurm"`` (default) or ``"direct"`` (scheduler-free).
* ``environment`` -- a map of *application* name to the shell lines that set
  that application up **on this cluster** (module loads, ``source`` of an env
  script, ``export`` of paths). This is where cluster- and site-specific DFT
  setup belongs, instead of being hardcoded in an actor. Actors that need it
  look up their key -- ``VASPEvaluator`` uses ``"vasp"``, ``ORCAEvaluator``
  uses ``"orca"``. This is the supported home for the setup; the legacy
  per-call ``slurm_header_add`` still works and **takes precedence** when both
  are given:

  .. code-block:: json

     {
       "remote_machine": "lugus",
       "partition": "normal",
       "conda_env": "remote_handler",
       "conda_source": "/home/me/miniconda3/bin/activate",
       "savepath": "/home/me/dump/remote_handler",
       "template": "dirona.sh.j2",
       "environment": {
         "vasp": "module purge\nmodule load vasp/6.4.2_intel\nsource /opt/intel/oneapi/setvars.sh --force\nexport VASP_PP_PATH=/home/me/cellar/vasp/paw",
         "orca": "module load orca/6.1.0"
       }
     }

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

``configure test``
------------------

Records which cluster the ``remote`` test tier runs against, by writing a
``test_cluster`` entry into the global config. The test fixtures then load
``<test_cluster>.json`` instead of defaulting to ``remote.json``:

.. code-block:: bash

   $ remote_handler configure test lugus
   Set test_cluster to lugus (was (remote.json))

Pass the name as an argument or you will be prompted (the known clusters are
offered as a hint). Passing an empty value clears the setting, restoring the
``remote.json`` default. A warning is shown if the named cluster JSON does not
exist yet, but the setting is still written.

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
   test_cluster: (remote.json)
   template search dirs:
     - .../remote_handler/templates
   clusters: lugus

When ``REMOTE_HANDLER_CONFIG_DIR`` is set it is shown as an explicit override
line, and ``test_cluster`` reflects any value set via ``configure test``.

``show templates`` / ``show clusters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two sub-listings print the discovered resources as ``name<TAB>path`` pairs,
handy for scripting or confirming what the resolver sees:

.. code-block:: bash

   $ remote_handler show templates
   datalab.sh.j2	/.../remote_handler/templates/datalab.sh.j2
   dirona.sh.j2	/.../remote_handler/templates/dirona.sh.j2

   $ remote_handler show clusters
   lugus	~/.config/remote_handler/lugus.json

``show templates`` walks the template search path in priority order, so when
the same template name exists in both a custom directory and the built-ins, the
higher-priority one wins (matching how jobs actually resolve templates).

``show extras``
~~~~~~~~~~~~~~~

Reports which optional dependency groups (the ``[mace]``, ``[mace_jax]``, …
extras from :doc:`installation`) are usable in the current environment:

.. code-block:: bash

   $ remote_handler show extras
   remote_handler 0.4.0 (local, env: rh_dev)

   Probing optional extras (imports torch / jax / ...; may be slow):

     [x] dev
     [ ] mace_jax  (missing: mace_jax)
     [x] mace
     ...

The first line reports the installed ``remote_handler`` version and the active
conda env it was read from — handy for catching a **stale install** (e.g. a
cluster conda env running an older copy than your checkout).

Unlike the other ``show`` commands, this does not just inspect metadata — it
**actually imports** each extra's key modules and reports what genuinely works.
That means it loads heavy stacks (torch, jax, …), so it is deliberately slower
than the other listings. Importing is also the only way to tell the stock
``mace`` distribution apart from the ``mace_freeze`` fork, which share a
distribution name. A ``[ ]`` line names the module(s) that failed to import, so
you can see exactly which piece is absent.

Probing other machines
^^^^^^^^^^^^^^^^^^^^^^^^

Pass one or more **targets** — ``local`` and/or registered cluster names — to
compare what is installed *where*. Each cluster is probed over SSH inside its
conda env (the host needs no ML stacks for the remote columns), and any binary
applications declared in the cluster's ``environment`` map (e.g. ``vasp``) are
checked on ``PATH`` after their setup runs:

.. code-block:: bash

   $ remote_handler show extras local lugus musica
   remote_handler version
     local   0.4.0  (env: rh_dev)
     lugus   0.0.0  (env: jaxrens_al)
     musica  0.4.0  (env: jaxrens_al)

   extra        local  lugus  musica
   ---------------------------------
   mace           ✓      ✓      ✓
   mace_jax       ✗      ✓      ✗
   torch_sim      ✓      ✓      ✓
   ---------------------------------
   vasp           ✗      ✓      —

The version block above the grid shows the installed ``remote_handler`` on each
machine (``?`` if unreachable) **and the conda env it was read from** — each
cluster is probed inside the env named in its config (``conda_env``), which may
differ from the one on your interactive ``PATH``. A cluster lagging behind your
checkout is then obvious at a glance. A version of ``0.0.0`` (as ``lugus`` shows
above) is the ``setuptools-scm`` fallback: that env was installed from a source
tree without git tags/history, so its real version is unknowable — reinstall it
from a proper ``git`` clone.

``✓`` available, ``✗`` not available, ``?`` machine unreachable, ``—`` the
binary is not configured on that cluster (no ``environment`` entry, so the tool
will not guess how to set it up). ``local`` has no ``environment`` map, so its
binary cells reflect the **current** local ``PATH`` (``shutil.which``) — which
is exactly right when ``local`` is itself a cluster node with modules loaded.
On a cluster, the app's ``environment`` setup is run first (in a login shell,
so ``module load`` works) before checking the binary. The command exits
non-zero if any requested machine could not be reached.

Detach a job, collect it later
------------------------------

Long remote jobs no longer need the launching process to stay alive. **By
default**, every task ``submit``/``run`` writes a small ``task_manifest.json``
(both locally, as ``<local_savepath>.manifest.json``, and inside the remote job
directory) right after submission and *before* the blocking wait — so even a
plain blocking run that dies mid-wait (SSH drop, laptop sleep, ``Ctrl-C``)
leaves a resume handle on disk and the running job is not orphaned. The write
is best-effort and never aborts the job; opt out with
``write_manifest_on_submit=False``.

A task can also **detach**: start the remote job, write the manifest, and
return immediately without waiting — so submitting feels like a local,
backgrounded command.

.. code-block:: python

   task = VASPEvaluator(...).build_task(...)   # any SingleScriptTask/ArrayScriptTask
   manifest_path = task.detach()               # fires the job, returns at once
   # ...or just run it blocking; the manifest is written either way.

The manifest is a self-contained resume handle: it embeds the full cluster
config and the resolved remote directory, so collecting later needs neither the
original process nor the cluster JSON still on disk. The day after, ``cd`` back
to where you launched (or point at the manifest) and retrieve the results:

.. code-block:: bash

   $ remote_handler status            # searches the cwd for a manifest
   task 'vasp'  [array/slurm, job 4815162]
     remote: /scratch/u/dump/vasp-xR3k9
     local:  ./evaluated_configs
     status: running

   $ remote_handler collect --wait    # block until done, then fetch
     status: running -- waiting for it to finish...
     collected results to ./evaluated_configs

``status`` exits ``0`` when finished and ``2`` while still running, so it drops
straight into a shell loop:

.. code-block:: bash

   until remote_handler status results.manifest.json; do sleep 60; done
   remote_handler collect results.manifest.json

``collect`` refuses to fetch a job that is still running unless you pass
``--wait`` (block until it finishes) or ``--force`` (fetch whatever files exist
now — handy when the remote env has since drifted and you only want the
outputs). ``PATH`` may be the manifest file itself or a directory containing
one; an ambiguous directory (more than one manifest) is an error rather than a
guess.

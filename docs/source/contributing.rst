Contributing
============

Thanks for considering a contribution! This page sketches how we expect
changes to land: how the project is laid out, the conventions we follow, and
what a reviewable merge request looks like.

Development setup
-----------------

Install the package editable with the development extras and enable the
pre-commit hooks:

.. code-block:: bash

   git clone <repo-url>
   cd remote_handler
   pip install -e ".[dev]"      # pytest + pytest-cov + pre-commit
   pre-commit install           # run black/isort/eof checks on every commit

Add the relevant ML extra(s) only if you touch code that imports them (see
:doc:`installation`). For docs work, add ``docs`` and use the live preview:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-autobuild docs/source docs/build/html --open-browser

Where code belongs
------------------

The package has three layers; please keep contributions on the right side of
these boundaries (see :doc:`api/index` for the module map):

``core/``
    Transport and execution primitives: SSH/SFTP, filesystem handlers, the
    executor abstraction (SLURM and scheduler-free ``direct`` backends), task
    definitions, ``ClusterConfig``. Host-side, dependency-light.

``workflows/``
    The active-learning loop and its actors (``Sampler`` /
    ``GroundTruthEvaluator`` / ``ModelTrainer`` and the concrete samplers,
    evaluators, trainers). Host-side orchestration.

``scripts/``
    **Remote payloads.** Code here is shipped to and executed on the cluster,
    in the remote conda environment -- not on the host. Each script validates
    its own input and resolves paths relative to its calc directory. Keep
    host-side orchestration (``core``/``workflows``) out of ``scripts`` and
    vice versa.

A good rule of thumb: if a new module needs ``mace`` / ``torch_sim`` /
``jax`` at import time, it almost certainly belongs in ``scripts/`` (behind an
extra), not in the host-side layers.

Code style
----------

Formatting is enforced by pre-commit, so you rarely need to think about it --
but for reference:

- **black** and **isort** (``--profile black``), **line length 79**.
- ``end-of-file-fixer`` and a large-file guard also run.
- Run everything manually with ``pre-commit run --all-files``.

A few project-specific conventions the formatters can't catch:

- **Raise the custom exceptions** from :mod:`remote_handler.exceptions`
  (``ConfigurationError``, ``FileSystemError``, ``TaskError``,
  ``WorkflowError``, ``DependencyError``, ``RemoteConnectionError``,
  ``ScriptInputError``) rather than bare ``Exception``.
- **Respect the pinned dependencies** (``numpy``, ``ase``, ``paramiko``,
  ``jinja2`` in the core deps; exact ML pins in the extras). If you touch
  ``scripts/``, match the versions the corresponding extra pins.
- **Prefer the config singleton**: read configuration through
  ``get_config()`` rather than re-reading environment variables ad hoc.

Docstrings
~~~~~~~~~~

Docstrings are Google-style and rendered by Sphinx + napoleon. To keep the
docs build warning-free:

- Put a **blank line before** ``Args:`` / ``Returns:`` / ``Example:`` sections.
- Don't hand-write ``Attributes:`` or ``Methods:`` sections that just relist
  members -- autodoc already documents them, and the duplication both drifts
  and raises warnings. Document a dataclass field with an inline ``#:`` comment
  instead.
- Wrap directory trees / ASCII art in a literal block (``::``) so characters
  like ``|`` aren't parsed as RST.
- Use raw strings (``r"""``) for docstrings containing backslashes (e.g. regex
  examples).

The docs build should stay at **zero warnings**; check with
``cd docs && make html`` before submitting doc changes.

Testing
-------

Tests live in ``test/`` and are organised into tiers by what the environment
must provide. The markers are registered in ``pyproject.toml``:

*(unmarked)*
    Pure-logic unit tests. No special deps. Always run.

``local``
    File-handler tests against the local filesystem (synthesized
    ``ClusterConfig``). No cluster needed.

``integration``
    Needs the ML stack (mace/torch_sim/clinamen). Deselected by default;
    individual tests self-skip when their specific dep is absent.

``slurm``
    Needs a working local ``sbatch``. Auto-skips when absent.

``sshlive``
    Real SSH/SFTP + scheduler-free execution against a throwaway sshd
    (``SSHLIVE_HOST`` or Docker). Auto-skips when no endpoint is reachable.

``remote``
    Needs a configured live cluster. Self-skips when unconfigured.

The default run (``pytest``) is ``-m 'not remote'``: every tier runs and
self-skips when its capability is absent; only the live-cluster ``remote`` tier
is deselected. Useful invocations:

.. code-block:: bash

   pytest                         # default: everything but the remote tier
   pytest -m integration          # ML end-to-end (needs the ML stack)
   pytest -m sshlive              # real SSH/SFTP round-trip
   pytest -k test_name -s --log-cli-level info   # one test, live logs

What we expect of a contribution:

- **New behavior comes with a test**, placed in the lightest tier that can
  exercise it -- prefer an unmarked unit test over an integration one when the
  logic can be isolated (the test suite runs with no configuration and no
  cluster, and we want to keep it that way).
- **Tests must self-skip, never hard-fail, when a capability is missing**
  (no Docker, no ``sbatch``, no ML stack). Follow the existing fixtures.
- CI enforces a coverage floor (``--cov-fail-under=50``); don't regress it.

Submitting changes
------------------

- Branch off ``main``; keep the change focused.
- Make sure ``pre-commit run --all-files`` and ``pytest`` are green locally,
  and that ``cd docs && make html`` is warning-free if you touched docs or
  docstrings.
- Open a merge request against ``main`` with a short description of *what* and
  *why*. CI runs the full suite (with every extra installed) on a clean
  ``python:3.13`` image plus a best-effort ``sshlive`` round-trip; the
  live-cluster ``remote`` tier self-skips.
- The ``pages`` job publishes the docs to GitLab Pages from ``main`` only, so
  doc changes go live once merged.

Active Learning Loops
======================

:doc:`concepts` introduced the active-learning cycle at a glance --
``Sampler -> GroundTruthEvaluator -> ModelTrainer``, closing the loop back
onto the sampler. This page fills that in properly: the actor interfaces
every concrete implementation satisfies, and the flexible, stage-based
:class:`~remote_handler.workflows.staged_active_learning.StagedActiveLearningHandler`
that drives them.

The actor interfaces
---------------------

Four roles, each an ``ABC`` in
:mod:`remote_handler.workflows.active_learning_base`, share one shape:
``update(calcdir, actor_input, **kwargs) -> None`` does the iteration's work
(writing into ``calcdir``, typically by building and running a
:class:`~remote_handler.core.tasks.RemoteTask`), and a getter hands back the
path(s) the next actor consumes.

.. mermaid::

   flowchart LR
       Sampler["Sampler<br/>get_sampled_configs()"] --> Selector["Selector<br/>get_selected_configs()"]
       Selector --> Eval["GroundTruthEvaluator<br/>get_evaluated_configs()"]
       Eval --> Trainer["ModelTrainer<br/>get_trained_model()"]
       Trainer -.->|"next iteration"| Sampler

- :class:`~remote_handler.workflows.active_learning_base.Sampler` proposes
  candidate configurations (e.g. by running MD under the current model).
- :class:`~remote_handler.workflows.active_learning_base.Selector` narrows
  that pool down before the (usually expensive) labelling step -- by
  diversity, uncertainty, novelty relative to already-labelled data, or a
  fixed stride. It is a peer of the other three actors, not a sampler
  sub-step, so it can run its own remote job when selection itself is heavy
  (e.g. a descriptor computation).
- :class:`~remote_handler.workflows.active_learning_base.GroundTruthEvaluator`
  labels the selected configurations (DFT via VASP/ORCA, or a cheap ASE
  calculator such as ``EMTEvaluator`` for testing).
- :class:`~remote_handler.workflows.active_learning_base.ModelTrainer`
  retrains the force field on the accumulated labelled data; the new model
  feeds the next sampler.

A :class:`~remote_handler.workflows.active_learning_base.Schedule` can vary
any actor's ``**kwargs`` from one iteration to the next (e.g. ramping a
sampler's temperature or a selector's ``n_configs``), and the optional
``pending_manifest`` hook is what lets a stage reattach to an
already-submitted job instead of resubmitting after the driving process dies
-- see :doc:`concepts` for that detach/reattach lifecycle.

Two ways to get a ``Selector``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most selection logic is a pure, host-side strategy -- score or filter a list
of ``ase.Atoms``, no remote job needed. That shape lives in
:mod:`remote_handler.workflows.selectors` as the
:class:`~remote_handler.workflows.selectors.ConfigSelector` interface
(``select(configs, reference=None) -> list[Atoms]``), with concrete
strategies such as stride, uncertainty, novelty (QUESTS) and pipelines of
several selectors -- see :doc:`api/index` for the full list.
:class:`~remote_handler.workflows.staged_active_learning.SelectorActor` wraps
any ``ConfigSelector`` into a ``Selector`` actor, so it slots into the loop
unchanged. Implement the ``Selector`` ABC directly only when selection needs
its own execution.

Stages: a pipeline, not a fixed triple
----------------------------------------

The original :class:`~remote_handler.workflows.active_learning_base.ActiveLearningHandler`
hardcodes the three-actor loop: its bookkeeping is a fixed
``[sample, eval, train]`` done-flag triple, so adding a fourth actor means
widening that triple by hand. ``StagedActiveLearningHandler`` instead models
the loop as an ordered list of uniform
:class:`~remote_handler.workflows.staged_active_learning.Stage` objects, each
wrapping one actor -- so adding the selector (or any other actor) is just
inserting one ``Stage``, with no changes to the orchestration code.

By default, stages form a ring: stage *N* consumes stage *N-1*'s output from
the same iteration, and the first stage consumes the last stage's output from
the *previous* iteration (closing the loop):

.. mermaid::

   flowchart LR
       S0["sampler<br/>(it. N)"] --> S1["selector<br/>(it. N)"]
       S1 --> S2["evaluator<br/>(it. N)"]
       S2 --> S3["trainer<br/>(it. N)"]
       S3 -.->|"it. N+1"| S0

Each ``Stage`` also has a ``combine`` flag: when true, its *consumer*
receives that stage's outputs from *all* completed iterations (plus the
consumer's seed) rather than only the most recent one -- e.g. an evaluator
whose labelled data should keep accumulating into the training set.

Beyond the ring, a stage may declare ``extra_inputs`` -- named secondary
streams pulled from *any* other stage's outputs via
:class:`~remote_handler.workflows.staged_active_learning.InputSource`,
optionally accumulated across iterations. This turns the ring into a small
DAG, which is what enables *novelty* selection: a selector receives the
evaluator's accumulated outputs from all previous iterations as a
``reference`` extra input and scores candidates by how novel they are
relative to the already-labelled data:

.. mermaid::

   flowchart LR
       S0["sampler"] --> S1["selector"]
       S1 --> S2["evaluator"]
       S2 --> S3["trainer"]
       S3 -.->|"it. N+1"| S0
       S2 -.->|"reference<br/>(accumulated)"| S1

Resume works the same way it does for tasks (see :doc:`concepts`): progress
is a cursor, not an enumerated flag combination. The handler precomputes a
deterministic plan of ``(iteration, stage)`` work items from
``num_iterations`` and the entry/exit stages, and records each completed
item's outputs in ``AL_progress.json``. Resuming re-runs only the plan items
not yet recorded, and each stage actor is transparently reattached to a
pending job rather than resubmitted, exactly as ``ActiveLearningHandler``
did -- but now generically over any stage (see below).

Stage actors don't have to be one of the four interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Stage``'s ``actor`` is not required to subclass ``Sampler`` /
``Selector`` / ``GroundTruthEvaluator`` / ``ModelTrainer`` at all -- those
ABCs exist for the four canonical roles, but the handler itself only ever
calls ``actor.update(calcdir, actor_input, **kwargs)`` and then looks up an
output getter by name (in order:
:func:`~remote_handler.workflows.staged_active_learning.resolve_actor_output`
tries ``get_output``, then the four actor-specific getters). Anything
exposing that shape works, so a stage that doesn't fit any of the four
roles -- a filtering step, a format conversion, an analysis/reporting step
you want checkpointed and resumable alongside the rest of the loop -- can be
plugged in directly by implementing ``update`` and ``get_output``, with no
interface to import. The optional reattach hooks
(``pending_manifest`` / ``collect_only``) are looked up the same
duck-typed way, so a plain object without them simply never gets
reattached, instead of erroring, and can add them later if it starts doing
real remote work.

Building a loop
-----------------

For the classic three-or-four-actor shape,
:meth:`~remote_handler.workflows.staged_active_learning.StagedActiveLearningHandler.from_actors`
builds the stage list for you and mirrors
``ActiveLearningHandler``'s own arguments, so existing scripts port with
minimal changes:

.. literalinclude:: ../../examples/local_emt_AL/run_al.py
   :language: python
   :lines: 96-107

Set ``selector_reference_from="evaluator"`` on ``from_actors`` to switch a
novelty-aware selector (such as
:class:`~remote_handler.workflows.selectors.QUESTSSelector`) into the DAG
shape above; ``initial_selector_reference`` seeds it so novelty scoring is
active from iteration 0 instead of falling back to diversity.

For anything beyond that shape -- extra actors, a different topology, custom
``extra_inputs`` -- construct the ``Stage`` list and
``StagedActiveLearningHandler`` directly; ``from_actors`` is a convenience
constructor, not a different code path. See :doc:`examples` for the full
runnable driver this snippet is taken from.

The legacy ``ActiveLearningHandler``
---------------------------------------

:class:`~remote_handler.workflows.active_learning_base.ActiveLearningHandler`
(the fixed sampler/evaluator/trainer triple, no selector) still exists,
untouched, but is deprecated -- it now raises a ``DeprecationWarning`` on
construction and will be removed in a future release. Use
``StagedActiveLearningHandler`` for new work; ``from_actors`` above is its
drop-in replacement.

Where to go next
------------------

- :doc:`concepts` -- the host/remote split, tasks and executors underneath
  every actor.
- :doc:`examples` -- the runnable, no-cluster AL loop this page's snippet is
  taken from.
- :doc:`api/index` -- the full ``Stage``/``InputSource``/selector-strategy
  reference.

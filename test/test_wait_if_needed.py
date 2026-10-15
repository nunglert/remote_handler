"""Unit tests for ``DetachableTaskMixin.wait_if_needed`` (no real cluster,
no real sleeping).

A reattached (``collect_only=True``) task must wait for its job to leave
the queue -- via the executor ``from_manifest`` re-attached -- before its
``finish_check()`` is applied once; only an unattached task (old manifest,
no job id) falls back to polling ``finish_check()`` itself. A fresh
(non-``collect_only``) task already blocked inside ``submit()``, so a single
check there is correct and must not turn into a poll loop.
"""

from unittest.mock import MagicMock

import pytest

from remote_handler.core import tasks as tasks_module
from remote_handler.core.tasks import DetachableTaskMixin
from remote_handler.exceptions import TaskError


class _Actor(DetachableTaskMixin):
    """Minimal stand-in for an actor mixing in DetachableTaskMixin."""

    def __init__(self, collect_only):
        self.collect_only = collect_only
        self.remote_fs = None


class FakeClock:
    """Deterministic clock: sleeping advances virtual time, never blocks."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds

    def monotonic(self):
        return self.t


def _patch_clock(monkeypatch, clock):
    monkeypatch.setattr(tasks_module.time, "sleep", clock.sleep)
    monkeypatch.setattr(tasks_module.time, "monotonic", clock.monotonic)


def _task(loop_time=30.0):
    task = MagicMock()
    task.loop_time = loop_time
    return task


# --------------------------------------------------------------------------
# Non-collect_only: unchanged, single-check behavior.
# --------------------------------------------------------------------------


def test_non_collect_only_finished_succeeds_without_sleeping(monkeypatch):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=False)
    task = _task()
    task.finish_check.return_value = True

    actor.wait_if_needed(task)

    task.finish_check.assert_called_once()
    assert clock.sleeps == []


def test_non_collect_only_unfinished_raises_immediately(monkeypatch):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=False)
    task = _task()
    task.finish_check.return_value = False

    with pytest.raises(TaskError, match="did not finish"):
        actor.wait_if_needed(task)

    task.finish_check.assert_called_once()
    assert clock.sleeps == []


def test_non_collect_only_uses_custom_error_message(monkeypatch):
    _patch_clock(monkeypatch, FakeClock())
    actor = _Actor(collect_only=False)
    task = _task()
    task.finish_check.return_value = False

    with pytest.raises(TaskError, match="custom failure text"):
        actor.wait_if_needed(task, error_message="custom failure text")


def test_finish_check_is_injectable(monkeypatch):
    """The array/fraction variant: finish_check takes required_fraction."""
    _patch_clock(monkeypatch, FakeClock())
    actor = _Actor(collect_only=False)
    task = _task()
    task.finish_check = MagicMock(return_value=True)

    actor.wait_if_needed(task, finish_check=lambda: task.finish_check(0.8))

    task.finish_check.assert_called_once_with(0.8)


# --------------------------------------------------------------------------
# collect_only: polls, bounded by max_wait, logging on each iteration.
# --------------------------------------------------------------------------


def test_collect_only_polls_until_finished(monkeypatch):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _task(loop_time=5.0)
    task.finish_check.side_effect = [False, False, True]

    actor.wait_if_needed(task)

    assert task.finish_check.call_count == 3
    # Slept once per not-yet-finished poll, using the task's own loop_time.
    assert clock.sleeps == [5.0, 5.0]


def test_collect_only_defaults_poll_interval_from_task_loop_time(
    monkeypatch,
):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _task(loop_time=12.0)
    task.finish_check.side_effect = [False, True]

    actor.wait_if_needed(task)

    assert clock.sleeps == [12.0]


def test_collect_only_explicit_poll_interval_overrides_loop_time(
    monkeypatch,
):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _task(loop_time=30.0)
    task.finish_check.side_effect = [False, True]

    actor.wait_if_needed(task, poll_interval=2.0)

    assert clock.sleeps == [2.0]


def test_collect_only_raises_after_max_wait_elapses(monkeypatch):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _task(loop_time=10.0)
    task.finish_check.return_value = False  # never finishes

    with pytest.raises(TaskError, match="did not finish"):
        actor.wait_if_needed(task, max_wait=25.0)

    # 0 -> sleep 10 -> 10 -> sleep 10 -> 20 -> sleep 10 -> 30 >= 25 -> raise
    assert clock.sleeps == [10.0, 10.0, 10.0]


def test_collect_only_max_wait_none_keeps_polling_past_a_finite_bound(
    monkeypatch,
):
    """No direct 'waits forever' test (that would hang); instead, confirm
    it keeps polling well past a bound that a finite max_wait test failed
    at, then succeeds once finish_check flips True."""
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _task(loop_time=10.0)
    # More iterations than the max_wait=25 test above would tolerate.
    task.finish_check.side_effect = [False] * 5 + [True]

    actor.wait_if_needed(task, max_wait=None)

    assert task.finish_check.call_count == 6
    assert clock.sleeps == [10.0] * 5


def test_collect_only_logs_on_each_poll(monkeypatch, caplog):
    import logging

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _task(loop_time=5.0)
    task.finish_check.side_effect = [False, False, True]

    with caplog.at_level(logging.INFO, logger=tasks_module.__name__):
        actor.wait_if_needed(task)

    poll_logs = [r for r in caplog.records if "not finished yet" in r.message]
    assert len(poll_logs) == 2


# --------------------------------------------------------------------------
# collect_only with an attached executor: wait on the queue, then check once.
# Regression for run25/run31: a reattached VASP array (required_fraction
# 0.5) was collected as soon as half of its is_finished.info markers
# existed, while the other half was still running.
# --------------------------------------------------------------------------


class _GraduallyFinishingArray:
    """A reattached array whose tasks finish one per poll.

    ``executor.is_finished()`` (the queue) only reports done once every
    task has run; each poll lets one more task write its marker.
    ``finish_check(required_fraction)`` counts markers like
    ``RemoteAtomsHandler.finish_check``.
    """

    def __init__(self, n_tasks, n_failed=0, loop_time=5.0):
        self.n_tasks = n_tasks
        self.n_failed = n_failed
        self.n_ran = 0
        self.loop_time = loop_time
        self.executor = MagicMock()
        self.executor.is_submitted = True
        self.executor.is_finished.side_effect = self._poll_queue
        self.fractions_seen = []

    def _poll_queue(self):
        if self.n_ran >= self.n_tasks:
            return True
        self.n_ran += 1
        return False

    def finish_check(self, required_fraction):
        markers = max(self.n_ran - self.n_failed, 0)
        fraction = markers / self.n_tasks
        self.fractions_seen.append(fraction)
        return fraction >= required_fraction


def test_collect_only_attached_waits_for_queue_not_fraction(monkeypatch):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _GraduallyFinishingArray(n_tasks=10)

    actor.wait_if_needed(task, finish_check=lambda: task.finish_check(0.5))

    # Kept waiting well past the 50% mark, until every task had run ...
    assert task.n_ran == 10
    assert len(clock.sleeps) == 10
    # ... and the fraction check ran exactly once, after the job was done.
    assert task.fractions_seen == [1.0]


def test_collect_only_attached_does_not_return_early_on_max_wait(
    monkeypatch,
):
    """Passing the fraction while the job is still queued is not success:
    hitting max_wait raises rather than collecting a partial job."""
    _patch_clock(monkeypatch, FakeClock())
    actor = _Actor(collect_only=True)
    task = _GraduallyFinishingArray(n_tasks=10, loop_time=10.0)

    with pytest.raises(TaskError, match="still not finished"):
        actor.wait_if_needed(
            task,
            finish_check=lambda: task.finish_check(0.5),
            max_wait=65.0,
        )

    assert task.n_ran < 10
    assert task.fractions_seen == []


def test_collect_only_attached_raises_if_fraction_fails_once_done(
    monkeypatch,
):
    _patch_clock(monkeypatch, FakeClock())
    actor = _Actor(collect_only=True)
    task = _GraduallyFinishingArray(n_tasks=10, n_failed=6)

    with pytest.raises(TaskError, match="custom failure text"):
        actor.wait_if_needed(
            task,
            finish_check=lambda: task.finish_check(0.5),
            error_message="custom failure text",
        )

    assert task.fractions_seen == [0.4]


def test_collect_only_attached_finished_job_checks_without_sleeping(
    monkeypatch,
):
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    actor = _Actor(collect_only=True)
    task = _GraduallyFinishingArray(n_tasks=3)
    task.n_ran = 3  # job already left the queue before the restart

    actor.wait_if_needed(task, finish_check=lambda: task.finish_check(0.5))

    assert clock.sleeps == []
    assert task.fractions_seen == [1.0]


def test_collect_only_unattached_warns_and_falls_back(monkeypatch, caplog):
    import logging

    _patch_clock(monkeypatch, FakeClock())
    actor = _Actor(collect_only=True)
    task = _task(loop_time=5.0)
    task.executor.is_submitted = False  # e.g. old manifest without job_id
    task.finish_check.side_effect = [False, True]

    with caplog.at_level(logging.WARNING, logger=tasks_module.__name__):
        actor.wait_if_needed(task)

    assert task.finish_check.call_count == 2
    task.executor.is_finished.assert_not_called()
    assert any("no attached job" in r.message for r in caplog.records)

"""Unit tests for ``DetachableTaskMixin.wait_if_needed`` (no real cluster,
no real sleeping).

Covers the gap described in ``experiments/remote_handler_reattach_wait_task.md``:
a reattached (``collect_only=True``) task has no scheduler-based wait to
lean on, so ``wait_if_needed`` must poll ``finish_check()`` itself; a fresh
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

"""Tests for scripts/selection/select_configs.py -- the remote-side payload
run by ``RemoteSelector`` (see workflows/staged_active_learning.py).

Runs the script in-process via ``runpy`` (as ``__main__``, with ``sys.argv``
pointed at a real input JSON in a temp calc dir) rather than as a real
subprocess, since ``RemoteSelector``'s own tests already exercise the full
remote-job path end to end; this focuses on the script's own logic.
"""

import json
import runpy
import sys
from pathlib import Path

import ase.io
import pytest
from ase import Atoms

from remote_handler.workflows import selectors as selectors_mod

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "remote_handler"
    / "scripts"
    / "selection"
    / "select_configs.py"
)


def _write_pool(path: Path, n: int) -> Path:
    frames = [
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7 + 0.05 * i]],
            cell=[6, 6, 6],
            pbc=True,
        )
        for i in range(n)
    ]
    ase.io.write(path, frames)
    return path


def _run_script(input_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), str(input_path)])
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


def test_select_configs_writes_output_and_marker(tmp_path, monkeypatch):
    _write_pool(tmp_path / "pool.extxyz", 10)
    input_path = tmp_path / "input_select.json"
    input_path.write_text(
        json.dumps(
            {
                "selector_type": "StrideSelector",
                "selector_kwargs": {"n_configs": 4},
                "input_files": ["pool.extxyz"],
            }
        )
    )

    _run_script(input_path, monkeypatch)

    selected = ase.io.read(tmp_path / "selected.extxyz", index=":")
    assert len(selected) == 4
    assert (tmp_path / "is_finished.info").exists()


def test_select_configs_reads_multiple_input_files(tmp_path, monkeypatch):
    _write_pool(tmp_path / "pool_a.extxyz", 5)
    _write_pool(tmp_path / "pool_b.extxyz", 5)
    input_path = tmp_path / "input_select.json"
    input_path.write_text(
        json.dumps(
            {
                "selector_type": "StrideSelector",
                "selector_kwargs": {"n_configs": 3},
                "input_files": ["pool_a.extxyz", "pool_b.extxyz"],
                "output_filename": "picked.extxyz",
            }
        )
    )

    _run_script(input_path, monkeypatch)

    selected = ase.io.read(tmp_path / "picked.extxyz", index=":")
    assert len(selected) == 3


def test_select_configs_forwards_reference_to_selector(tmp_path, monkeypatch):
    """A novelty-style selector receives the reference pool as ``Atoms``."""
    _write_pool(tmp_path / "pool.extxyz", 6)
    _write_pool(tmp_path / "ref.extxyz", 4)

    seen_reference_len = []

    class _RecordingSelector(selectors_mod.ConfigSelector):
        def __init__(self, n_configs):
            self.n_configs = n_configs

        def select(self, configs, reference=None):
            seen_reference_len.append(
                None if reference is None else len(reference)
            )
            flat = [a for g in configs for a in g]
            return flat[: self.n_configs]

    monkeypatch.setattr(
        selectors_mod, "_RecordingSelector", _RecordingSelector, raising=False
    )

    input_path = tmp_path / "input_select.json"
    input_path.write_text(
        json.dumps(
            {
                "selector_type": "_RecordingSelector",
                "selector_kwargs": {"n_configs": 2},
                "input_files": ["pool.extxyz"],
                "reference_files": ["ref.extxyz"],
            }
        )
    )

    _run_script(input_path, monkeypatch)

    assert seen_reference_len == [4]
    assert len(ase.io.read(tmp_path / "selected.extxyz", index=":")) == 2


def test_select_configs_missing_selector_type_raises(tmp_path, monkeypatch):
    _write_pool(tmp_path / "pool.extxyz", 5)
    input_path = tmp_path / "input_select.json"
    input_path.write_text(
        json.dumps(
            {
                "selector_type": "NoSuchSelector",
                "selector_kwargs": {},
                "input_files": ["pool.extxyz"],
            }
        )
    )

    with pytest.raises(AttributeError):
        _run_script(input_path, monkeypatch)

    # No success marker written when the selector can't even be resolved.
    assert not (tmp_path / "is_finished.info").exists()

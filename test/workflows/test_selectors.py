"""Tests for configuration selectors."""

import importlib.util
import json

import numpy as np
import pytest
from ase import Atoms

# QUESTS-based selectors need the optional 'quests' extra.
_HAS_QUESTS = importlib.util.find_spec("quests") is not None

from remote_handler.workflows.selectors import (
    ConfigSelector,
    ForceFilterSelector,
    LastFrameSelector,
    PipelineSelector,
    QUESTSSelector,
    RandomSelector,
    StrideSelector,
    UncertaintySelector,
    UniformForceSelector,
    UniformSelector,
)


def _make_atoms(energy_uncert=None, forces_uncert=None, info=None):
    """Helper to create a dummy Atoms object with optional uncertainty data."""
    atoms = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    atoms.cell = [5, 5, 5]
    atoms.pbc = True
    if energy_uncert is not None:
        atoms.info["energy_uncertainty"] = energy_uncert
    if forces_uncert is not None:
        atoms.arrays["forces_uncertainty"] = forces_uncert
    if info is not None:
        atoms.info.update(info)
    return atoms


def _make_config_pool(n, energy_uncerts=None, forces_uncerts=None):
    """Create a flat list of n Atoms with optional uncertainty arrays."""
    configs = []
    for i in range(n):
        eu = energy_uncerts[i] if energy_uncerts is not None else None
        fu = forces_uncerts[i] if forces_uncerts is not None else None
        configs.append(
            _make_atoms(energy_uncert=eu, forces_uncert=fu, info={"index": i})
        )
    return configs


def _make_grouped_configs(groups):
    """Create grouped configs: list[list[Atoms]].

    Args:
        groups: list of ints, each int is the number of frames in that group.
    """
    grouped = []
    idx = 0
    for n_frames in groups:
        group = []
        for _ in range(n_frames):
            group.append(_make_atoms(info={"index": idx}))
            idx += 1
        grouped.append(group)
    return grouped


# ---------------------------------------------------------------------------
# LastFrameSelector
# ---------------------------------------------------------------------------
class TestLastFrameSelector:
    def test_flat_input_returns_all(self):
        """Flat input: returns unchanged (each is already a 'last frame')."""
        configs = _make_config_pool(5)
        selector = LastFrameSelector()
        result = selector.select(configs)
        assert len(result) == 5

    def test_grouped_input_returns_last_per_group(self):
        """Grouped input: picks last frame from each trajectory."""
        grouped = _make_grouped_configs([3, 5, 2])
        # Last indices should be 2, 7, 9
        selector = LastFrameSelector()
        result = selector.select(grouped)
        assert len(result) == 3
        assert result[0].info["index"] == 2
        assert result[1].info["index"] == 7
        assert result[2].info["index"] == 9

    def test_grouped_single_frame_per_group(self):
        grouped = _make_grouped_configs([1, 1, 1])
        selector = LastFrameSelector()
        result = selector.select(grouped)
        assert len(result) == 3
        assert [at.info["index"] for at in result] == [0, 1, 2]

    def test_grouped_skips_empty_groups(self):
        grouped = [
            [_make_atoms(info={"index": 0})],
            [],
            [_make_atoms(info={"index": 1})],
        ]
        selector = LastFrameSelector()
        result = selector.select(grouped)
        assert len(result) == 2

    def test_empty_input(self):
        selector = LastFrameSelector()
        assert selector.select([]) == []


# ---------------------------------------------------------------------------
# StrideSelector
# ---------------------------------------------------------------------------
class TestStrideSelector:
    def test_selects_correct_count(self):
        configs = _make_config_pool(100)
        selector = StrideSelector(n_configs=10)
        result = selector.select(configs)
        assert len(result) == 10

    def test_selects_first_and_last(self):
        configs = _make_config_pool(50)
        selector = StrideSelector(n_configs=5)
        result = selector.select(configs)
        assert result[0].info["index"] == 0
        assert result[-1].info["index"] == 49

    def test_evenly_spaced(self):
        configs = _make_config_pool(100)
        selector = StrideSelector(n_configs=5)
        result = selector.select(configs)
        indices = [at.info["index"] for at in result]
        assert indices == [0, 24, 49, 74, 99]

    def test_n_configs_larger_than_pool(self):
        configs = _make_config_pool(3)
        selector = StrideSelector(n_configs=10)
        result = selector.select(configs)
        assert len(result) == 3

    def test_single_config(self):
        configs = _make_config_pool(1)
        selector = StrideSelector(n_configs=5)
        result = selector.select(configs)
        assert len(result) == 1

    def test_empty_input(self):
        selector = StrideSelector(n_configs=10)
        assert selector.select([]) == []

    def test_grouped_input_flattens(self):
        """Grouped input should be flattened before strided selection."""
        grouped = _make_grouped_configs([5, 5])  # 10 total
        selector = StrideSelector(n_configs=3)
        result = selector.select(grouped)
        assert len(result) == 3
        # Should span across both groups
        indices = [at.info["index"] for at in result]
        assert indices[0] == 0
        assert indices[-1] == 9


class TestUniformSelectorDeprecation:
    """UniformSelector is a deprecated alias for StrideSelector."""

    def test_warns_but_behaves_identically(self):
        configs = _make_config_pool(100)
        with pytest.warns(DeprecationWarning, match="StrideSelector"):
            old = UniformSelector(n_configs=5)
        # Same selection as the new name (deterministic evenly-spaced).
        new = StrideSelector(n_configs=5)
        assert [a.info["index"] for a in old.select(configs)] == [
            a.info["index"] for a in new.select(configs)
        ]

    def test_is_a_strideselector(self):
        with pytest.warns(DeprecationWarning):
            sel = UniformSelector(n_configs=3)
        assert isinstance(sel, StrideSelector)


# ---------------------------------------------------------------------------
# RandomSelector
# ---------------------------------------------------------------------------
class TestRandomSelector:
    def test_selects_correct_count(self):
        configs = _make_config_pool(20)
        selector = RandomSelector(n_configs=5)
        result = selector.select(configs)
        assert len(result) == 5

    def test_no_duplicates(self):
        configs = _make_config_pool(20)
        selector = RandomSelector(n_configs=10)
        result = selector.select(configs)
        indices = [at.info["index"] for at in result]
        assert len(set(indices)) == len(indices)

    def test_grouped_input_flattens(self):
        grouped = _make_grouped_configs([5, 5])  # 10 total
        selector = RandomSelector(n_configs=4)
        result = selector.select(grouped)
        assert len(result) == 4
        assert {at.info["index"] for at in result} <= set(range(10))

    def test_empty_input(self):
        selector = RandomSelector(n_configs=5)
        assert selector.select([]) == []

    def test_n_configs_larger_than_pool_raises(self):
        """Unlike StrideSelector/UncertaintySelector, RandomSelector does
        not clamp n_configs to the pool size: np.random.choice(...,
        replace=False) raises when asked for more samples than the pool
        has. Documents current behaviour -- callers must ensure
        n_configs <= pool size themselves."""
        configs = _make_config_pool(3)
        selector = RandomSelector(n_configs=10)
        with pytest.raises(ValueError):
            selector.select(configs)


# ---------------------------------------------------------------------------
# UncertaintySelector
# ---------------------------------------------------------------------------
class TestUncertaintySelector:
    def test_selects_highest_energy_uncertainty(self):
        energy_uncerts = [0.1, 0.5, 0.9, 0.2, 0.8]
        configs = _make_config_pool(5, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            n_configs=2, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)

        assert len(result) == 2
        selected_indices = {at.info["index"] for at in result}
        assert selected_indices == {2, 4}

    def test_selects_highest_forces_uncertainty(self):
        forces_uncerts = [np.ones((4, 3)) * (i + 1) * 0.1 for i in range(5)]
        configs = _make_config_pool(5, forces_uncerts=forces_uncerts)

        selector = UncertaintySelector(
            n_configs=2, energy_key=None, forces_key="forces_uncertainty"
        )
        result = selector.select(configs)

        assert len(result) == 2
        selected_indices = {at.info["index"] for at in result}
        assert selected_indices == {3, 4}

    def test_combined_energy_and_forces(self):
        energy_uncerts = [0.9, 0.1, 0.1, 0.1, 0.1]
        forces_uncerts = [np.ones((4, 3)) * (i + 1) * 0.1 for i in range(5)]
        configs = _make_config_pool(
            5, energy_uncerts=energy_uncerts, forces_uncerts=forces_uncerts
        )

        selector = UncertaintySelector(
            n_configs=2, energy_weight=1.0, forces_weight=1.0
        )
        result = selector.select(configs)
        assert len(result) == 2
        selected_indices = {at.info["index"] for at in result}
        assert selected_indices == {0, 4}

    def test_custom_weights(self):
        energy_uncerts = [1.0, 0.0, 0.0, 0.0, 0.0]
        forces_uncerts = [
            np.zeros((4, 3)),
            np.zeros((4, 3)),
            np.zeros((4, 3)),
            np.zeros((4, 3)),
            np.ones((4, 3)),
        ]
        configs = _make_config_pool(
            5, energy_uncerts=energy_uncerts, forces_uncerts=forces_uncerts
        )

        selector = UncertaintySelector(
            n_configs=1, energy_weight=0.1, forces_weight=10.0
        )
        result = selector.select(configs)
        assert result[0].info["index"] == 4

    def test_uncertainty_score_in_info(self):
        energy_uncerts = [0.1, 0.5, 0.9]
        configs = _make_config_pool(3, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            n_configs=3, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)

        for at in result:
            assert "uncertainty_score" in at.info

    def test_n_configs_larger_than_pool(self):
        energy_uncerts = [0.1, 0.5]
        configs = _make_config_pool(2, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            n_configs=10, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)
        assert len(result) == 2

    def test_missing_energy_key_raises(self):
        configs = _make_config_pool(3)

        selector = UncertaintySelector(
            n_configs=2, energy_key="energy_uncertainty", forces_key=None
        )
        with pytest.raises(KeyError, match="energy_uncertainty"):
            selector.select(configs)

    def test_missing_forces_key_raises(self):
        configs = _make_config_pool(3)

        selector = UncertaintySelector(
            n_configs=2, energy_key=None, forces_key="forces_uncertainty"
        )
        with pytest.raises(KeyError, match="forces_uncertainty"):
            selector.select(configs)

    def test_both_keys_none_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            UncertaintySelector(n_configs=2, energy_key=None, forces_key=None)

    def test_empty_input(self):
        selector = UncertaintySelector(
            n_configs=10, energy_key="energy_uncertainty", forces_key=None
        )
        assert selector.select([]) == []

    def test_uniform_uncertainty_returns_n_configs(self):
        energy_uncerts = [0.5] * 10
        configs = _make_config_pool(10, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            n_configs=3, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)
        assert len(result) == 3

    def test_grouped_input_flattens(self):
        """Grouped input should be flattened before uncertainty selection."""
        eu = [0.1, 0.9, 0.5, 0.8, 0.2, 0.7]
        flat = _make_config_pool(6, energy_uncerts=eu)
        grouped = [flat[:3], flat[3:]]

        selector = UncertaintySelector(
            n_configs=2, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(grouped)
        assert len(result) == 2
        selected_indices = {at.info["index"] for at in result}
        assert selected_indices == {1, 3}  # 0.9 and 0.8


# ---------------------------------------------------------------------------
# UncertaintySelector — threshold mode
# ---------------------------------------------------------------------------
class TestUncertaintySelectorThreshold:
    def test_threshold_only(self):
        """Pure threshold mode: select all configs above threshold."""
        energy_uncerts = [0.1, 0.5, 0.9, 0.2, 0.8]
        configs = _make_config_pool(5, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            threshold=0.5, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)

        # Normalized scores: [0.0, 0.5, 1.0, 0.125, 0.875]
        # Above 0.5: indices 1 (==0.5), 2 (1.0), 4 (0.875)
        assert len(result) == 3
        selected_indices = {at.info["index"] for at in result}
        assert selected_indices == {1, 2, 4}

    def test_threshold_with_n_configs_cap(self):
        """Threshold + n_configs: threshold filters, then cap at n_configs."""
        energy_uncerts = [0.1, 0.5, 0.9, 0.2, 0.8]
        configs = _make_config_pool(5, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            n_configs=2,
            threshold=0.5,
            energy_key="energy_uncertainty",
            forces_key=None,
        )
        result = selector.select(configs)

        # 3 configs above threshold, but capped at 2
        assert len(result) == 2
        selected_indices = {at.info["index"] for at in result}
        assert selected_indices == {2, 4}  # top-2 by score

    def test_threshold_none_selected(self):
        """All scores below threshold: returns empty list."""
        energy_uncerts = [0.1, 0.2, 0.3]
        configs = _make_config_pool(3, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            threshold=1.5, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)
        assert len(result) == 0

    def test_threshold_all_selected(self):
        """All scores above threshold: returns all configs."""
        energy_uncerts = [0.5, 0.6, 0.7, 0.8, 0.9]
        configs = _make_config_pool(5, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            threshold=0.0, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)
        assert len(result) == 5

    def test_threshold_uniform_scores(self):
        """Uniform scores: threshold at 0 selects all, threshold above 0 selects none."""
        energy_uncerts = [0.5] * 5
        configs = _make_config_pool(5, energy_uncerts=energy_uncerts)

        # Normalized scores are all 0.0 (zero range)
        selector = UncertaintySelector(
            threshold=0.0, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)
        assert len(result) == 5

        selector_high = UncertaintySelector(
            threshold=0.1, energy_key="energy_uncertainty", forces_key=None
        )
        result_high = selector_high.select(configs)
        assert len(result_high) == 0

    def test_both_none_raises(self):
        """Must provide at least one of n_configs or threshold."""
        with pytest.raises(ValueError, match="At least one of n_configs"):
            UncertaintySelector(
                n_configs=None,
                threshold=None,
                energy_key="energy_uncertainty",
            )

    def test_threshold_empty_input(self):
        selector = UncertaintySelector(
            threshold=0.5, energy_key="energy_uncertainty", forces_key=None
        )
        assert selector.select([]) == []

    def test_threshold_scores_in_info(self):
        """Selected configs should have uncertainty_score in info."""
        energy_uncerts = [0.1, 0.9]
        configs = _make_config_pool(2, energy_uncerts=energy_uncerts)

        selector = UncertaintySelector(
            threshold=0.5, energy_key="energy_uncertainty", forces_key=None
        )
        result = selector.select(configs)
        for at in result:
            assert "uncertainty_score" in at.info


def _make_force_atoms(magnitudes):
    """Create one Atoms per magnitude, with a uniform per-atom force of
    that magnitude (along x). Positions vary slightly per config so
    distinct configs never compare structurally equal under ASE's
    ``Atoms.__eq__`` (relevant for ``UniformForceSelector``'s ``in``
    check when topping up with ``force_n_configs``).
    """
    configs = []
    for i, mag in enumerate(magnitudes):
        atoms = Atoms(
            "H4",
            positions=[
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1 + 0.001 * i],
            ],
        )
        atoms.cell = [5, 5, 5]
        atoms.pbc = True
        atoms.arrays["forces"] = np.tile([mag, 0.0, 0.0], (len(atoms), 1))
        atoms.info["index"] = i
        configs.append(atoms)
    return configs


# ---------------------------------------------------------------------------
# ForceFilterSelector
# ---------------------------------------------------------------------------
class TestForceFilterSelector:
    def test_filters_above_threshold(self):
        configs = _make_force_atoms([0.5, 1.5, 2.5, 0.9, 3.0])
        selector = ForceFilterSelector(force_threshold=1.0)
        result = selector.select(configs)
        assert {at.info["index"] for at in result} == {0, 3}

    def test_all_below_threshold_keeps_all(self):
        configs = _make_force_atoms([0.1, 0.2, 0.3])
        selector = ForceFilterSelector(force_threshold=10.0)
        result = selector.select(configs)
        assert len(result) == 3

    def test_all_above_threshold_returns_empty(self):
        configs = _make_force_atoms([5.0, 6.0])
        selector = ForceFilterSelector(force_threshold=1.0)
        assert selector.select(configs) == []

    def test_boundary_is_inclusive(self):
        """max_force == threshold is kept (<=, not <)."""
        configs = _make_force_atoms([1.0])
        selector = ForceFilterSelector(force_threshold=1.0)
        result = selector.select(configs)
        assert len(result) == 1

    def test_empty_input(self):
        selector = ForceFilterSelector(force_threshold=1.0)
        assert selector.select([]) == []

    def test_grouped_input_flattens(self):
        flat = _make_force_atoms([0.5, 5.0, 0.3, 5.0])
        grouped = [flat[:2], flat[2:]]
        selector = ForceFilterSelector(force_threshold=1.0)
        result = selector.select(grouped)
        assert {at.info["index"] for at in result} == {0, 2}


# ---------------------------------------------------------------------------
# UniformForceSelector
#
# Two fixed bugs worth keeping in mind when reading these tests:
#   * Per-bin (and force_n_configs top-up) subsampling used to call
#     np.random.choice directly on a list of Atoms objects instead of on
#     their indices; since Atoms is itself sized/iterable, numpy's array
#     construction failed with "a must be 1-dimensional" for any pool
#     large enough to need real subsampling. Fixed by sampling indices and
#     mapping back to Atoms afterwards.
#   * np.digitize (right=False, the default) places a value exactly equal
#     to the top bin edge one index past the last valid bin, so the
#     config(s) with the pool's maximum force magnitude used to be
#     silently dropped. Fixed by clipping the digitized indices into
#     [0, n_bins - 1] -- the max now lands in the last bin like it should.
# ---------------------------------------------------------------------------
class TestUniformForceSelector:
    def test_selects_up_to_n_configs_across_bins(self):
        """20 configs spread over [0, 9.99) into 5 bins (4-5 per bin,
        including the max at the top edge). configs_per_bin = 10 // 5 = 2,
        so exactly 10 are selected (which bin members get picked is
        random when a bin exceeds its cap, but the total count isn't)."""
        magnitudes = list(np.linspace(0, 9.99, 20))
        selector = UniformForceSelector(n_configs=10, n_bins=5)
        for _ in range(5):  # re-run to also guard against flaky sampling
            result = selector.select(_make_force_atoms(magnitudes))
            assert len(result) == 10

    def test_max_value_lands_in_the_last_bin(self):
        """The config whose force equals the pool maximum is included,
        via the last bin -- not dropped. One config per bin (no
        oversubscription), so every one of the 5 is kept."""
        magnitudes = [0.0, 2.5, 5.0, 7.5, 10.0]  # 10.0 is the pool max
        configs = _make_force_atoms(magnitudes)
        selector = UniformForceSelector(n_configs=5, n_bins=5)
        result = selector.select(configs)
        selected_indices = {at.info["index"] for at in result}
        assert len(result) == 5
        assert selected_indices == {0, 1, 2, 3, 4}

    def test_empty_bins_excluded_from_configs_per_bin(self):
        """Two clusters of 3 with a gap: 3 of the 5 bins are empty and
        don't count towards the configs_per_bin division (denominator
        shrinks from n_bins=5 to the 2 non-empty bins) -> 4 // 2 = 2 per
        bin, capped from 3 each -> 4 selected total, deterministically,
        even though which 2-of-3 per cluster is random."""
        magnitudes = [0.0, 0.1, 0.2, 9.0, 9.1, 9.2]
        selector = UniformForceSelector(n_configs=4, n_bins=5)
        for _ in range(5):
            result = selector.select(_make_force_atoms(magnitudes))
            assert len(result) == 4

    def test_configs_per_bin_zero_returns_empty(self):
        """n_configs smaller than the number of non-empty bins rounds
        configs_per_bin down to 0 -> returns [], even though the bins
        themselves are not actually empty (the log message says
        otherwise)."""
        magnitudes = list(np.linspace(0, 9.0, 10))
        configs = _make_force_atoms(magnitudes)
        selector = UniformForceSelector(n_configs=2, n_bins=5)
        assert selector.select(configs) == []

    def test_uniform_magnitudes_all_land_in_the_last_bin(self):
        """When every candidate has the exact same average force, min ==
        max and every bin edge collapses to that value; the clip keeps
        all of them in the last bin (index n_bins - 1) instead of an
        out-of-range one, so they're all selected -- this used to return
        [] unconditionally (see class docstring)."""
        configs = _make_force_atoms([1.0, 1.0, 1.0, 1.0, 1.0])
        selector = UniformForceSelector(n_configs=5, n_bins=5)
        result = selector.select(configs)
        assert len(result) == 5

    def test_force_n_configs_false_stops_at_the_bin_cap(self):
        """5 configs at 0.0 (one bin, cap 2) + 1 at 9.0 (the max, its own
        bin): bins alone provide 2 + 1 = 3, short of the 4 requested, even
        though the pool has 6 configs to draw from. Without
        force_n_configs the selector stops there."""
        magnitudes = [0.0, 0.0, 0.0, 0.0, 0.0, 9.0]
        configs = _make_force_atoms(magnitudes)
        selector = UniformForceSelector(
            n_configs=4, n_bins=5, force_n_configs=False
        )
        result = selector.select(configs)
        assert len(result) == 3

    def test_force_n_configs_true_tops_up_past_the_bin_cap(self):
        """Same pool as above, but force_n_configs=True pulls the
        remaining slot from whatever the bin pass left out (one of the
        four 0.0-magnitude configs it didn't already pick), reaching the
        full n_configs=4."""
        magnitudes = [0.0, 0.0, 0.0, 0.0, 0.0, 9.0]
        configs = _make_force_atoms(magnitudes)
        selector = UniformForceSelector(
            n_configs=4, n_bins=5, force_n_configs=True
        )
        result = selector.select(configs)
        assert len(result) == 4

    def test_empty_input(self):
        selector = UniformForceSelector(n_configs=5)
        assert selector.select([]) == []

    def test_grouped_input_flattens(self):
        """Grouped input is flattened before binning."""
        flat = _make_force_atoms([0.0, 3.0, 6.0])  # 6.0 is the pool max
        grouped = [flat[:2], flat[2:]]
        selector = UniformForceSelector(n_configs=3, n_bins=3)
        result = selector.select(grouped)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# ForceFilterSelector / UniformForceSelector
# ---------------------------------------------------------------------------
def _make_atoms_with_forces(forces, index=None, position_offset=0.0):
    """Create an Atoms object carrying a real per-atom ``forces`` array.

    ``forces`` is ``list[list[float]]``, one ``[fx, fy, fz]`` per atom (atom
    count is inferred from its length). ``position_offset`` nudges each
    config's positions apart so configs with different ``index`` aren't
    value-equal under ASE's ``Atoms.__eq__`` (which compares positions/
    numbers/cell/pbc, not ``arrays["forces"]`` or ``info``) -- relevant for
    ``UniformForceSelector``'s ``at not in selected`` membership check.
    """
    n_atoms = len(forces)
    positions = [[i + position_offset, 0, 0] for i in range(n_atoms)]
    atoms = Atoms("H" * n_atoms, positions=positions)
    atoms.cell = [20, 20, 20]
    atoms.pbc = True
    atoms.arrays["forces"] = np.array(forces, dtype=float)
    if index is not None:
        atoms.info["index"] = index
    return atoms


class TestForceFilterSelector:
    def test_keeps_configs_at_or_below_threshold(self):
        configs = [
            _make_atoms_with_forces([[0.5, 0, 0]], index=0, position_offset=0),
            _make_atoms_with_forces([[2.0, 0, 0]], index=1, position_offset=1),
            _make_atoms_with_forces([[1.0, 0, 0]], index=2, position_offset=2),
        ]
        selector = ForceFilterSelector(force_threshold=1.0)
        result = selector.select(configs)
        indices = sorted(at.info["index"] for at in result)
        assert indices == [0, 2]  # 0.5 and 1.0 pass; 2.0 doesn't

    def test_threshold_is_inclusive(self):
        configs = [_make_atoms_with_forces([[1.0, 0, 0]], index=0)]
        selector = ForceFilterSelector(force_threshold=1.0)
        result = selector.select(configs)
        assert len(result) == 1

    def test_uses_max_atom_force_not_mean(self):
        # Per-atom magnitudes [3.0, 0.0, 0.0]: mean is 1.0 (<= threshold),
        # max is 3.0 (> threshold). A config should be excluded whenever any
        # single atom's force exceeds the threshold, not just on average.
        configs = [
            _make_atoms_with_forces(
                [[3.0, 0, 0], [0, 0, 0], [0, 0, 0]], index=0
            )
        ]
        selector = ForceFilterSelector(force_threshold=2.0)
        result = selector.select(configs)
        assert result == []

    def test_empty_input_returns_empty_list(self):
        selector = ForceFilterSelector(force_threshold=1.0)
        assert selector.select([]) == []

    def test_grouped_input_flattens_before_filtering(self):
        grouped = [
            [
                _make_atoms_with_forces(
                    [[0.5, 0, 0]], index=0, position_offset=0
                )
            ],
            [
                _make_atoms_with_forces(
                    [[2.0, 0, 0]], index=1, position_offset=1
                ),
                _make_atoms_with_forces(
                    [[0.2, 0, 0]], index=2, position_offset=2
                ),
            ],
        ]
        selector = ForceFilterSelector(force_threshold=1.0)
        result = selector.select(grouped)
        indices = sorted(at.info["index"] for at in result)
        assert indices == [0, 2]

    def test_reference_argument_is_ignored(self):
        configs = [_make_atoms_with_forces([[0.5, 0, 0]], index=0)]
        ref = [
            _make_atoms_with_forces([[9.0, 0, 0]], index=99, position_offset=5)
        ]
        selector = ForceFilterSelector(force_threshold=1.0)
        result = selector.select(configs, reference=ref)
        assert len(result) == 1


class TestUniformForceSelector:
    """Regression note: ``UniformForceSelector`` bins configs by average
    force using ``bins = np.linspace(avg.min(), avg.max(), n_bins + 1)`` and
    ``np.digitize(avg, bins) - 1``. Because ``np.digitize`` (default
    ``right=False``) treats a value equal to the *last* bin edge as falling
    past it, the single config with the largest average force always maps
    to bin index ``n_bins`` -- one past the ``range(n_bins)`` the selection
    loop actually iterates -- so it is silently dropped from every call with
    at least two distinct force magnitudes. Separately, whenever a bin holds
    more configs than its quota, ``np.random.choice(bin_configs, ...)`` is
    called directly on a ``list[Atoms]``; since ``Atoms`` defines
    ``__len__``/``__getitem__``, NumPy tries to build a 2D array instead of
    treating the list as 1D objects, and raises
    ``ValueError: a must be 1-dimensional``. Both are pre-existing bugs in
    the selector as merged, not something introduced by these tests -- they
    are captured below as strict xfails (still exercising, and covering,
    the buggy code paths) rather than silently worked around, so a future
    fix is picked up automatically (a strict xfail that starts passing
    fails the suite, flagging the marker for removal).
    """

    def test_empty_input_returns_empty_list(self):
        selector = UniformForceSelector(n_configs=5)
        assert selector.select([]) == []

    def test_zero_n_configs_returns_empty_list(self):
        configs = [
            _make_atoms_with_forces(
                [[float(i), 0, 0]], index=i, position_offset=i
            )
            for i in range(5)
        ]
        selector = UniformForceSelector(n_configs=0, n_bins=5)
        assert selector.select(configs) == []

    def test_all_equal_forces_returns_empty_list(self):
        # Degenerate case: avg.min() == avg.max() collapses every bin edge
        # to the same value, so np.digitize pushes every config past
        # range(n_bins) (the same boundary behavior described above) and
        # none are ever assigned to a bin -- documenting current behavior
        # for this edge case, not asserting it's the ideal outcome.
        configs = [
            _make_atoms_with_forces([[2.0, 0, 0]], index=i, position_offset=i)
            for i in range(3)
        ]
        selector = UniformForceSelector(n_configs=3, n_bins=3)
        assert selector.select(configs) == []

    def test_selects_across_bins_except_max_which_is_dropped(self):
        """Documents *current* (buggy) behavior: with avg forces 0/1/2 across
        3 bins, the max (index 2) is dropped per the digitize boundary bug
        above, leaving only indices 0 and 1."""
        configs = [
            _make_atoms_with_forces(
                [[float(i), 0, 0]], index=i, position_offset=i
            )
            for i in range(3)
        ]
        selector = UniformForceSelector(n_configs=3, n_bins=3)
        result = selector.select(configs)
        indices = sorted(at.info["index"] for at in result)
        assert indices == [0, 1]

    def test_grouped_input_flattens_before_binning(self):
        grouped = [
            [
                _make_atoms_with_forces(
                    [[0.0, 0, 0]], index=0, position_offset=0
                )
            ],
            [
                _make_atoms_with_forces(
                    [[1.0, 0, 0]], index=1, position_offset=1
                ),
                _make_atoms_with_forces(
                    [[2.0, 0, 0]], index=2, position_offset=2
                ),
            ],
        ]
        selector = UniformForceSelector(n_configs=3, n_bins=3)
        result = selector.select(grouped)
        indices = {at.info["index"] for at in result}
        # Configs from both groups are pooled together (index 2, the max,
        # is dropped per the known bug, not a flattening failure).
        assert indices == {0, 1}

    def test_reference_argument_is_ignored(self):
        configs = [
            _make_atoms_with_forces(
                [[float(i), 0, 0]], index=i, position_offset=i
            )
            for i in range(3)
        ]
        ref = [
            _make_atoms_with_forces([[9.0, 0, 0]], index=99, position_offset=9)
        ]
        selector = UniformForceSelector(n_configs=3, n_bins=3)
        result = selector.select(configs, reference=ref)
        indices = sorted(at.info["index"] for at in result)
        assert indices == [0, 1]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "np.random.choice(bin_configs, ...) is called directly on a "
            "list[Atoms]; NumPy tries to build a 2D array (Atoms defines "
            "__len__/__getitem__) and raises "
            "'ValueError: a must be 1-dimensional' -- see class docstring."
        ),
    )
    def test_random_subsample_within_oversized_bin(self):
        # bin 0 (avg force 0.0) holds 3 configs against a quota of 1 ->
        # triggers the np.random.choice(bin_configs, ...) call.
        configs = [
            _make_atoms_with_forces([[0.0, 0, 0]], index=i, position_offset=i)
            for i in range(3)
        ] + [
            _make_atoms_with_forces([[5.0, 0, 0]], index=3, position_offset=3),
            _make_atoms_with_forces(
                [[10.0, 0, 0]], index=4, position_offset=4
            ),
        ]
        selector = UniformForceSelector(n_configs=2, n_bins=2)
        result = selector.select(configs)
        assert len(result) == 2

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "force_n_configs=True's fill-in path also calls "
            "np.random.choice(remaining_configs, ...) directly on a "
            "list[Atoms], hitting the same NumPy crash as "
            "test_random_subsample_within_oversized_bin -- see class "
            "docstring."
        ),
    )
    def test_force_n_configs_fills_remaining_from_pool(self):
        # avg forces 0/1/2 across 3 bins -> selected = [idx0, idx1] (idx2
        # dropped per the digitize bug); force_n_configs=True should then
        # top up to n_configs=3 from the remaining pool (which still
        # contains idx2).
        configs = [
            _make_atoms_with_forces(
                [[float(i), 0, 0]], index=i, position_offset=i
            )
            for i in range(3)
        ]
        selector = UniformForceSelector(
            n_configs=3, n_bins=3, force_n_configs=True
        )
        result = selector.select(configs)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# QUESTSSelector
# ---------------------------------------------------------------------------
def _make_varied_configs(n, rng_seed=42):
    """Create structurally distinct configs for QUESTS descriptor tests.

    Generates bulk-like structures with random perturbations so that
    QUESTS descriptors differ across configs.
    """
    rng = np.random.default_rng(rng_seed)
    configs = []
    for i in range(n):
        # 2x2x2 FCC-like arrangement with 32 atoms
        base_positions = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    base_positions.append([x * 2.5, y * 2.5, z * 2.5])
                    base_positions.append(
                        [x * 2.5 + 1.25, y * 2.5 + 1.25, z * 2.5]
                    )
                    base_positions.append(
                        [x * 2.5 + 1.25, y * 2.5, z * 2.5 + 1.25]
                    )
                    base_positions.append(
                        [x * 2.5, y * 2.5 + 1.25, z * 2.5 + 1.25]
                    )
        positions = np.array(base_positions)
        # Increasing perturbation amplitude → more "disordered" structures
        noise_scale = 0.05 + i * 0.05
        positions += rng.normal(0, noise_scale, positions.shape)
        n_atoms = len(positions)
        atoms = Atoms(
            symbols=["Si"] * n_atoms,
            positions=positions,
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        )
        atoms.info["index"] = i
        configs.append(atoms)
    return configs


@pytest.mark.skipif(
    not _HAS_QUESTS, reason="needs the optional 'quests' extra"
)
class TestQUESTSSelector:
    def test_diversity_selects_correct_count(self):
        configs = _make_varied_configs(20)
        selector = QUESTSSelector(n_configs=5, method="msc")
        result = selector.select(configs)
        assert len(result) == 5

    def test_diversity_n_configs_larger_than_pool(self):
        configs = _make_varied_configs(3)
        selector = QUESTSSelector(n_configs=10, method="msc")
        result = selector.select(configs)
        assert len(result) == 3

    def test_diversity_marks_mode_in_info(self):
        configs = _make_varied_configs(10)
        selector = QUESTSSelector(n_configs=3, method="msc")
        result = selector.select(configs)
        for at in result:
            assert at.info["quests_mode"] == "diversity"

    def test_novelty_mode(self):
        """Novelty mode: select configs most different from reference."""
        # Reference: low-noise configs
        reference = _make_varied_configs(5, rng_seed=0)
        # Candidates: mix of similar + very different
        candidates = _make_varied_configs(10, rng_seed=100)
        selector = QUESTSSelector(n_configs=3, reference=reference)
        result = selector.select(candidates)
        assert len(result) == 3
        for at in result:
            assert "quests_novelty_score" in at.info
            assert at.info["quests_mode"] == "novelty"

    def test_novelty_scores_ordered(self):
        """Selected configs should be ordered by decreasing novelty."""
        reference = _make_varied_configs(5, rng_seed=0)
        candidates = _make_varied_configs(10, rng_seed=100)
        selector = QUESTSSelector(n_configs=5, reference=reference)
        result = selector.select(candidates)
        scores = [at.info["quests_novelty_score"] for at in result]
        assert scores == sorted(scores, reverse=True)

    def test_reference_arg_enables_novelty(self):
        """A per-call reference= switches select() to novelty mode.

        The reference is now an argument (the ``set_reference`` mutator was
        removed), so passing it per call -- without mutating the selector --
        flips diversity to novelty mode.
        """
        configs = _make_varied_configs(10, rng_seed=42)
        selector = QUESTSSelector(n_configs=3)

        # No reference → diversity mode
        result1 = selector.select(configs)
        assert result1[0].info["quests_mode"] == "diversity"

        # Reference passed as a call argument → novelty mode
        ref = _make_varied_configs(5, rng_seed=0)
        result2 = selector.select(configs, reference=ref)
        assert result2[0].info["quests_mode"] == "novelty"

        # It was not stored on the selector: the next call is diversity again.
        result3 = selector.select(configs)
        assert result3[0].info["quests_mode"] == "diversity"

    def test_empty_input(self):
        selector = QUESTSSelector(n_configs=5, method="msc")
        assert selector.select([]) == []

    def test_grouped_input_flattens(self):
        """Grouped input should be flattened before QUESTS selection."""
        configs = _make_varied_configs(10)
        grouped = [configs[:5], configs[5:]]
        selector = QUESTSSelector(n_configs=3, method="msc")
        result = selector.select(grouped)
        assert len(result) == 3

    def test_custom_descriptor_params(self):
        """Custom k and cutoff should work without error."""
        configs = _make_varied_configs(10)
        selector = QUESTSSelector(n_configs=3, method="fps", k=16, cutoff=4.0)
        result = selector.select(configs)
        assert len(result) == 3

    def test_explicit_bandwidth(self):
        configs = _make_varied_configs(10)
        selector = QUESTSSelector(n_configs=3, method="msc", bandwidth=0.02)
        result = selector.select(configs)
        assert len(result) == 3

    def test_pca_output_diversity(self, tmp_path):
        """PCA JSON is written with correct structure in diversity mode."""
        configs = _make_varied_configs(10)
        out = tmp_path / "pca.json"
        selector = QUESTSSelector(n_configs=3, method="msc", pca_output=out)
        result = selector.select(configs)

        assert out.exists()
        data = json.loads(out.read_text())

        assert data["mode"] == "diversity"
        assert data["method"] == "msc"
        assert data["n_total"] == 10
        assert data["n_selected"] == 3
        assert len(data["structures"]) == 10
        assert len(data["explained_variance_ratio"]) == 2
        assert sum(1 for s in data["structures"] if s["selected"]) == 3

        for s in data["structures"]:
            assert "pc1" in s
            assert "pc2" in s
            assert "index" in s
            assert "formula" in s

    def test_pca_output_novelty(self, tmp_path):
        """PCA JSON includes novelty scores in novelty mode."""
        reference = _make_varied_configs(5, rng_seed=0)
        candidates = _make_varied_configs(10, rng_seed=100)
        out = tmp_path / "pca_novelty.json"
        selector = QUESTSSelector(
            n_configs=3, reference=reference, pca_output=out
        )
        selector.select(candidates)

        data = json.loads(out.read_text())
        assert data["mode"] == "novelty"

        selected = [s for s in data["structures"] if s["selected"]]
        assert len(selected) == 3
        for s in selected:
            assert "novelty_score" in s

    def test_pca_not_written_when_none(self, tmp_path):
        """No JSON file when pca_output is None."""
        configs = _make_varied_configs(10)
        selector = QUESTSSelector(n_configs=3, method="msc")
        selector.select(configs)
        assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# PipelineSelector
# ---------------------------------------------------------------------------
class TestPipelineSelector:
    def test_uncertainty_then_uniform(self):
        """Two-stage: filter by uncertainty, then subsample uniformly."""
        energy_uncerts = [0.1 * i for i in range(20)]
        configs = _make_config_pool(20, energy_uncerts=energy_uncerts)

        pipeline = PipelineSelector(
            [
                UncertaintySelector(
                    n_configs=10,
                    energy_key="energy_uncertainty",
                    forces_key=None,
                ),
                StrideSelector(n_configs=3),
            ]
        )
        result = pipeline.select(configs)
        assert len(result) == 3
        # Uncertainty metadata should survive through the pipeline
        for at in result:
            assert "uncertainty_score" in at.info

    def test_uncertainty_then_quests(self):
        """Uncertainty filter → QUESTS diversity selection."""
        pytest.importorskip("quests")
        # Create varied configs with uncertainty data
        configs = _make_varied_configs(20)
        for i, at in enumerate(configs):
            at.info["energy_uncertainty"] = 0.1 * i

        pipeline = PipelineSelector(
            [
                UncertaintySelector(
                    n_configs=10,
                    energy_key="energy_uncertainty",
                    forces_key=None,
                ),
                QUESTSSelector(n_configs=3, method="msc"),
            ]
        )
        result = pipeline.select(configs)
        assert len(result) == 3
        for at in result:
            assert "uncertainty_score" in at.info
            assert at.info["quests_mode"] == "diversity"

    def test_forwards_reference_to_members(self):
        """A reference passed to the pipeline reaches a QUESTS member.

        This is the fix's headline: composing "stride then QUESTS-novelty"
        must run QUESTS in *novelty* mode. The pipeline forwards ``reference``
        to every member, while a non-novelty member (StrideSelector) ignores
        it -- so no per-member ``set_reference`` plumbing is needed.
        """
        pytest.importorskip("quests")
        configs = _make_varied_configs(20, rng_seed=42)
        reference = _make_varied_configs(5, rng_seed=0)

        pipeline = PipelineSelector(
            [StrideSelector(n_configs=10), QUESTSSelector(n_configs=3)]
        )

        # No reference -> the QUESTS member runs in diversity mode.
        diversity = pipeline.select(configs)
        assert all(at.info["quests_mode"] == "diversity" for at in diversity)

        # Reference forwarded -> the QUESTS member runs in novelty mode.
        novelty = pipeline.select(configs, reference=reference)
        assert all(at.info["quests_mode"] == "novelty" for at in novelty)
        assert all("quests_novelty_score" in at.info for at in novelty)

    def test_single_selector(self):
        """Pipeline with one selector behaves like the selector alone."""
        configs = _make_config_pool(10)
        selector = StrideSelector(n_configs=3)
        pipeline = PipelineSelector([selector])

        direct = selector.select(configs)
        piped = pipeline.select(configs)
        assert len(piped) == len(direct)
        assert [at.info["index"] for at in piped] == [
            at.info["index"] for at in direct
        ]

    def test_empty_intermediate(self):
        """If first stage returns nothing, pipeline returns empty."""
        energy_uncerts = [0.1, 0.2, 0.3]
        configs = _make_config_pool(3, energy_uncerts=energy_uncerts)

        pipeline = PipelineSelector(
            [
                UncertaintySelector(
                    threshold=999.0,
                    energy_key="energy_uncertainty",
                    forces_key=None,
                ),
                StrideSelector(n_configs=5),
            ]
        )
        result = pipeline.select(configs)
        assert result == []

    def test_empty_input(self):
        pipeline = PipelineSelector([StrideSelector(n_configs=5)])
        assert pipeline.select([]) == []

    def test_grouped_input_first_stage(self):
        """Grouped input is handled by first selector, rest get flat."""
        grouped = _make_grouped_configs([3, 5, 2])  # 10 configs total

        pipeline = PipelineSelector(
            [
                LastFrameSelector(),  # picks last from each group → 3 configs
                StrideSelector(n_configs=2),  # picks 2 from those 3
            ]
        )
        result = pipeline.select(grouped)
        assert len(result) == 2

    def test_empty_selectors_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            PipelineSelector([])

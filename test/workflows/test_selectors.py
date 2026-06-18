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
    LastFrameSelector,
    PipelineSelector,
    QUESTSSelector,
    UncertaintySelector,
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
# UniformSelector
# ---------------------------------------------------------------------------
class TestUniformSelector:
    def test_selects_correct_count(self):
        configs = _make_config_pool(100)
        selector = UniformSelector(n_configs=10)
        result = selector.select(configs)
        assert len(result) == 10

    def test_selects_first_and_last(self):
        configs = _make_config_pool(50)
        selector = UniformSelector(n_configs=5)
        result = selector.select(configs)
        assert result[0].info["index"] == 0
        assert result[-1].info["index"] == 49

    def test_evenly_spaced(self):
        configs = _make_config_pool(100)
        selector = UniformSelector(n_configs=5)
        result = selector.select(configs)
        indices = [at.info["index"] for at in result]
        assert indices == [0, 24, 49, 74, 99]

    def test_n_configs_larger_than_pool(self):
        configs = _make_config_pool(3)
        selector = UniformSelector(n_configs=10)
        result = selector.select(configs)
        assert len(result) == 3

    def test_single_config(self):
        configs = _make_config_pool(1)
        selector = UniformSelector(n_configs=5)
        result = selector.select(configs)
        assert len(result) == 1

    def test_empty_input(self):
        selector = UniformSelector(n_configs=10)
        assert selector.select([]) == []

    def test_grouped_input_flattens(self):
        """Grouped input should be flattened before uniform selection."""
        grouped = _make_grouped_configs([5, 5])  # 10 total
        selector = UniformSelector(n_configs=3)
        result = selector.select(grouped)
        assert len(result) == 3
        # Should span across both groups
        indices = [at.info["index"] for at in result]
        assert indices[0] == 0
        assert indices[-1] == 9


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

    def test_set_reference(self):
        """set_reference updates the reference dataset."""
        configs = _make_varied_configs(10, rng_seed=42)
        selector = QUESTSSelector(n_configs=3)

        # Initially no reference → diversity mode
        result1 = selector.select(configs)
        assert result1[0].info["quests_mode"] == "diversity"

        # Set reference → novelty mode
        ref = _make_varied_configs(5, rng_seed=0)
        selector.set_reference(ref)
        result2 = selector.select(configs)
        assert result2[0].info["quests_mode"] == "novelty"

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
                UniformSelector(n_configs=3),
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

    def test_single_selector(self):
        """Pipeline with one selector behaves like the selector alone."""
        configs = _make_config_pool(10)
        selector = UniformSelector(n_configs=3)
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
                UniformSelector(n_configs=5),
            ]
        )
        result = pipeline.select(configs)
        assert result == []

    def test_empty_input(self):
        pipeline = PipelineSelector([UniformSelector(n_configs=5)])
        assert pipeline.select([]) == []

    def test_grouped_input_first_stage(self):
        """Grouped input is handled by first selector, rest get flat."""
        grouped = _make_grouped_configs([3, 5, 2])  # 10 configs total

        pipeline = PipelineSelector(
            [
                LastFrameSelector(),  # picks last from each group → 3 configs
                UniformSelector(n_configs=2),  # picks 2 from those 3
            ]
        )
        result = pipeline.select(grouped)
        assert len(result) == 2

    def test_empty_selectors_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            PipelineSelector([])

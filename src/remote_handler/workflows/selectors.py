"""Configuration selectors for active learning sampling workflows.

Selectors determine which configurations from a pool of candidates
are passed forward to the ground-truth evaluation step.

The ``select`` method accepts either:
  - ``list[Atoms]`` — a flat list of configurations
  - ``list[list[Atoms]]`` — grouped configurations (e.g., one inner
    list per trajectory/replica)

Selectors that care about grouping (e.g., ``LastFrameSelector``) use
the nested structure. Selectors that don't (e.g., ``StrideSelector``)
flatten it automatically.

Example:
    >>> from remote_handler.workflows.selectors import StrideSelector
    >>> selector = StrideSelector(n_configs=20)
    >>> sampler = TorchSimSampler(..., selector=selector)

    >>> from remote_handler.workflows.selectors import QUESTSSelector
    >>> selector = QUESTSSelector(n_configs=20, method="msc")
    >>> # With novelty relative to existing training data:
    >>> selector = QUESTSSelector(n_configs=20, reference=train_atoms)
"""

import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

# Input type: flat list or grouped (one inner list per trajectory)
Configs = Union[list[Atoms], list[list[Atoms]]]


def _flatten(configs: Configs) -> list[Atoms]:
    """Flatten grouped configs into a single list."""
    if not configs:
        return []
    if isinstance(configs[0], list):
        return [at for group in configs for at in group]
    return configs


def _is_grouped(configs: Configs) -> bool:
    """Check whether configs are grouped (list[list[Atoms]])."""
    return bool(configs) and isinstance(configs[0], list)


class ConfigSelector(ABC):
    """Abstract base class for configuration selection strategies.

    A ConfigSelector receives configurations (flat or grouped) and
    returns a flat list of selected Atoms.
    """

    @abstractmethod
    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        """Select configurations from a pool of candidates.

        Args:
            configs: Pool of candidate configurations, either as a flat
                ``list[Atoms]`` or grouped ``list[list[Atoms]]`` (e.g.,
                one inner list per trajectory).
            reference: Optional auxiliary set this selection scores against
                (e.g. the already-labelled data, for novelty). Selectors that
                don't use a reference ignore it. Passed as an argument -- not
                set via a mutator -- so it composes cleanly (PipelineSelector
                forwards it to its members).

        Returns:
            Flat list of selected configurations.
        """
        pass


class LastFrameSelector(ConfigSelector):
    """Select the last frame from each group.

    When given grouped input (``list[list[Atoms]]``), returns the last
    element of each inner list — e.g., the final frame from each
    trajectory/replica.

    When given a flat list, returns it unchanged (assumes each element
    is already a "last frame").
    """

    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        if not configs:
            return []

        if _is_grouped(configs):
            selected = [group[-1] for group in configs if group]
            logger.info(
                f"LastFrameSelector: selected {len(selected)} last frames "
                f"from {len(configs)} groups"
            )
            return selected

        logger.info(
            f"LastFrameSelector: returning all {len(configs)} configs "
            f"(flat input)"
        )
        return configs


class StrideSelector(ConfigSelector):
    """Select configurations at evenly spaced (strided) intervals.

    Picks ``n_configs`` evenly spaced configurations from the (flattened)
    input -- i.e. every *stride*-th frame, with the stride chosen so the
    selection spans the whole pool. This is deterministic, *not* random
    subsampling; useful for thinning long trajectories without bias.

    Args:
        n_configs: Number of configurations to select.
    """

    def __init__(self, n_configs: int):
        self.n_configs = n_configs

    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        flat = _flatten(configs)
        n_total = len(flat)
        if n_total == 0:
            logger.warning("StrideSelector: received empty config list")
            return []

        n_select = min(self.n_configs, n_total)
        indices = np.linspace(0, n_total - 1, n_select, dtype=int)
        # Remove duplicates while preserving order
        indices = list(dict.fromkeys(indices))

        selected = [flat[i] for i in indices]
        logger.info(
            f"StrideSelector: selected {len(selected)} from "
            f"{n_total} configs"
        )
        return selected


class UniformSelector(StrideSelector):
    """Deprecated alias for :class:`StrideSelector`.

    The old name read as *uniform-random* sampling, but the behaviour is
    deterministic evenly-spaced (strided) selection. Kept for backward
    compatibility; instantiating it emits a :class:`DeprecationWarning`.
    """

    def __init__(self, n_configs: int):
        warnings.warn(
            "UniformSelector is deprecated and will be removed in a future "
            "release; use StrideSelector instead (same behaviour, clearer "
            "name -- it selects evenly-spaced frames, not a random sample).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(n_configs)


class UncertaintySelector(ConfigSelector):
    """Select configurations with highest uncertainty.

    Reads uncertainty values that are expected to already be stored
    in the Atoms objects (e.g., computed by a committee model during
    the simulation). Selects the ``n_configs`` frames with highest
    combined uncertainty score.

    The score is computed as::

        score = energy_weight * normalized_energy_uncertainty
              + forces_weight * normalized_forces_uncertainty

    where normalization maps each component to [0, 1] across all
    candidates.

    Args:
        n_configs: Maximum number of configurations to select. Can be
            None when ``threshold`` is set (pure threshold mode).
        energy_key: Key in ``atoms.info`` for per-structure energy
            uncertainty. Set to None to ignore energy uncertainty.
        forces_key: Key in ``atoms.arrays`` for per-atom force
            uncertainty (shape ``(n_atoms, 3)``). Set to None to
            ignore force uncertainty.
        energy_weight: Weight for energy uncertainty in combined score.
        forces_weight: Weight for force uncertainty in combined score.
        threshold: Minimum uncertainty score for selection. When set,
            only configs with score above this value are selected.
            Can be combined with ``n_configs`` to cap the count.
    """

    def __init__(
        self,
        n_configs: int | None = None,
        energy_key: str | None = "energy_uncertainty",
        forces_key: str | None = "forces_uncertainty",
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        threshold: float | None = None,
    ):
        if energy_key is None and forces_key is None:
            raise ValueError(
                "At least one of energy_key or forces_key must be provided"
            )
        if n_configs is None and threshold is None:
            raise ValueError(
                "At least one of n_configs or threshold must be provided"
            )
        self.n_configs = n_configs
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.threshold = threshold

    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        flat = _flatten(configs)
        n_total = len(flat)
        if n_total == 0:
            logger.warning("UncertaintySelector: received empty config list")
            return []

        scores = np.zeros(n_total)

        if self.energy_key is not None:
            energy_uncerts = []
            for at in flat:
                if self.energy_key not in at.info:
                    raise KeyError(
                        f"Energy uncertainty key '{self.energy_key}' not "
                        f"found in atoms.info. Available keys: "
                        f"{list(at.info.keys())}. "
                        f"Ensure the simulation records uncertainty data."
                    )
                energy_uncerts.append(float(at.info[self.energy_key]))

            energy_uncerts = np.array(energy_uncerts)
            e_range = energy_uncerts.max() - energy_uncerts.min()
            if e_range > 0:
                energy_norm = (energy_uncerts - energy_uncerts.min()) / e_range
            else:
                energy_norm = np.zeros_like(energy_uncerts)
            scores += self.energy_weight * energy_norm

        if self.forces_key is not None:
            forces_uncerts = []
            for at in flat:
                if self.forces_key not in at.arrays:
                    raise KeyError(
                        f"Forces uncertainty key '{self.forces_key}' not "
                        f"found in atoms.arrays. Available keys: "
                        f"{list(at.arrays.keys())}. "
                        f"Ensure the simulation records uncertainty data."
                    )
                # Aggregate per-atom force uncertainty: mean of L2 norms
                f_uncert = at.arrays[self.forces_key]
                forces_uncerts.append(np.linalg.norm(f_uncert, axis=-1).mean())

            forces_uncerts = np.array(forces_uncerts)
            f_range = forces_uncerts.max() - forces_uncerts.min()
            if f_range > 0:
                forces_norm = (forces_uncerts - forces_uncerts.min()) / f_range
            else:
                forces_norm = np.zeros_like(forces_uncerts)
            scores += self.forces_weight * forces_norm

        # Rank by score (descending)
        ranked_indices = np.argsort(scores)[::-1]

        # Apply threshold filter if set
        if self.threshold is not None:
            ranked_indices = ranked_indices[
                scores[ranked_indices] >= self.threshold
            ]

        # Apply n_configs cap if set
        if self.n_configs is not None:
            n_select = min(self.n_configs, len(ranked_indices))
            ranked_indices = ranked_indices[:n_select]

        selected = [flat[i] for i in ranked_indices]
        for at, idx in zip(selected, ranked_indices):
            at.info["uncertainty_score"] = float(scores[idx])

        logger.info(
            f"UncertaintySelector: selected {len(selected)} from "
            f"{n_total} configs "
            f"(score range: {scores.min():.4f} - {scores.max():.4f})"
        )
        return selected

class ForceFilterSelector(ConfigSelector):
    """Select configurations based on force magnitude threshold.

    Filters out configurations where the maximum atomic force exceeds
    a specified threshold. Useful for removing high-energy or unstable
    structures from the candidate pool.

    Args:
        force_threshold: Maximum allowed force magnitude (in eV/Å).
    """

    def __init__(self, force_threshold: float):
        self.force_threshold = force_threshold

    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        flat = _flatten(configs)
        n_total = len(flat)
        if n_total == 0:
            logger.warning("ForceFilterSelector: received empty config list")
            return []

        selected = []
        for at in flat:
            max_force = np.max(np.linalg.norm(at.arrays["forces"], axis=-1))
            if max_force <= self.force_threshold:
                selected.append(at)

        logger.info(
            f"ForceFilterSelector: selected {len(selected)} from "
            f"{n_total} configs (force threshold: {self.force_threshold:.4f} eV/Å)"
        )
        return selected
    
class UniformForceSelector(ConfigSelector):
    """Select configurations uniformly across force magnitude bins.

    Divides the range of average force magnitudes into bins and selects an equal number
    of configurations from each bin. This ensures a diverse representation of
    configurations across different force regimes.

    Args:
        n_configs: Total number of configurations to select.
        n_bins: Number of bins to divide the force magnitude range into.
    """

    def __init__(self, n_configs: int, n_bins: int = 10, force_n_configs: bool = False):
        self.n_configs = n_configs
        self.n_bins = n_bins
        self.force_n_configs = force_n_configs

    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        flat = _flatten(configs)
        n_total = len(flat)
        if n_total == 0:
            logger.warning("UniformForceSelector: received empty config list")
            return []

        # Compute average force magnitudes for each configuration
        avg_forces = np.array([
            np.mean(np.linalg.norm(at.arrays["forces"], axis=-1)) for at in flat
        ])

        # Create bins and assign configurations to bins
        bins = np.linspace(avg_forces.min(), avg_forces.max(), self.n_bins + 1)
        bin_indices = np.digitize(avg_forces, bins) - 1  # Bin indices start from 0

        n_empty_bins = np.sum([np.sum(bin_indices == i) == 0 for i in range(self.n_bins)])
        selected = []
        configs_per_bin = self.n_configs // (self.n_bins - n_empty_bins) if self.n_bins - n_empty_bins > 0 else 0
        
        if configs_per_bin == 0:
            logger.warning(
                "UniformForceSelector: All bins are empty"
            )
            return []

        for bin_idx in range(self.n_bins):
            bin_mask = bin_indices == bin_idx
            bin_configs = [flat[i] for i in range(n_total) if bin_mask[i]]

            if len(bin_configs) == 0:
                continue  # Skip empty bins
            
            if len(bin_configs) > configs_per_bin:
                # Randomly sample from the bin if there are more configs than needed
                selected.extend(np.random.choice(bin_configs, configs_per_bin, replace=False))
            else:
                selected.extend(bin_configs)
        
        if len(selected) < self.n_configs and not self.force_n_configs:
            logger.warning(
                f"UniformForceSelector: Only {len(selected)} configurations selected, "
                f"less than requested {self.n_configs} due to almost empty bins."
                f"set force_n_configs=True to force selection of n_configs."
            )
        elif len(selected) < self.n_configs and self.force_n_configs:
            # If force_n_configs is True, fill the remaining slots with random selections
            remaining = self.n_configs - len(selected)
            remaining_configs = [at for at in flat if at not in selected]
            if len(remaining_configs) > 0:
                selected.extend(np.random.choice(remaining_configs, min(remaining, len(remaining_configs)), replace=False))
            logger.info(
                f"UniformForceSelector: randomly selected {min(remaining, len(remaining_configs))} additional configurations to meet n_configs."
            )

        logger.info(
            f"UniformForceSelector: Selected {len(selected)} configurations from {self.n_bins} bins."
        )
        return selected

class QUESTSSelector(ConfigSelector):
    """Select configurations using QUESTS information entropy.

    Uses model-free structural descriptors and information theory to
    select the most informative configurations. Two modes:

    **Diversity mode** (no reference): Uses ``DatasetCompressor`` to
    select a structurally diverse subset from the candidates. The
    ``method`` parameter controls the selection strategy (MSC, FPS,
    etc.).

    **Novelty mode** (with reference): Uses ``delta_entropy`` to score
    each candidate by how novel its atomic environments are relative
    to a reference dataset (e.g., existing training data). Selects the
    ``n_configs`` candidates with highest mean novelty score.

    Requires the ``quests`` package (``pip install quests``).

    Args:
        n_configs: Number of configurations to select.
        method: Selection method for diversity mode. One of ``"msc"``
            (Maximum Structure Coverage, recommended), ``"fps"``
            (Farthest Point Sampling), ``"random"``, ``"k_means"``.
            Ignored in novelty mode.
        reference: Default reference dataset for novelty mode. Usually
            supplied per-call via ``select(configs, reference=...)`` instead
            (which takes precedence), so a stage can pass the accumulated
            labelled data each iteration without mutating the selector.
        k: Number of nearest neighbors for QUESTS descriptor.
        cutoff: Cutoff radius in Angstroms for descriptor.
        bandwidth: Kernel bandwidth for entropy estimation. If None,
            estimated automatically from the volume per atom.
        pca_output: If set, save a JSON file with PCA-projected
            descriptor data for visualization. Contains per-structure
            2D coordinates, selection status, and explained variance.
    """

    def __init__(
        self,
        n_configs: int,
        method: str = "msc",
        reference: list[Atoms] | None = None,
        k: int = 32,
        cutoff: float = 5.0,
        bandwidth: float | None = None,
        pca_output: Path | str | None = None,
    ):
        self.n_configs = n_configs
        self.method = method
        self.reference = reference
        self.k = k
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.pca_output = Path(pca_output) if pca_output is not None else None

    def _get_bandwidth(self, atoms_list: list[Atoms]) -> float:
        """Estimate bandwidth from volume per atom if not set explicitly."""
        if self.bandwidth is not None:
            return self.bandwidth

        from quests.entropy import get_bandwidth

        volumes = [
            at.get_volume() / len(at) for at in atoms_list if at.pbc.all()
        ]
        if volumes:
            avg_vol = np.mean(volumes)
            return get_bandwidth(avg_vol)

        return 0.015  # default fallback

    def _compute_descriptors(self, atoms_list: list[Atoms]) -> np.ndarray:
        """Compute QUESTS descriptors for a list of Atoms."""
        from quests.descriptor import get_descriptors

        return get_descriptors(atoms_list, k=self.k, cutoff=self.cutoff)

    def _atom_counts(self, atoms_list: list[Atoms]) -> np.ndarray:
        """Return cumulative atom counts for mapping atoms → structures."""
        counts = np.array([len(at) for at in atoms_list])
        return np.cumsum(counts)

    def _save_pca_json(self, flat: list[Atoms], selected: list[Atoms]) -> None:
        """Compute PCA on per-structure mean descriptors and save to JSON."""
        from sklearn.decomposition import PCA

        atom_desc = self._compute_descriptors(flat)
        cum_counts = self._atom_counts(flat)
        starts = np.concatenate([[0], cum_counts[:-1]])
        struct_desc = np.array(
            [atom_desc[s:e].mean(axis=0) for s, e in zip(starts, cum_counts)]
        )

        pca = PCA(
            n_components=min(2, struct_desc.shape[0], struct_desc.shape[1])
        )
        proj = pca.fit_transform(struct_desc)

        selected_ids = set(id(at) for at in selected)
        mode = "novelty" if self.reference else "diversity"

        structures = []
        for i, at in enumerate(flat):
            is_selected = id(at) in selected_ids
            entry = {
                "index": i,
                "pc1": float(proj[i, 0]),
                "pc2": float(proj[i, 1]) if proj.shape[1] > 1 else 0.0,
                "selected": is_selected,
                "n_atoms": len(at),
                "formula": at.get_chemical_formula(),
            }
            if is_selected and "quests_novelty_score" in at.info:
                entry["novelty_score"] = at.info["quests_novelty_score"]
            structures.append(entry)

        data = {
            "mode": mode,
            "method": self.method,
            "n_total": len(flat),
            "n_selected": len(selected),
            "k": self.k,
            "cutoff": self.cutoff,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "structures": structures,
        }

        self.pca_output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pca_output, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"QUESTSSelector: saved PCA data to {self.pca_output}")

    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        flat = _flatten(configs)
        n_total = len(flat)
        if n_total == 0:
            logger.warning("QUESTSSelector: received empty config list")
            return []

        n_select = min(self.n_configs, n_total)
        h = self._get_bandwidth(flat)

        # Per-call reference (e.g. accumulated labelled data) wins; the
        # constructor default is the fallback. No mutable state involved.
        ref = reference if reference is not None else self.reference
        if ref is not None:
            selected = self._select_by_novelty(flat, n_select, h, ref)
        else:
            selected = self._select_by_diversity(flat, n_select, h)

        if self.pca_output is not None:
            self._save_pca_json(flat, selected)

        logger.info(
            f"QUESTSSelector: selected {len(selected)} from "
            f"{n_total} configs "
            f"(mode={'novelty' if ref is not None else 'diversity'}, "
            f"method={self.method}, bandwidth={h:.4f})"
        )
        return selected

    def _select_by_diversity(
        self, flat: list[Atoms], n_select: int, h: float
    ) -> list[Atoms]:
        """Select diverse configs using DatasetCompressor."""
        from quests.compression import DatasetCompressor

        compressor = DatasetCompressor(flat, bandwidth=h, batch_size=20000)
        indices = compressor.get_indices(method=self.method, size=n_select)

        selected = [flat[i] for i in indices]
        summary = compressor.get_summary(indices)
        for at in selected:
            at.info["quests_mode"] = "diversity"
        logger.debug(
            f"QUESTSSelector diversity summary: entropy={summary['entropy']:.4f}, "
            f"diversity={summary['diversity']:.4f}, "
            f"overlap={summary['overlap']:.4f}"
        )
        return selected

    def _select_by_novelty(
        self,
        flat: list[Atoms],
        n_select: int,
        h: float,
        reference: list[Atoms],
    ) -> list[Atoms]:
        """Select configs most novel relative to reference dataset."""
        from quests.entropy import delta_entropy

        ref_desc = self._compute_descriptors(reference)
        cand_desc = self._compute_descriptors(flat)

        # Per-atom novelty scores
        dH = delta_entropy(cand_desc, ref_desc, h=h)

        # Aggregate per-atom dH to per-structure scores (mean)
        cum_counts = self._atom_counts(flat)
        starts = np.concatenate([[0], cum_counts[:-1]])
        struct_scores = np.array(
            [dH[s:e].mean() for s, e in zip(starts, cum_counts)]
        )

        # Select top-N by novelty
        top_indices = np.argsort(struct_scores)[::-1][:n_select]
        selected = [flat[i] for i in top_indices]
        for at, idx in zip(selected, top_indices):
            at.info["quests_novelty_score"] = float(struct_scores[idx])
            at.info["quests_mode"] = "novelty"

        return selected


class PipelineSelector(ConfigSelector):
    """Chain multiple selectors sequentially.

    Each selector's output becomes the next selector's input.
    Grouped input is only preserved for the first stage — subsequent
    stages receive flat lists (since selectors return ``list[Atoms]``).

    Example::

        >>> pipeline = PipelineSelector([
        ...     UncertaintySelector(n_configs=100),
        ...     QUESTSSelector(n_configs=20, method="msc"),
        ... ])
        >>> selected = pipeline.select(configs)  # 20 diverse high-uncertainty configs

    Args:
        selectors: Ordered list of selectors to apply.
    """

    def __init__(self, selectors: list[ConfigSelector]):
        if not selectors:
            raise ValueError("PipelineSelector requires at least one selector")
        self.selectors = selectors

    def select(
        self, configs: Configs, reference: list[Atoms] | None = None
    ) -> list[Atoms]:
        result = configs
        for selector in self.selectors:
            # Forward the reference to every member; non-novelty selectors
            # (e.g. StrideSelector) simply ignore it. This is what makes a
            # composed "stride then QUESTS-novelty" pipeline work -- no
            # per-member set_reference plumbing.
            result = selector.select(result, reference=reference)
            if not result:
                logger.warning(
                    f"PipelineSelector: selector {selector.__class__.__name__} "
                    f"returned empty selection; stopping pipeline"
                )
                break
        return result

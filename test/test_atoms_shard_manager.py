"""
Tests for AtomsShardManager - demonstrates clean testing of domain logic
without SLURM infrastructure dependencies.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from remote_handler.core.atoms_shard import (
    AtomsShardManager,
    front_tag,
    split_list,
)


class TestHelperFunctions:
    """Test standalone helper functions."""

    def test_split_list_even_division(self):
        """Test split_list with evenly divisible list."""
        lst = list(range(10))
        result = split_list(lst, shard_size=2)
        assert len(result) == 5
        assert result[0] == [0, 1]
        assert result[-1] == [8, 9]

    def test_split_list_uneven_division(self):
        """Test split_list with remainder."""
        lst = list(range(10))
        result = split_list(lst, shard_size=3)
        assert len(result) == 4
        assert result[0] == [0, 1, 2]
        assert result[-1] == [9]  # Remainder

    def test_front_tag(self):
        """Test front_tag prepends correctly."""
        result = front_tag("data/trajectory.extxyz", "opt")
        assert result == Path("data/opt_trajectory.extxyz")

    def test_front_tag_empty(self):
        """Test front_tag with empty tag."""
        result = front_tag("data/trajectory.extxyz", "")
        assert result == Path("data/trajectory.extxyz")


class TestAtomsShardManagerInit:
    """Test AtomsShardManager initialization."""

    def test_init_stores_parameters(self):
        """Verify initialization stores all parameters."""
        manager = AtomsShardManager(
            traj_files=["file1.extxyz", "file2.extxyz"],
            shard_size=10,
            indices=slice(0, 100, 2),
        )

        assert manager.traj_files == ["file1.extxyz", "file2.extxyz"]
        assert manager.shard_size == 10
        assert manager.indices == slice(0, 100, 2)
        assert manager.shards is None
        assert manager.num_configs_total is None


class TestAtomsShardManagerCombineTrajs:
    """Test trajectory combining and metadata labeling."""

    def test_combine_trajs_single_file(self):
        """Test combining single trajectory file."""
        mock_fs = MagicMock()

        # Create mock atoms
        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 1], [1, 0, 1], [0, 1, 1]])
        mock_fs.read_ase.return_value = [atoms1, atoms2]

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"],
            shard_size=5,
            indices=slice(None, None, 1),
        )

        result = manager.combine_trajs(
            ["traj.extxyz"], slice(None, None, 1), mock_fs
        )

        assert len(result) == 2
        # Check metadata
        assert result[0].info["file_index"] == 0
        assert result[0].info["iteration_index"] == 0
        assert result[1].info["file_index"] == 0
        assert result[1].info["iteration_index"] == 1

    def test_combine_trajs_multiple_files(self):
        """Test combining multiple trajectory files."""
        mock_fs = MagicMock()

        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 1], [1, 0, 1], [0, 1, 1]])

        # Mock read_ase to return different atoms for each file
        mock_fs.read_ase.side_effect = [[atoms1], [atoms2]]

        manager = AtomsShardManager(
            traj_files=["traj1.extxyz", "traj2.extxyz"],
            shard_size=5,
            indices=slice(None, None, 1),
        )

        result = manager.combine_trajs(
            ["traj1.extxyz", "traj2.extxyz"], slice(None, None, 1), mock_fs
        )

        assert len(result) == 2
        # First file
        assert result[0].info["file_index"] == 0
        # Second file
        assert result[1].info["file_index"] == 1

    def test_combine_trajs_with_step(self):
        """Test that iteration_index respects step parameter."""
        mock_fs = MagicMock()

        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 1], [1, 0, 1], [0, 1, 1]])
        mock_fs.read_ase.return_value = [atoms1, atoms2]

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"],
            shard_size=5,
            indices=slice(None, None, 3),
        )

        result = manager.combine_trajs(
            ["traj.extxyz"], slice(None, None, 3), mock_fs
        )

        # iteration_index should be multiplied by step
        assert result[0].info["iteration_index"] == 0  # 0 * 3
        assert result[1].info["iteration_index"] == 3  # 1 * 3


class TestAtomsShardManagerReadAndShard:
    """Test reading and sharding logic."""

    def test_read_and_shard_basic(self):
        """Test basic sharding calculation."""
        mock_fs = MagicMock()

        # Create 7 mock atoms
        atoms_list = [
            Atoms("H2O", positions=[[0, 0, i], [1, 0, i], [0, 1, i]])
            for i in range(7)
        ]
        mock_fs.read_ase.return_value = atoms_list

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"],
            shard_size=3,
            indices=slice(None, None, 1),
        )

        shards = manager.read_and_shard(mock_fs)

        assert len(shards) == 3  # ceil(7/3) = 3
        assert len(shards[0]) == 3
        assert len(shards[1]) == 3
        assert len(shards[2]) == 1
        assert manager.num_configs_total == 7
        assert manager.num_array_tasks == 3
        assert manager.all_shard_sizes == [3, 3, 1]

    def test_read_and_shard_exact_division(self):
        """Test sharding with exact division."""
        mock_fs = MagicMock()

        atoms_list = [
            Atoms("H2O", positions=[[0, 0, i], [1, 0, i], [0, 1, i]])
            for i in range(6)
        ]
        mock_fs.read_ase.return_value = atoms_list

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"],
            shard_size=2,
            indices=slice(None, None, 1),
        )

        shards = manager.read_and_shard(mock_fs)

        assert len(shards) == 3
        assert all(len(shard) == 2 for shard in shards)


class TestAtomsShardManagerRegroup:
    """Test regrouping logic."""

    def test_regroup_single_file(self):
        """Test regrouping atoms from single file."""
        atoms1 = Atoms("H2O")
        atoms1.info["file_index"] = 0
        atoms2 = Atoms("H2O")
        atoms2.info["file_index"] = 0

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"], shard_size=5, indices=slice(None)
        )

        result = manager.regroup([atoms1, atoms2])

        assert len(result) == 1
        assert len(result[0]) == 2

    def test_regroup_multiple_files(self):
        """Test regrouping atoms from multiple files."""
        # Note: regroup uses groupby which groups CONSECUTIVE items
        atoms1 = Atoms("H2O")
        atoms1.info["file_index"] = 0
        atoms2 = Atoms("H2O")
        atoms2.info["file_index"] = 0
        atoms3 = Atoms("H2O")
        atoms3.info["file_index"] = 1
        atoms4 = Atoms("H2O")
        atoms4.info["file_index"] = 1

        manager = AtomsShardManager(
            traj_files=["traj1.extxyz", "traj2.extxyz"],
            shard_size=5,
            indices=slice(None),
        )

        result = manager.regroup([atoms1, atoms2, atoms3, atoms4])

        assert len(result) == 2
        assert len(result[0]) == 2  # file_index 0 (consecutive)
        assert len(result[1]) == 2  # file_index 1 (consecutive)


class TestAtomsShardManagerRedistribute:
    """Test file redistribution with tags."""

    def test_redistribute_writes_with_tag(self):
        """Test redistribute writes files with correct tag."""
        mock_fs = MagicMock()

        atoms1 = Atoms("H2O")
        atoms1.info["file_index"] = 0
        atoms2 = Atoms("H2O")
        atoms2.info["file_index"] = 1

        manager = AtomsShardManager(
            traj_files=["data/traj1.extxyz", "data/traj2.extxyz"],
            shard_size=5,
            indices=slice(None),
        )

        manager.redistribute([[atoms1], [atoms2]], mock_fs, tag="opt")

        assert mock_fs.write_ase.call_count == 2
        # Check first call
        call_args_0 = mock_fs.write_ase.call_args_list[0]
        assert call_args_0[0][0] == Path("data/opt_traj1.extxyz")
        # Check second call
        call_args_1 = mock_fs.write_ase.call_args_list[1]
        assert call_args_1[0][0] == Path("data/opt_traj2.extxyz")


class TestAtomsShardManagerCollect:
    """Test collection from calculation directories."""

    def test_collect_all_successful(self):
        """Test collecting when all tasks succeed."""
        mock_fs = MagicMock()

        # Mock exists to return True for all files
        mock_fs.exists.return_value = True

        # Create mock atoms with is_finished flag
        atoms = Atoms("H2O")
        atoms.info["is_finished"] = True
        mock_fs.read_ase.return_value = [atoms]

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"], shard_size=1, indices=slice(None)
        )
        manager.all_shard_sizes = [1, 1]  # Two shards

        all_atoms, failed = manager.collect(
            calcdir=Path("/calc"),
            remote_fs=mock_fs,
            num_array_tasks=2,
            map_indices=None,
            tag="opt",
        )

        assert len(all_atoms) == 2
        assert failed == []

    def test_collect_missing_file(self):
        """Test collecting when result file is missing."""
        mock_fs = MagicMock()

        # Mock exists to return False for first shard
        mock_fs.exists.side_effect = [False, True]

        atoms = Atoms("H2O")
        atoms.info["is_finished"] = True
        mock_fs.read_ase.return_value = [atoms]

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"], shard_size=1, indices=slice(None)
        )
        manager.all_shard_sizes = [1, 1]

        all_atoms, failed = manager.collect(
            calcdir=Path("/calc"),
            remote_fs=mock_fs,
            num_array_tasks=2,
            map_indices=None,
            tag="opt",
        )

        assert len(all_atoms) == 1
        assert failed == [0]

    def test_collect_unfinished_job(self):
        """Test collecting when job didn't finish."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True

        # Create atoms with is_finished = False
        atoms = Atoms("H2O")
        atoms.info["is_finished"] = False
        mock_fs.read_ase.return_value = [atoms]

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"], shard_size=1, indices=slice(None)
        )
        manager.all_shard_sizes = [1]

        all_atoms, failed = manager.collect(
            calcdir=Path("/calc"),
            remote_fs=mock_fs,
            num_array_tasks=1,
            map_indices=None,
            tag="opt",
            only_finished=True,
        )

        assert failed == [0]


class TestAtomsShardManagerCheckAndHarvest:
    """Test validation and harvesting logic."""

    def test_check_and_harvest_success(self):
        """Test successful check and harvest."""
        mock_fs = MagicMock()

        atoms1 = Atoms("H2O")
        atoms1.info["file_index"] = 0
        atoms2 = Atoms("H2O")
        atoms2.info["file_index"] = 0

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"], shard_size=2, indices=slice(None)
        )
        manager.num_configs_total = 2

        # Should not raise
        manager.check_and_harvest(
            all_atoms=[atoms1, atoms2],
            failed_indices=[],
            local_fs=mock_fs,
            tag="opt",
            savepath=Path("/save"),
        )

        # Verify redistribute was called
        assert mock_fs.write_ase.called

    def test_check_and_harvest_failed_indices(self):
        """Test check_and_harvest raises on failed indices."""
        mock_fs = MagicMock()

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"], shard_size=2, indices=slice(None)
        )
        manager.num_configs_total = 2

        with pytest.raises(Exception, match="Something went wrong"):
            manager.check_and_harvest(
                all_atoms=[],
                failed_indices=[0, 1],
                local_fs=mock_fs,
                tag="opt",
                savepath=Path("/save"),
            )

    def test_check_and_harvest_insufficient_atoms(self):
        """Test check_and_harvest raises when count is too low."""
        mock_fs = MagicMock()

        atoms = Atoms("H2O")
        atoms.info["file_index"] = 0

        manager = AtomsShardManager(
            traj_files=["traj.extxyz"], shard_size=2, indices=slice(None)
        )
        manager.num_configs_total = 5

        with pytest.raises(Exception, match="Collected number"):
            manager.check_and_harvest(
                all_atoms=[atoms],  # Only 1, expected 5
                failed_indices=[],
                local_fs=mock_fs,
                tag="opt",
                savepath=Path("/save"),
            )

"""Tests for script input models."""

import json
from pathlib import Path

import pytest

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.input_models import (
    CmaesSamplingInput,
    DftSinglepointInput,
    LammpsSamplingInput,
    MaceFinetuningInput,
    MaceTrainingInput,
    OptimizeTrajInput,
    OrcaSinglepointInput,
    ScriptInput,
    TorchSimSamplingInput,
)


class TestMaceTrainingInput:
    """Tests for MaceTrainingInput."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        # Create config file
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model: MACE\nbatch_size: 16")

        # Create input JSON
        input_file = tmp_path / "input.json"
        input_data = {
            "database_files": ["data.xyz"],
            "train_runner": "mace_run_train",
            "config_file": str(config_yaml),
            "model_name": "test_model",
            "seed": 42,
            "validation_fraction": 0.2,
        }
        input_file.write_text(json.dumps(input_data))

        # Load
        config = MaceTrainingInput.from_file(input_file)

        # Check values
        assert config.database_files == ["data.xyz"]
        assert config.train_runner == Path("mace_run_train")
        assert config.config_file == config_yaml
        assert config.model_name == "test_model"
        assert config.seed == 42
        assert config.validation_fraction == 0.2

    def test_from_file_not_found(self, tmp_path):
        """Test with non-existent file."""
        with pytest.raises(ScriptInputError) as exc_info:
            MaceTrainingInput.from_file(tmp_path / "nonexistent.json")

        assert "not found" in str(exc_info.value)

    def test_from_file_invalid_json(self, tmp_path):
        """Test with invalid JSON."""
        input_file = tmp_path / "invalid.json"
        input_file.write_text("{ invalid }")

        with pytest.raises(ScriptInputError) as exc_info:
            MaceTrainingInput.from_file(input_file)

        assert "Invalid JSON" in str(exc_info.value)

    def test_from_file_missing_required_field(self, tmp_path):
        """Test with missing required field."""
        input_file = tmp_path / "input.json"
        input_data = {"database_files": ["data.xyz"]}  # Missing other fields
        input_file.write_text(json.dumps(input_data))

        with pytest.raises(ScriptInputError) as exc_info:
            MaceTrainingInput.from_file(input_file)

        assert "missing fields" in str(exc_info.value).lower()

    def test_validate_empty_database_files(self, tmp_path):
        """Test validation with empty database_files."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model: MACE")

        config = MaceTrainingInput(
            database_files=[],  # Empty!
            train_runner=Path("mace_run_train"),
            config_file=config_yaml,
            model_name="test",
            seed=42,
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "cannot be empty" in str(exc_info.value)

    def test_validate_invalid_validation_fraction(self, tmp_path):
        """Test validation with invalid validation_fraction."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model: MACE")

        # Test > 1
        config = MaceTrainingInput(
            database_files=["data.xyz"],
            train_runner=Path("mace_run_train"),
            config_file=config_yaml,
            model_name="test",
            seed=42,
            validation_fraction=1.5,  # Invalid!
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "between 0 and 1" in str(exc_info.value)

        # Test <= 0
        config.validation_fraction = 0.0
        with pytest.raises(ScriptInputError):
            config.validate()

    def test_validate_empty_model_name(self, tmp_path):
        """Test validation with empty model name."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model: MACE")

        config = MaceTrainingInput(
            database_files=["data.xyz"],
            train_runner=Path("mace_run_train"),
            config_file=config_yaml,
            model_name="",  # Empty!
            seed=42,
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "cannot be empty" in str(exc_info.value)


class TestLammpsSamplingInput:
    """Tests for LammpsSamplingInput."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        # Create mock lammps executable
        lammps_exe = tmp_path / "lmp"
        lammps_exe.touch()

        input_file = tmp_path / "input.json"
        input_data = {
            "lammps_path": str(lammps_exe),
            "template_file": "lammps.in.template",
            "data_file": "structure.data",
            "model_files": ["model1.pt", "model2.pt"],
            "element_mapping": {"1": "H", "2": "O"},
            "cutoff_radius": 5.0,
            "bin_count": 10,
            "seed": 42,
        }
        input_file.write_text(json.dumps(input_data))

        config = LammpsSamplingInput.from_file(input_file)

        assert config.lammps_path == lammps_exe
        assert config.template_file == "lammps.in.template"
        assert config.model_files == ["model1.pt", "model2.pt"]
        assert config.cutoff_radius == 5.0
        assert config.bin_count == 10

    def test_validate_lammps_not_found(self, tmp_path):
        """Test validation when LAMMPS executable doesn't exist."""
        config = LammpsSamplingInput(
            lammps_path=Path("/nonexistent/lmp"),
            template_file="template.in",
            data_file="data.in",
            model_files=["model.pt"],
            element_mapping={"1": "H"},
            cutoff_radius=5.0,
            bin_count=10,
            seed=42,
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "not found" in str(exc_info.value)

    def test_validate_empty_model_files(self, tmp_path):
        """Test validation with empty model_files."""
        lammps_exe = tmp_path / "lmp"
        lammps_exe.touch()

        config = LammpsSamplingInput(
            lammps_path=lammps_exe,
            template_file="template.in",
            data_file="data.in",
            model_files=[],  # Empty!
            element_mapping={"1": "H"},
            cutoff_radius=5.0,
            bin_count=10,
            seed=42,
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "cannot be empty" in str(exc_info.value)

    def test_validate_negative_cutoff(self, tmp_path):
        """Test validation with negative cutoff_radius."""
        lammps_exe = tmp_path / "lmp"
        lammps_exe.touch()

        config = LammpsSamplingInput(
            lammps_path=lammps_exe,
            template_file="template.in",
            data_file="data.in",
            model_files=["model.pt"],
            element_mapping={"1": "H"},
            cutoff_radius=-1.0,  # Invalid!
            bin_count=10,
            seed=42,
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "must be positive" in str(exc_info.value)


class TestDftSinglepointInput:
    """Tests for DftSinglepointInput."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        input_file = tmp_path / "input.json"
        input_data = {
            "DATABASE": "structures.xyz",
            "launch_command": "mpirun -np 4 vasp_std",
            "INCAR_kwargs": {
                "xc": "PBE",
                "encut": 500,
                "ismear": 0,
            },
        }
        input_file.write_text(json.dumps(input_data))

        config = DftSinglepointInput.from_file(input_file)

        assert config.DATABASE == "structures.xyz"
        assert config.launch_command == "mpirun -np 4 vasp_std"
        assert config.INCAR_kwargs["xc"] == "PBE"
        assert config.INCAR_kwargs["encut"] == 500

    def test_validate_empty_database(self):
        """Test validation with empty DATABASE."""
        config = DftSinglepointInput(
            DATABASE="",  # Empty!
            launch_command="vasp",
            INCAR_kwargs={},
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "cannot be empty" in str(exc_info.value)

    def test_validate_non_dict_incar(self):
        """Test validation with non-dict INCAR_kwargs."""
        config = DftSinglepointInput(
            DATABASE="data.xyz",
            launch_command="vasp",
            INCAR_kwargs="not a dict",  # Invalid!
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "must be a dictionary" in str(exc_info.value)


class TestOrcaSinglepointInput:
    """Tests for OrcaSinglepointInput."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        input_file = tmp_path / "input.json"
        input_data = {
            "DATABASE": "dft.extxyz",
            "orca_binary": "/opt/orca/orca",
            "orcasimpleinput": "wB97X 6-31G(d) ENGRAD",
            "orcablocks": "%pal nprocs 8 end",
        }
        input_file.write_text(json.dumps(input_data))

        config = OrcaSinglepointInput.from_file(input_file)

        assert config.DATABASE == "dft.extxyz"
        assert config.orca_binary == "/opt/orca/orca"
        assert config.orcasimpleinput == "wB97X 6-31G(d) ENGRAD"
        assert config.orcablocks == "%pal nprocs 8 end"

    def test_validate_empty_database(self):
        """Test validation with empty DATABASE."""
        config = OrcaSinglepointInput(
            DATABASE="",
            orca_binary="/opt/orca/orca",
            orcasimpleinput="wB97X 6-31G(d) ENGRAD",
            orcablocks="%pal nprocs 8 end",
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "DATABASE cannot be empty" in str(exc_info.value)

    def test_validate_empty_orca_binary(self):
        """Test validation with empty orca_binary."""
        config = OrcaSinglepointInput(
            DATABASE="dft.extxyz",
            orca_binary="",
            orcasimpleinput="wB97X 6-31G(d) ENGRAD",
            orcablocks="%pal nprocs 8 end",
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "orca_binary cannot be empty" in str(exc_info.value)

    def test_validate_empty_orcasimpleinput(self):
        """Test validation with empty orcasimpleinput."""
        config = OrcaSinglepointInput(
            DATABASE="dft.extxyz",
            orca_binary="/opt/orca/orca",
            orcasimpleinput="",
            orcablocks="%pal nprocs 8 end",
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "orcasimpleinput cannot be empty" in str(exc_info.value)

    def test_validate_empty_orcablocks(self):
        """Test validation with empty orcablocks."""
        config = OrcaSinglepointInput(
            DATABASE="dft.extxyz",
            orca_binary="/opt/orca/orca",
            orcasimpleinput="wB97X 6-31G(d) ENGRAD",
            orcablocks="",
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "orcablocks cannot be empty" in str(exc_info.value)


class TestOptimizeTrajInput:
    """Tests for OptimizeTrajInput."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        input_file = tmp_path / "input.json"
        input_data = {
            "tag": "optimized",
            "DATABASE": "trajectory.xyz",
            "do_optimization": True,
            "do_steinhardt": False,
            "do_rdf": True,
            "general_args": {"cutoff": 5.0, "nbins": 100},
            "calculator_args": {"optimization_backend": "LJ"},
            "optimization_args": {"fmax": 0.01},
        }
        input_file.write_text(json.dumps(input_data))

        config = OptimizeTrajInput.from_file(input_file)

        assert config.tag == "optimized"
        assert config.DATABASE == "trajectory.xyz"
        assert config.do_optimization is True
        assert config.general_args["cutoff"] == 5.0

    def test_validate_optimization_without_calculator_args(self):
        """Test validation when do_optimization=True but no calculator_args."""
        config = OptimizeTrajInput(
            tag="test",
            DATABASE="data.xyz",
            do_optimization=True,
            do_steinhardt=False,
            do_rdf=False,
            general_args={},
            calculator_args=None,  # Missing!
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "calculator_args required" in str(exc_info.value)

    def test_validate_steinhardt_without_args(self):
        """Test validation when do_steinhardt=True but no steinhardt_args."""
        config = OptimizeTrajInput(
            tag="test",
            DATABASE="data.xyz",
            do_optimization=False,
            do_steinhardt=True,
            do_rdf=False,
            general_args={},
            steinhardt_args=None,  # Missing!
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "steinhardt_args required" in str(exc_info.value)


class TestCmaesSamplingInput:
    """Tests for CmaesSamplingInput."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        input_file = tmp_path / "input.json"
        input_data = {
            "founder": "/path/to/POSCAR",
            "label": "test_evo",
            "random_seed": 123,
            "step_size": 0.12,
            "generations": 500,
            "pop_size": 20,
            "save_nth": 5,
            "batch": True,
            "models": ["model1.pt", "model2.pt"],
            "loss_mode": "energy-only",
            "loss_threshold": -1e4,
            "scaled_center": [0.5, 0.5, 0.5],
            "radius": 3.0,
            "c_r": 15.0,
            "continue_evolution": False,
        }
        input_file.write_text(json.dumps(input_data))

        config = CmaesSamplingInput.from_file(input_file)

        assert config.founder == "/path/to/POSCAR"
        assert config.label == "test_evo"
        assert config.random_seed == 123
        assert config.step_size == 0.12
        assert config.generations == 500
        assert config.pop_size == 20
        assert config.save_nth == 5
        assert config.batch is True
        assert config.models == ["model1.pt", "model2.pt"]
        assert config.loss_mode == "energy-only"
        assert config.loss_threshold == -1e4
        assert config.scaled_center == [0.5, 0.5, 0.5]
        assert config.radius == 3.0
        assert config.c_r == 15.0
        assert config.continue_evolution is False

    def test_from_file_defaults(self, tmp_path):
        """Test loading with only required fields uses correct defaults."""
        input_file = tmp_path / "input.json"
        input_data = {"founder": "/path/to/POSCAR"}
        input_file.write_text(json.dumps(input_data))

        config = CmaesSamplingInput.from_file(input_file)

        assert config.founder == "/path/to/POSCAR"
        assert config.label == "dummy"
        assert config.random_seed == 0
        assert config.step_size == 1.0
        assert config.generations == 1000
        assert config.pop_size is None
        assert config.save_nth == 1
        assert config.batch is False
        assert config.models is None
        assert config.loss_mode == "energy-only"
        assert config.loss_threshold == -1e6
        assert config.scaled_center is None
        assert config.radius is None
        assert config.c_r == 20.0
        assert config.continue_evolution is False
        assert config.generation_checkpoint is None
        assert config.repository_path is None

    def test_from_file_not_found(self, tmp_path):
        """Test with non-existent file."""
        with pytest.raises(ScriptInputError) as exc_info:
            CmaesSamplingInput.from_file(tmp_path / "nonexistent.json")

        assert "not found" in str(exc_info.value)

    def test_from_file_invalid_json(self, tmp_path):
        """Test with invalid JSON."""
        input_file = tmp_path / "invalid.json"
        input_file.write_text("{ invalid }")

        with pytest.raises(ScriptInputError) as exc_info:
            CmaesSamplingInput.from_file(input_file)

        assert "Invalid JSON" in str(exc_info.value)

    def test_from_file_missing_required_field(self, tmp_path):
        """Test with missing required founder field."""
        input_file = tmp_path / "input.json"
        input_data = {"label": "test"}  # Missing founder
        input_file.write_text(json.dumps(input_data))

        with pytest.raises(ScriptInputError) as exc_info:
            CmaesSamplingInput.from_file(input_file)

        assert "missing fields" in str(exc_info.value).lower()

    def test_validate_empty_founder(self):
        """Test validation with empty founder."""
        config = CmaesSamplingInput(founder="")

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "cannot be empty" in str(exc_info.value)

    def test_validate_invalid_generations(self):
        """Test validation with non-positive generations."""
        config = CmaesSamplingInput(founder="/path/to/POSCAR", generations=0)

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "must be positive" in str(exc_info.value)

    def test_validate_invalid_step_size(self):
        """Test validation with non-positive step_size."""
        config = CmaesSamplingInput(founder="/path/to/POSCAR", step_size=-0.1)

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "must be positive" in str(exc_info.value)

    def test_validate_invalid_save_nth(self):
        """Test validation with non-positive save_nth."""
        config = CmaesSamplingInput(founder="/path/to/POSCAR", save_nth=0)

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "must be positive" in str(exc_info.value)

    def test_validate_invalid_loss_mode(self):
        """Test validation with invalid loss_mode."""
        config = CmaesSamplingInput(
            founder="/path/to/POSCAR", loss_mode="invalid-mode"
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "loss_mode must be one of" in str(exc_info.value)

    def test_validate_continue_without_models(self):
        """Test validation when continue_evolution=True but models is None."""
        config = CmaesSamplingInput(
            founder="/path/to/POSCAR",
            continue_evolution=True,
            models=None,
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "models must be provided" in str(exc_info.value)

    def test_validate_valid_config(self):
        """Test validation passes with valid configuration."""
        config = CmaesSamplingInput(
            founder="/path/to/POSCAR",
            models=["model.pt"],
            loss_mode="loss-threshold",
        )

        config.validate()  # Should not raise


class TestValidateWithRoot:
    """Path-existence checks run only when ``validate`` gets a ``root``."""

    def _mace(self, root, **over):
        data = dict(
            database_files=["db.xyz"],
            train_runner="mace_run_train",
            config_file="config.yaml",
            model_name="m",
            seed=0,
        )
        data.update(over)
        return MaceTrainingInput.from_dict(data)

    def test_no_root_skips_path_checks(self, tmp_path):
        # Files don't exist, but without a root no path check runs.
        self._mace(tmp_path).validate()  # must not raise

    def test_root_flags_missing_data_file(self, tmp_path):
        (tmp_path / "config.yaml").write_text("x")  # config present
        with pytest.raises(ScriptInputError, match="database file"):
            self._mace(tmp_path).validate(tmp_path)  # db.xyz missing

    def test_root_passes_when_all_present(self, tmp_path):
        (tmp_path / "db.xyz").write_text("x")
        (tmp_path / "config.yaml").write_text("x")
        self._mace(tmp_path).validate(tmp_path)  # must not raise

    def test_relative_resolved_against_root(self, tmp_path):
        # A "../shared.xyz" path resolves above the calc dir.
        calc = tmp_path / "calcdirs" / "0"
        calc.mkdir(parents=True)
        (tmp_path / "shared.xyz").write_text("x")
        (tmp_path / "config.yaml").write_text("x")
        cfg = self._mace(
            calc,
            database_files=["../../shared.xyz"],
            config_file="../../config.yaml",
        )
        cfg.validate(calc)  # ../../shared.xyz -> tmp_path/shared.xyz, exists

    def test_finetuning_checks_foundation_model(self, tmp_path):
        (tmp_path / "db.xyz").write_text("x")
        (tmp_path / "config.yaml").write_text("x")
        cfg = MaceFinetuningInput.from_dict(
            dict(
                database_files=["db.xyz"],
                train_runner="mace_run_train",
                config_file="config.yaml",
                foundation_model="found.model",  # missing
                model_name="m",
                seed=0,
            )
        )
        with pytest.raises(ScriptInputError, match="foundation_model"):
            cfg.validate(tmp_path)

    def test_torchsim_skips_foundation_sentinel(self, tmp_path):
        (tmp_path / "start.xyz").write_text("x")
        cfg = TorchSimSamplingInput.from_dict(
            dict(
                founder="start.xyz",
                model_paths=["foundation"],  # sentinel, not a path
                ensemble="nvt",
                temperatures=[300.0],
                target_temperature=300.0,
            )
        )
        cfg.validate(tmp_path)  # must not try to stat "foundation"

    def test_dft_root_checks_database(self, tmp_path):
        cfg = DftSinglepointInput(
            DATABASE="missing.xyz", launch_command="vasp", INCAR_kwargs={}
        )
        with pytest.raises(ScriptInputError, match="DATABASE"):
            cfg.validate(tmp_path)


class TestScriptInputBaseClass:
    """Tests for ScriptInput base class functionality."""

    def test_from_dict_filters_unknown_keys(self):
        """Test that from_dict filters unknown keys with warning."""
        data = {
            "founder": "/path/to/POSCAR",
            "label": "test",
            "stub": True,  # Unknown key
            "extra_field": 42,  # Unknown key
        }

        config = CmaesSamplingInput.from_dict(data)

        assert config.founder == "/path/to/POSCAR"
        assert config.label == "test"
        assert not hasattr(config, "stub")
        assert not hasattr(config, "extra_field")

    def test_from_dict_converts_path_fields(self, tmp_path):
        """Test that from_dict converts _path_fields to Path objects."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model: MACE")

        data = {
            "database_files": ["data.xyz"],
            "train_runner": "/usr/bin/mace_run_train",
            "config_file": str(config_yaml),
            "model_name": "test",
            "seed": 42,
        }

        config = MaceTrainingInput.from_dict(data)

        assert isinstance(config.train_runner, Path)
        assert isinstance(config.config_file, Path)

    def test_from_dict_missing_required_field(self):
        """Test that from_dict raises on missing required fields."""
        data = {"label": "test"}  # Missing 'founder'

        with pytest.raises(ScriptInputError) as exc_info:
            CmaesSamplingInput.from_dict(data)

        assert "missing fields" in str(exc_info.value).lower()

    def test_from_file_delegates_to_from_dict(self, tmp_path):
        """Test that from_file with unknown keys still works via from_dict."""
        input_file = tmp_path / "input.json"
        input_data = {
            "founder": "/path/to/POSCAR",
            "stub": True,  # Unknown key - should be filtered
        }
        input_file.write_text(json.dumps(input_data))

        config = CmaesSamplingInput.from_file(input_file)

        assert config.founder == "/path/to/POSCAR"
        assert not hasattr(config, "stub")

    def test_from_dict_no_path_fields(self):
        """Test from_dict with a class that has no _path_fields."""
        data = {
            "DATABASE": "data.xyz",
            "launch_command": "mpirun vasp",
            "INCAR_kwargs": {"encut": 500},
        }

        config = DftSinglepointInput.from_dict(data)

        assert config.DATABASE == "data.xyz"
        assert isinstance(config.DATABASE, str)

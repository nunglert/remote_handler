"""Tests for script input models."""

import json
from pathlib import Path

import pytest

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing import OptimizeArgs, ProcessTrajInput, RdfArgs
from remote_handler.scripts.input_models import (
    DftSinglepointInput,
    MaceFinetuningInput,
    MaceTrainingInput,
    OrcaSinglepointInput,
    ScriptInput,
    SelectConfigsInput,
)


class TestProcessorArgSchemas:
    """The per-step arg schemas: single source of truth for build/validate."""

    def test_optimize_to_dict_drops_none_for_lj(self):
        # LJ needs no model knobs -> only the three required keys survive.
        args = OptimizeArgs(backend="LJ", fmax=0.1, steps=100).to_dict()
        assert args == {"backend": "LJ", "fmax": 0.1, "steps": 100}

    def test_optimize_from_dict_validate_round_trip(self):
        OptimizeArgs.from_dict(
            {"backend": "NNFF", "fmax": 0.1, "steps": 5, "pickle_file": "m"}
        ).validate()

    def test_optimize_missing_required_key_raises(self):
        # Missing 'backend' is caught by from_dict (construction), not validate.
        with pytest.raises(ScriptInputError, match="backend"):
            OptimizeArgs.from_dict({"fmax": 0.1, "steps": 5}).validate()

    def test_optimize_nnff_without_pickle_raises(self):
        with pytest.raises(ScriptInputError, match="pickle_file"):
            OptimizeArgs.from_dict(
                {"backend": "NNFF", "fmax": 0.1, "steps": 5}
            ).validate()

    def test_rdf_validate(self):
        RdfArgs.from_dict({"cutoff": 5.0, "nbins": 200}).validate()
        with pytest.raises(ScriptInputError, match="nbins"):
            RdfArgs.from_dict({"cutoff": 5.0, "nbins": 0}).validate()


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


class TestSelectConfigsInput:
    """Tests for SelectConfigsInput (selection/select_configs.py script)."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        input_file = tmp_path / "input.json"
        input_data = {
            "selector_type": "QUESTSSelector",
            "selector_kwargs": {"n_configs": 100, "method": "msc"},
            "input_files": ["pool.extxyz"],
            "reference_files": ["ref.extxyz"],
        }
        input_file.write_text(json.dumps(input_data))

        config = SelectConfigsInput.from_file(input_file)

        assert config.selector_type == "QUESTSSelector"
        assert config.selector_kwargs == {"n_configs": 100, "method": "msc"}
        assert config.input_files == ["pool.extxyz"]
        assert config.reference_files == ["ref.extxyz"]
        assert config.output_filename == "selected.extxyz"  # default

    def test_from_dict_without_reference_defaults_to_none(self):
        config = SelectConfigsInput.from_dict(
            {
                "selector_type": "StrideSelector",
                "selector_kwargs": {"n_configs": 5},
                "input_files": ["pool.extxyz"],
            }
        )
        assert config.reference_files is None

    def test_validate_empty_selector_type(self):
        config = SelectConfigsInput(
            selector_type="",
            selector_kwargs={"n_configs": 5},
            input_files=["pool.extxyz"],
        )
        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()
        assert "selector_type cannot be empty" in str(exc_info.value)

    def test_validate_non_dict_selector_kwargs(self):
        config = SelectConfigsInput(
            selector_type="StrideSelector",
            selector_kwargs="not a dict",  # Invalid!
            input_files=["pool.extxyz"],
        )
        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()
        assert "selector_kwargs must be a dictionary" in str(exc_info.value)

    def test_validate_empty_input_files(self):
        config = SelectConfigsInput(
            selector_type="StrideSelector",
            selector_kwargs={"n_configs": 5},
            input_files=[],
        )
        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()
        assert "input_files cannot be empty" in str(exc_info.value)


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


class TestProcessTrajInput:
    """Tests for ProcessTrajInput (the processor-pipeline schema)."""

    def test_from_file_valid(self, tmp_path):
        """Test loading from valid JSON file."""
        input_file = tmp_path / "input.json"
        input_data = {
            "tag": "processed",
            "DATABASE": "trajectory.xyz",
            "steps": ["optimize", "rdf"],
            "args": {
                "optimize": {"backend": "LJ", "fmax": 0.01, "steps": 100},
                "rdf": {"cutoff": 5.0, "nbins": 100},
            },
        }
        input_file.write_text(json.dumps(input_data))

        config = ProcessTrajInput.from_file(input_file)

        assert config.tag == "processed"
        assert config.DATABASE == "trajectory.xyz"
        assert config.steps == ["optimize", "rdf"]
        assert config.args["optimize"]["backend"] == "LJ"
        # Absent in JSON -> defaults to empty (built-in-only pipeline).
        assert config.registration_modules == []

    def test_registration_modules_round_trip(self, tmp_path):
        """A downstream pipeline's registration modules survive from_file."""
        input_file = tmp_path / "input.json"
        input_data = {
            "tag": "processed",
            "DATABASE": "trajectory.xyz",
            "steps": ["rdf", "steinhardt"],
            "args": {"rdf": {}, "steinhardt": {}},
            "registration_modules": ["jaxrens_al.processors.steinhardt"],
        }
        input_file.write_text(json.dumps(input_data))

        config = ProcessTrajInput.from_file(input_file)
        assert config.registration_modules == [
            "jaxrens_al.processors.steinhardt"
        ]

    def test_validate_empty_steps(self):
        """Validation fails when no steps are requested."""
        config = ProcessTrajInput(
            tag="test",
            DATABASE="data.xyz",
            steps=[],  # nothing to run
            args={},
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "steps cannot be empty" in str(exc_info.value)

    def test_validate_empty_database(self):
        """Validation fails when DATABASE is empty."""
        config = ProcessTrajInput(
            tag="test",
            DATABASE="",  # Missing!
            steps=["rdf"],
            args={"rdf": {"cutoff": 5.0, "nbins": 100}},
        )

        with pytest.raises(ScriptInputError) as exc_info:
            config.validate()

        assert "DATABASE cannot be empty" in str(exc_info.value)


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

    def test_dft_root_checks_database(self, tmp_path):
        cfg = DftSinglepointInput(
            DATABASE="missing.xyz", launch_command="vasp", INCAR_kwargs={}
        )
        with pytest.raises(ScriptInputError, match="DATABASE"):
            cfg.validate(tmp_path)

    def test_select_configs_no_root_skips_path_checks(self, tmp_path):
        cfg = SelectConfigsInput(
            selector_type="StrideSelector",
            selector_kwargs={"n_configs": 2},
            input_files=["missing.extxyz"],
        )
        cfg.validate()  # must not raise: no root, no path check

    def test_select_configs_root_flags_missing_input_file(self, tmp_path):
        cfg = SelectConfigsInput(
            selector_type="StrideSelector",
            selector_kwargs={"n_configs": 2},
            input_files=["missing.extxyz"],
        )
        with pytest.raises(ScriptInputError, match="input file"):
            cfg.validate(tmp_path)

    def test_select_configs_root_flags_missing_reference_file(self, tmp_path):
        (tmp_path / "pool.extxyz").write_text("x")
        cfg = SelectConfigsInput(
            selector_type="StrideSelector",
            selector_kwargs={"n_configs": 2},
            input_files=["pool.extxyz"],
            reference_files=["missing_ref.extxyz"],
        )
        with pytest.raises(ScriptInputError, match="reference file"):
            cfg.validate(tmp_path)

    def test_select_configs_root_passes_when_all_present(self, tmp_path):
        (tmp_path / "pool.extxyz").write_text("x")
        (tmp_path / "ref.extxyz").write_text("x")
        cfg = SelectConfigsInput(
            selector_type="StrideSelector",
            selector_kwargs={"n_configs": 2},
            input_files=["pool.extxyz"],
            reference_files=["ref.extxyz"],
        )
        cfg.validate(tmp_path)  # must not raise


class TestScriptInputBaseClass:
    """Tests for ScriptInput base class functionality."""

    def test_from_dict_filters_unknown_keys(self):
        """Test that from_dict filters unknown keys with warning."""
        data = {
            "DATABASE": "structures.xyz",
            "launch_command": "vasp",
            "INCAR_kwargs": {"encut": 500},
            "stub": True,  # Unknown key
            "extra_field": 42,  # Unknown key
        }

        config = DftSinglepointInput.from_dict(data)

        assert config.DATABASE == "structures.xyz"
        assert config.launch_command == "vasp"
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
        data = {"launch_command": "vasp"}  # Missing DATABASE + INCAR_kwargs

        with pytest.raises(ScriptInputError) as exc_info:
            DftSinglepointInput.from_dict(data)

        assert "missing fields" in str(exc_info.value).lower()

    def test_from_file_delegates_to_from_dict(self, tmp_path):
        """Test that from_file with unknown keys still works via from_dict."""
        input_file = tmp_path / "input.json"
        input_data = {
            "DATABASE": "structures.xyz",
            "launch_command": "vasp",
            "INCAR_kwargs": {},
            "stub": True,  # Unknown key - should be filtered
        }
        input_file.write_text(json.dumps(input_data))

        config = DftSinglepointInput.from_file(input_file)

        assert config.DATABASE == "structures.xyz"
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

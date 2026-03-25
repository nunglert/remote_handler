"""Tests for configuration management."""

import json
import os
from pathlib import Path

import pytest

from remote_handler.config import (
    RemoteHandlerConfig,
    get_config,
    reset_config,
    set_config,
)
from remote_handler.exceptions import ConfigurationError


class TestRemoteHandlerConfig:
    """Tests for RemoteHandlerConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = RemoteHandlerConfig()
        assert config.config_dir is None
        assert config.slurm_templates_dir is None
        assert config.default_partition == "normal"
        assert config.default_conda_env == "base"

    def test_init_with_custom_values(self, tmp_path):
        """Test initialization with custom values."""
        config = RemoteHandlerConfig(
            config_dir=tmp_path,
            default_partition="debug",
            default_conda_env="test_env",
        )
        assert config.config_dir == tmp_path
        assert config.default_partition == "debug"
        assert config.default_conda_env == "test_env"

    def test_from_env_without_vars(self, monkeypatch):
        """Test from_env when no environment variables are set."""
        monkeypatch.delenv("REMOTE_HANDLER_CONFIG_DIR", raising=False)
        monkeypatch.delenv("REMOTE_HANDLER_SLURM_TEMPLATES", raising=False)

        config = RemoteHandlerConfig.from_env()
        assert config.config_dir is None
        assert config.slurm_templates_dir is None

    def test_from_env_with_config_dir(self, monkeypatch, tmp_path):
        """Test from_env with REMOTE_HANDLER_CONFIG_DIR set."""
        monkeypatch.setenv("REMOTE_HANDLER_CONFIG_DIR", str(tmp_path))

        config = RemoteHandlerConfig.from_env()
        assert config.config_dir == tmp_path

    def test_from_env_with_invalid_config_dir(self, monkeypatch):
        """Test from_env with non-existent config directory."""
        monkeypatch.setenv("REMOTE_HANDLER_CONFIG_DIR", "/nonexistent/path")

        with pytest.raises(ConfigurationError) as exc_info:
            RemoteHandlerConfig.from_env()

        assert "non-existent directory" in str(exc_info.value)
        assert "/nonexistent/path" in str(exc_info.value)

    def test_from_env_with_config_dir_as_file(self, monkeypatch, tmp_path):
        """Test from_env when config_dir points to a file."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")
        monkeypatch.setenv("REMOTE_HANDLER_CONFIG_DIR", str(file_path))

        with pytest.raises(ConfigurationError) as exc_info:
            RemoteHandlerConfig.from_env()

        assert "must be a directory" in str(exc_info.value)

    def test_from_env_with_templates_dir(self, monkeypatch, tmp_path):
        """Test from_env with custom templates directory."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        monkeypatch.setenv(
            "REMOTE_HANDLER_SLURM_TEMPLATES", str(templates_dir)
        )

        config = RemoteHandlerConfig.from_env()
        assert config.slurm_templates_dir == templates_dir

    def test_from_env_with_invalid_templates_dir_warns(self, monkeypatch):
        """Test from_env with invalid templates dir (should warn, not fail)."""
        monkeypatch.setenv("REMOTE_HANDLER_SLURM_TEMPLATES", "/nonexistent")

        # Should not raise, just warn and set to None
        config = RemoteHandlerConfig.from_env()
        assert config.slurm_templates_dir is None

    def test_from_file_valid(self, tmp_path):
        """Test loading configuration from valid JSON file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "config_dir": str(tmp_path / "configs"),
            "default_partition": "gpu",
            "default_conda_env": "ml_env",
        }
        config_file.write_text(json.dumps(config_data))

        config = RemoteHandlerConfig.from_file(config_file)
        assert config.config_dir == tmp_path / "configs"
        assert config.default_partition == "gpu"
        assert config.default_conda_env == "ml_env"

    def test_from_file_not_found(self, tmp_path):
        """Test from_file with non-existent file."""
        config_file = tmp_path / "nonexistent.json"

        with pytest.raises(ConfigurationError) as exc_info:
            RemoteHandlerConfig.from_file(config_file)

        assert "not found" in str(exc_info.value)

    def test_from_file_invalid_json(self, tmp_path):
        """Test from_file with invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(ConfigurationError) as exc_info:
            RemoteHandlerConfig.from_file(config_file)

        assert "Invalid JSON" in str(exc_info.value)

    def test_from_file_not_dict(self, tmp_path):
        """Test from_file when JSON is not an object."""
        config_file = tmp_path / "array.json"
        config_file.write_text("[]")

        with pytest.raises(ConfigurationError) as exc_info:
            RemoteHandlerConfig.from_file(config_file)

        assert "must contain a JSON object" in str(exc_info.value)

    def test_get_config_dir_when_set(self, tmp_path):
        """Test get_config_dir when config_dir is set."""
        config = RemoteHandlerConfig(config_dir=tmp_path)
        assert config.get_config_dir() == tmp_path

    def test_get_config_dir_required_but_not_set(self):
        """Test get_config_dir with required=True when not set."""
        config = RemoteHandlerConfig()

        with pytest.raises(ConfigurationError) as exc_info:
            config.get_config_dir(required=True)

        assert "not set" in str(exc_info.value)
        assert "REMOTE_HANDLER_CONFIG_DIR" in str(exc_info.value)

    def test_get_config_dir_not_required_and_not_set(self):
        """Test get_config_dir with required=False when not set."""
        config = RemoteHandlerConfig()
        assert config.get_config_dir(required=False) is None

    def test_get_config_dir_not_exists(self, tmp_path):
        """Test get_config_dir when directory doesn't exist."""
        non_existent = tmp_path / "nonexistent"
        config = RemoteHandlerConfig(config_dir=non_existent)

        with pytest.raises(ConfigurationError) as exc_info:
            config.get_config_dir()

        assert "does not exist" in str(exc_info.value)

    def test_get_config_file_success(self, tmp_path):
        """Test get_config_file with existing file."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        cluster_config = config_dir / "cluster.json"
        cluster_config.write_text("{}")

        config = RemoteHandlerConfig(config_dir=config_dir)
        result = config.get_config_file("cluster.json")

        assert result == cluster_config

    def test_get_config_file_not_found(self, tmp_path):
        """Test get_config_file with non-existent file."""
        config = RemoteHandlerConfig(config_dir=tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            config.get_config_file("nonexistent.json")

        assert "not found" in str(exc_info.value)

    def test_get_template_dirs_with_custom(self, tmp_path):
        """Test get_template_dirs with custom directory."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()

        config = RemoteHandlerConfig(slurm_templates_dir=custom_dir)
        dirs = config.get_template_dirs()

        assert len(dirs) >= 2
        assert dirs[0] == custom_dir  # Custom has priority
        # Built-in templates should be included as fallback

    def test_get_template_dirs_without_custom(self):
        """Test get_template_dirs without custom directory."""
        config = RemoteHandlerConfig()
        dirs = config.get_template_dirs()

        assert len(dirs) >= 1
        # Should include built-in templates

    def test_find_template_found(self, tmp_path):
        """Test find_template when template exists."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        template = custom_dir / "test.sh.j2"
        template.write_text("#!/bin/bash\n")

        config = RemoteHandlerConfig(slurm_templates_dir=custom_dir)
        found = config.find_template("test.sh.j2")

        assert found == template

    def test_find_template_not_found(self):
        """Test find_template when template doesn't exist."""
        config = RemoteHandlerConfig()
        found = config.find_template("nonexistent_template.j2")

        assert found is None

    def test_validate_valid_config(self, tmp_path):
        """Test validate with valid configuration."""
        config = RemoteHandlerConfig(config_dir=tmp_path)
        config.validate()  # Should not raise

    def test_validate_invalid_config_dir(self):
        """Test validate with invalid config_dir."""
        config = RemoteHandlerConfig(config_dir=Path("/nonexistent"))

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_validate_invalid_templates_dir(self, tmp_path):
        """Test validate with invalid templates directory."""
        config = RemoteHandlerConfig(
            config_dir=tmp_path, slurm_templates_dir=Path("/nonexistent")
        )

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_to_dict(self, tmp_path):
        """Test to_dict conversion."""
        config = RemoteHandlerConfig(
            config_dir=tmp_path,
            default_partition="gpu",
            default_conda_env="test",
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["config_dir"] == str(tmp_path)
        assert result["default_partition"] == "gpu"
        assert result["default_conda_env"] == "test"

    def test_to_dict_with_none_values(self):
        """Test to_dict with None values."""
        config = RemoteHandlerConfig()
        result = config.to_dict()

        assert result["config_dir"] is None
        assert result["slurm_templates_dir"] is None

    def test_repr(self, tmp_path):
        """Test string representation."""
        config = RemoteHandlerConfig(config_dir=tmp_path)
        repr_str = repr(config)

        assert "RemoteHandlerConfig" in repr_str
        assert str(tmp_path) in repr_str


class TestConfigurationSingleton:
    """Tests for global configuration singleton."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_config()

    def test_get_config_initializes_from_env(self):
        """Test that get_config initializes from environment."""
        config = get_config()
        assert isinstance(config, RemoteHandlerConfig)

    def test_get_config_returns_same_instance(self):
        """Test that get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_overrides_singleton(self, tmp_path):
        """Test that set_config changes global instance."""
        custom_config = RemoteHandlerConfig(config_dir=tmp_path)
        set_config(custom_config)

        retrieved = get_config()
        assert retrieved is custom_config
        assert retrieved.config_dir == tmp_path

    def test_reset_config_clears_singleton(self, tmp_path):
        """Test that reset_config clears the singleton."""
        custom_config = RemoteHandlerConfig(config_dir=tmp_path)
        set_config(custom_config)

        reset_config()

        # Next get_config should initialize fresh
        new_config = get_config()
        assert new_config is not custom_config

"""
Unit tests for user configuration management.
Tests loading, saving, and accessing user configuration values.
"""

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from tldwatch.core.user_config import UserConfig, get_user_config


class TestUserConfig:
    """Test UserConfig functionality."""

    def test_config_initialization_no_files(self, temp_config_dir):
        """Test initialization when no config files exist."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path(temp_config_dir)
            config = UserConfig()

            # Should use defaults
            assert config.get_default_provider() is None
            assert config.get_default_temperature() is None
            assert config.get_default_chunking_strategy() is None

    def test_config_load_json(self, temp_config_dir, sample_user_config):
        """Test loading configuration from JSON file."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / ".config" / "tldwatch" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(sample_user_config, f)

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path(temp_config_dir)
            config = UserConfig()

            assert config.get_default_provider() == "openai"
            assert config.get_default_temperature() == 0.7
            assert config.get_default_chunking_strategy() == "standard"
            assert config.is_cache_enabled()

    def test_config_load_yaml(self, temp_config_dir, sample_user_config):
        """Test loading configuration from YAML file."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / "config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(sample_user_config, f)

        config = UserConfig(config_dir=config_dir)

        assert config.get_default_provider() == "openai"
        assert config.get_default_model("openai") == "gpt-4o"
        assert config.get_default_model("anthropic") == "claude-3-5-sonnet-20241022"

    def test_config_precedence_json_over_yaml(self, temp_config_dir):
        """Test that JSON config takes precedence over YAML."""
        config_dir = Path(temp_config_dir)

        # Create both JSON and YAML configs with different values
        json_config = {"default_provider": "openai", "default_temperature": 0.5}
        yaml_config = {"default_provider": "anthropic", "default_temperature": 0.8}

        with open(config_dir / "config.json", "w") as f:
            json.dump(json_config, f)

        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(yaml_config, f)

        config = UserConfig(config_dir=config_dir)

        # Should use JSON values
        assert config.get_default_provider() == "openai"
        assert config.get_default_temperature() == 0.5

    def test_get_provider_default_model_with_config(
        self, temp_config_dir, sample_user_config
    ):
        """Test getting provider default model from config."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / "config.json"

        with open(config_file, "w") as f:
            json.dump(sample_user_config, f)

        config = UserConfig(config_dir=config_dir)

        assert config.get_default_model("openai") == "gpt-4o"
        assert config.get_default_model("anthropic") == "claude-3-5-sonnet-20241022"

    def test_get_provider_default_model_fallback(self, temp_config_dir):
        """Test getting provider default model with fallback."""
        config_dir = Path(temp_config_dir)

        config = UserConfig(config_dir=config_dir)

        # Should return None when no config exists
        assert config.get_default_model("openai") is None
        assert config.get_default_model("anthropic") is None

    def test_get_provider_temperature(self, temp_config_dir, sample_user_config):
        """Test getting provider-specific temperature."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / "config.json"

        with open(config_file, "w") as f:
            json.dump(sample_user_config, f)

        config = UserConfig(config_dir=config_dir)

        assert config.get_default_temperature("openai") == 0.5
        assert config.get_default_temperature("anthropic") is None  # Not set in config

    def test_cache_configuration(self, temp_config_dir):
        """Test cache configuration options."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / "config.json"

        cache_config = {"cache": {"enabled": False, "cache_dir": "/custom/cache/path"}}

        with open(config_file, "w") as f:
            json.dump(cache_config, f)

        config = UserConfig(config_dir=config_dir)

        assert not config.is_cache_enabled()
        assert config.get_cache_dir() == "/custom/cache/path"

    def test_cache_defaults(self, temp_config_dir):
        """Test cache configuration defaults."""
        config_dir = Path(temp_config_dir)

        config = UserConfig(config_dir=config_dir)

        # Should default to enabled
        assert config.is_cache_enabled()

        # Should return None for custom cache dir when not configured
        assert config.get_cache_dir() is None

    def test_create_example_config_json(self, temp_config_dir):
        """Test creating example JSON configuration."""
        config_dir = Path(temp_config_dir)

        config = UserConfig(config_dir=config_dir)
        config.create_example_config()

        config_file = config_dir / "config.json"
        assert config_file.exists()

        with open(config_file, "r") as f:
            loaded_config = json.load(f)

        assert "default_provider" in loaded_config
        assert "providers" in loaded_config
        assert "cache" in loaded_config

    def test_create_example_config_yaml(self, temp_config_dir):
        """Test creating example configuration and loading it works."""
        config_dir = Path(temp_config_dir)

        config = UserConfig(config_dir=config_dir)
        config.create_example_config()

        config_file = config_dir / "config.json"
        assert config_file.exists()

        # Create a new UserConfig instance to test loading
        config2 = UserConfig(config_dir=config_dir)
        assert config2.get_default_provider() == "openai"

    def test_get_config_info(self, temp_config_dir, sample_user_config):
        """Test getting configuration information."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / "config.json"

        with open(config_file, "w") as f:
            json.dump(sample_user_config, f)

        config = UserConfig(config_dir=config_dir)

        # Test that we can get various config values
        assert config.get_default_provider() == "openai"
        assert config.is_cache_enabled()
        assert config.get_config_path() == config_file

    def test_invalid_json_config(self, temp_config_dir):
        """Test handling of invalid JSON configuration."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / "config.json"

        # Write invalid JSON
        with open(config_file, "w") as f:
            f.write("{ invalid json }")

        config = UserConfig(config_dir=config_dir)

        # Should fall back to defaults
        assert config.get_default_provider() is None
        assert config.get_default_temperature() is None

    def test_invalid_yaml_config(self, temp_config_dir):
        """Test handling of invalid YAML configuration."""
        config_dir = Path(temp_config_dir)
        config_file = config_dir / "config.yaml"

        # Write invalid YAML
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content:")

        config = UserConfig(config_dir=config_dir)

        # Should fall back to defaults
        assert config.get_default_provider() is None
        assert config.get_default_temperature() is None


class TestGetUserConfig:
    """Test the get_user_config() function."""

    def test_get_user_config_singleton(self):
        """Test that get_user_config returns the same instance."""
        config1 = get_user_config()
        config2 = get_user_config()

        assert config1 is config2

    def test_get_user_config_instance_type(self):
        """Test that get_user_config returns UserConfig instance."""
        config = get_user_config()
        assert isinstance(config, UserConfig)

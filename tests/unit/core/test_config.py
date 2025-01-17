import json

import pytest

from tests.conftest import PROVIDER_TEST_CONFIG
from tldwatch.core.config import Config, ConfigError


def test_config_defaults(tmp_path):
    """Test default configuration values"""
    config_path = tmp_path / "test_config.json"
    config = Config({}, config_path)
    assert config.get("provider") == "openai"
    assert config.get("chunk_size") == 4000
    assert config.get("chunk_overlap") == 200
    assert config.get("temperature") == 0.7
    assert not config.get("use_full_context")


def test_config_load_save(tmp_path):
    """Test loading and saving configuration"""
    config_path = tmp_path / "test_config.json"
    config_data = {
        "provider": "groq",
        "model": PROVIDER_TEST_CONFIG["groq"]["default_model"],
        "chunk_size": 5000,
        "temperature": 0.8,
    }

    config = Config(config_data, config_path)
    config.save()

    # Verify file was created with correct content
    assert config_path.exists()
    with open(config_path) as f:
        saved_data = json.load(f)
    assert saved_data == config_data

    # Test loading
    loaded_config = Config.load(config_path)
    assert loaded_config.get("provider") == "groq"
    assert loaded_config.get("model") == PROVIDER_TEST_CONFIG["groq"]["default_model"]
    assert loaded_config.get("chunk_size") == 5000
    assert loaded_config.get("temperature") == 0.8


def test_provider_validation(tmp_path):
    """Test provider config validation"""
    config_path = tmp_path / "test_config.json"

    # Test valid provider
    config = Config(
        {
            "provider": "openai",
            "model": PROVIDER_TEST_CONFIG["openai"]["default_model"],
        },
        config_path,
    )
    assert config.validate_provider_config()

    # Test invalid provider
    config = Config({"provider": "invalid", "model": "invalid-model"}, config_path)
    assert not config.validate_provider_config()


def test_environment_handling(monkeypatch, tmp_path):
    """Test XDG config path handling"""
    test_config_dir = tmp_path / "test_config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(test_config_dir))

    config_path = Config.get_config_path()
    assert str(test_config_dir) in str(config_path)
    assert "tldwatch" in str(config_path)

    # Test home directory fallback
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    config_path = Config.get_config_path()
    assert ".config/tldwatch" in str(config_path)


def test_config_updates(tmp_path):
    """Test configuration updates"""
    config_path = tmp_path / "test_config.json"
    config = Config({"provider": "openai"}, config_path)

    config.update({"model": "new-model", "temperature": 0.5})

    assert config.get("model") == "new-model"
    assert config.get("temperature") == 0.5
    assert config.get("provider") == "openai"


def test_invalid_config_file(tmp_path):
    """Test handling of invalid configuration file"""
    config_path = tmp_path / "test_config.json"

    # Create invalid JSON file
    config_path.write_text("{invalid json")

    with pytest.raises(ConfigError):
        Config.load(config_path)


def test_config_reset(tmp_path):
    """Test configuration reset"""
    config_path = tmp_path / "test_config.json"
    config = Config(
        {"provider": "groq", "model": "custom-model", "temperature": 0.9}, config_path
    )

    config.reset()

    assert config.get("provider") == "openai"
    assert config.get("model") == "gpt-4o-mini"
    assert config.get("temperature") == 0.7

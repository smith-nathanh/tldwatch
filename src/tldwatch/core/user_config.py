"""
User configuration management for TLDWatch.
Supports user config files at ~/.config/tldwatch/config.json or ~/.config/tldwatch/config.yaml
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class UserConfig:
    """
    Manages user configuration files for TLDWatch.

    Supports configuration files at:
    - ~/.config/tldwatch/config.json
    - ~/.config/tldwatch/config.yaml
    - ~/.config/tldwatch/config.yml

    Configuration format:
    {
        "default_provider": "openai",
        "default_temperature": 0.7,
        "default_chunking_strategy": "standard",
        "providers": {
            "openai": {
                "default_model": "gpt-4o",
                "temperature": 0.5
            },
            "anthropic": {
                "default_model": "claude-3-5-sonnet-20241022"
            }
        }
    }
    """

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "tldwatch"
        self.config_paths = [
            self.config_dir / "config.json",
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
        ]
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        """Load user configuration from file"""
        self._config = {}

        for config_path in self.config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        if config_path.suffix == ".json":
                            self._config = json.load(f)
                        else:  # .yaml or .yml
                            self._config = yaml.safe_load(f)

                    logger.info(f"Loaded user config from: {config_path}")
                    return

                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
                    continue

        logger.debug("No user config file found, using defaults")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        providers = self._config.get("providers", {})
        return providers.get(provider, {})

    def get_default_provider(self) -> Optional[str]:
        """Get the user's default provider"""
        return self.get("default_provider")

    def get_default_model(self, provider: str) -> Optional[str]:
        """Get the user's default model for a provider"""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("default_model")

    def get_default_temperature(
        self, provider: Optional[str] = None
    ) -> Optional[float]:
        """Get the user's default temperature, optionally for a specific provider"""
        if provider:
            provider_config = self.get_provider_config(provider)
            temp = provider_config.get("temperature")
            if temp is not None:
                return temp

        return self.get("default_temperature")

    def get_default_chunking_strategy(self) -> Optional[str]:
        """Get the user's default chunking strategy"""
        return self.get("default_chunking_strategy")

    def create_example_config(self) -> None:
        """Create an example configuration file"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        example_config = {
            "default_provider": "openai",
            "default_temperature": 0.7,
            "default_chunking_strategy": "standard",
            "cache": {
                "enabled": True,
                "cache_dir": None,  # Uses default ~/.cache/tldwatch/summaries if None
                "max_age_days": 30,  # Cleanup entries older than this
            },
            "providers": {
                "openai": {"default_model": "gpt-4o-mini", "temperature": 0.7},
                "anthropic": {
                    "default_model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.5,
                },
                "google": {"default_model": "gemini-2.5-flash"},
                "groq": {"default_model": "llama-3.1-8b-instant"},
                "deepseek": {"default_model": "deepseek-chat"},
                "cerebras": {"default_model": "llama3.1-8b"},
                "ollama": {"default_model": "llama3.1:8b"},
            },
        }

        config_path = self.config_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(example_config, f, indent=2)

        logger.info(f"Created example config at: {config_path}")
        return config_path

    def has_config(self) -> bool:
        """Check if user has a configuration file"""
        return any(path.exists() for path in self.config_paths)

    def get_config_path(self) -> Optional[Path]:
        """Get the path to the active configuration file"""
        for path in self.config_paths:
            if path.exists():
                return path
        return None

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration settings"""
        default_cache_config = {"enabled": True, "cache_dir": None, "max_age_days": 30}
        return self.get("cache", default_cache_config)

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled"""
        cache_config = self.get_cache_config()
        return cache_config.get("enabled", True)

    def get_cache_dir(self) -> Optional[str]:
        """Get custom cache directory if specified"""
        cache_config = self.get_cache_config()
        return cache_config.get("cache_dir")

    def get_cache_max_age_days(self) -> int:
        """Get maximum cache age in days"""
        cache_config = self.get_cache_config()
        return cache_config.get("max_age_days", 30)


# Global user config instance
_user_config = None


def get_user_config() -> UserConfig:
    """Get the global user configuration instance"""
    global _user_config
    if _user_config is None:
        _user_config = UserConfig()
    return _user_config


def reload_user_config() -> None:
    """Reload the user configuration"""
    global _user_config
    _user_config = None
    get_user_config()

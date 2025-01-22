import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigError(Exception):
    """Base exception for configuration errors"""

    pass


class Config:
    """Configuration management for tldwatch"""

    # Default configuration values
    DEFAULTS = {
        "provider": "openai",
        "chunk_size": 4000,
        "chunk_overlap": 200,
        "temperature": 0.7,
        "use_full_context": False,
    }

    # Provider-specific default models
    PROVIDER_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "deepseek": "deepseek-chat",
        "groq": "mixtral-8x7b-32768",
        "cerebras": "llama3.1-8b",
        "google": "gemini-1.5-flash",
        "ollama": "llama3.1:8b",
    }

    def __init__(self, config_data: Dict[str, Any], config_path: Optional[Path] = None):
        """
        Initialize configuration with provided data

        Args:
            config_data: Dictionary containing configuration values
            config_path: Optional path to the configuration file
        """
        self._config = config_data
        self._config_path = config_path or self.get_config_path()

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the path to the configuration file"""
        # Use XDG Base Directory specification if possible
        if xdg_config_home := os.environ.get("XDG_CONFIG_HOME"):
            config_dir = Path(xdg_config_home) / "tldwatch"
        else:
            config_dir = Path.home() / ".config" / "tldwatch"

        return config_dir / "config.json"

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from file, creating with defaults if it doesn't exist

        Args:
            config_path: Optional path to the configuration file

        Returns:
            Config instance with loaded or default values
        """
        config_path = config_path or cls.get_config_path()

        # Start with default values
        config_data = cls.DEFAULTS.copy()

        # Load from file if it exists
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_data = json.load(f)
                config_data.update(file_data)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid configuration file: {str(e)}")
            except Exception as e:
                raise ConfigError(f"Error loading configuration: {str(e)}")

        # Set default model if not specified
        if (
            "model" not in config_data
            and config_data["provider"] in cls.PROVIDER_MODELS
        ):
            config_data["model"] = cls.PROVIDER_MODELS[config_data["provider"]]

        return cls(config_data, config_path)

    def save(self):
        """Save configuration to file"""
        config_path = self._config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key to retrieve
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value

        Args:
            key: Configuration key to set
            value: Value to set
        """
        self._config[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        """
        Update multiple configuration values

        Args:
            values: Dictionary of configuration keys and values to update
        """
        self._config.update(values)

    def reset(self) -> None:
        """Reset configuration to default values"""
        self._config = self.DEFAULTS.copy()
        if self._config["provider"] in self.PROVIDER_MODELS:
            self._config["model"] = self.PROVIDER_MODELS[self._config["provider"]]

    @property
    def current_provider(self) -> str:
        """Get current provider name"""
        return self._config.get("provider", self.DEFAULTS["provider"])

    @property
    def current_model(self) -> Optional[str]:
        """Get current model name"""
        return self._config.get(
            "model", self.PROVIDER_MODELS.get(self.current_provider)
        )

    def validate_provider_config(self) -> bool:
        """
        Validate provider-specific configuration

        Returns:
            True if configuration is valid, False otherwise
        """
        provider = self.current_provider

        # Check if provider is supported
        if provider not in self.PROVIDER_MODELS:
            return False

        # Check if model is set for provider
        if "model" not in self._config:
            return False

        return True

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration values"""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of configuration values"""
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key"""
        return key in self._config

    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self._config, indent=2)

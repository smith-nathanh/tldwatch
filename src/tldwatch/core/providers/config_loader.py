"""Provider configuration loader utility"""

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for configuration loading. Install with: pip install pyyaml"
    )


def load_provider_config() -> Dict[str, Any]:
    """Load provider configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Provider config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("providers", {})


def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """Get configuration for a specific provider"""
    config = load_provider_config()

    if provider_name not in config:
        raise ValueError(f"Provider '{provider_name}' not found in configuration")

    return config[provider_name]


def get_default_model(provider_name: str) -> str:
    """Get the default model for a provider"""
    provider_config = get_provider_config(provider_name)
    return provider_config.get("default_model", "")


def get_context_windows(provider_name: str) -> Dict[str, Any]:
    """Get context windows configuration for a provider"""
    provider_config = get_provider_config(provider_name)
    return provider_config.get("context_windows", {})


def get_rate_limits(provider_name: str) -> Optional[Dict[str, Any]]:
    """Get rate limits configuration for a provider"""
    provider_config = get_provider_config(provider_name)
    return provider_config.get("rate_limits")

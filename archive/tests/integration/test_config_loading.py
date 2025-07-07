#!/usr/bin/env python3
"""
Test script to verify that the provider configuration loading from YAML works correctly.
"""

from src.tldwatch.core.providers.unified_provider import UnifiedProvider


def test_config_loading():
    """Test that provider configurations are loaded correctly from config.yaml"""

    # Test provider listing
    providers = UnifiedProvider.list_providers()
    expected_providers = [
        "openai",
        "anthropic",
        "google",
        "groq",
        "deepseek",
        "cerebras",
        "ollama",
    ]

    print("Available providers:", providers)
    assert set(providers) == set(
        expected_providers
    ), f"Expected {expected_providers}, got {providers}"

    # Test provider initialization
    provider = UnifiedProvider("openai")
    print("OpenAI provider config:", provider.config)

    # Test that config has all required fields
    assert provider.config.name == "openai"
    assert provider.config.api_key_env == "OPENAI_API_KEY"
    assert provider.config.api_base == "https://api.openai.com/v1"
    assert provider.config.default_model == "gpt-4o-mini"

    # Test get_default_model class method
    default_model = UnifiedProvider.get_default_model("openai")
    print("Default OpenAI model:", default_model)
    assert default_model == "gpt-4o-mini"

    # Test another provider
    anthropic_provider = UnifiedProvider("anthropic")
    assert anthropic_provider.config.name == "anthropic"
    assert anthropic_provider.config.api_key_env == "ANTHROPIC_API_KEY"
    assert anthropic_provider.config.default_model == "claude-3-5-sonnet-20241022"

    # Test ollama (which has null api_key_env)
    ollama_provider = UnifiedProvider("ollama")
    assert ollama_provider.config.api_key_env is None
    assert ollama_provider.config.api_base == "http://localhost:11434/v1"

    print("✅ All configuration loading tests passed!")


if __name__ == "__main__":
    test_config_loading()

#!/usr/bin/env python3
"""
Demo of the user configuration system for TLDWatch.
Shows how users can customize their defaults.
"""

import asyncio
import json
from pathlib import Path


async def demo_user_config():
    """Demonstrate user configuration functionality"""
    print("=== TLDWatch User Configuration Demo ===\n")

    # Import after setting up the demo environment
    from tldwatch import SimpleSummarizer
    from tldwatch.core.providers.unified_provider import UnifiedProvider
    from tldwatch.core.user_config import get_user_config, reload_user_config

    # 1. Show default behavior without user config
    print("1. Default behavior (no user config):")

    # Temporarily move any existing config
    config_dir = Path.home() / ".config" / "tldwatch"
    backup_path = None
    if config_dir.exists():
        backup_path = config_dir.with_suffix(".backup")
        if backup_path.exists():
            import shutil

            shutil.rmtree(backup_path)
        config_dir.rename(backup_path)

    try:
        # Reload config to clear any cached config
        reload_user_config()

        provider = UnifiedProvider()
        print(f"   Default provider: {provider.config.name}")
        print(f"   Default model: {provider.model}")
        print(f"   Default temperature: {provider.temperature}")
        print(f"   Default chunking: {provider.chunking_strategy.value}")

        # 2. Create user configuration
        print("\n2. Creating user configuration:")

        user_config = get_user_config()
        config_path = user_config.create_example_config()
        print(f"   Created config at: {config_path}")

        # 3. Show default config values
        print("\n3. Default configuration values:")
        with open(config_path, "r") as f:
            config_data = json.load(f)

        print(f"   Default provider: {config_data['default_provider']}")
        print(f"   Default temperature: {config_data['default_temperature']}")
        print(f"   Default chunking: {config_data['default_chunking_strategy']}")
        print("   Provider-specific models:")
        for provider_name, provider_config in config_data["providers"].items():
            model = provider_config.get("default_model", "Not set")
            temp = provider_config.get("temperature", "Not set")
            print(f"     {provider_name}: {model} (temp: {temp})")

        # 4. Test with default config
        print("\n4. Testing with default configuration:")
        reload_user_config()  # Reload to pick up new config

        provider = UnifiedProvider()
        print(f"   Provider: {provider.config.name}")
        print(f"   Model: {provider.model}")
        print(f"   Temperature: {provider.temperature}")
        print(f"   Chunking: {provider.chunking_strategy.value}")

        # 5. Customize configuration
        print("\n5. Customizing configuration:")

        custom_config = {
            "default_provider": "anthropic",
            "default_temperature": 0.3,
            "default_chunking_strategy": "large",
            "providers": {
                "openai": {"default_model": "gpt-4o", "temperature": 0.8},
                "anthropic": {
                    "default_model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.2,
                },
                "google": {"default_model": "gemini-2.5-pro"},
            },
        }

        with open(config_path, "w") as f:
            json.dump(custom_config, f, indent=2)

        print("   Updated configuration:")
        print(f"     Default provider: {custom_config['default_provider']}")
        print(f"     Default temperature: {custom_config['default_temperature']}")
        print(f"     Default chunking: {custom_config['default_chunking_strategy']}")

        # 6. Test with custom config
        print("\n6. Testing with custom configuration:")
        reload_user_config()  # Reload to pick up changes

        # Test default behavior
        provider = UnifiedProvider()
        print(
            f"   Default: {provider.config.name}, {provider.model}, temp={provider.temperature}, chunking={provider.chunking_strategy.value}"
        )

        # Test specific providers
        openai_provider = UnifiedProvider(provider="openai")
        print(
            f"   OpenAI: {openai_provider.config.name}, {openai_provider.model}, temp={openai_provider.temperature}"
        )

        google_provider = UnifiedProvider(provider="google")
        print(
            f"   Google: {google_provider.config.name}, {google_provider.model}, temp={google_provider.temperature}"
        )

        # 7. Test parameter override
        print("\n7. Testing parameter override:")
        override_provider = UnifiedProvider(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.9,
            chunking_strategy="small",
        )
        print(
            f"   Override: {override_provider.config.name}, {override_provider.model}, temp={override_provider.temperature}, chunking={override_provider.chunking_strategy.value}"
        )

        # 8. Test SimpleSummarizer with config
        print("\n8. Testing SimpleSummarizer with user config:")
        summarizer = SimpleSummarizer()
        print(f"   Available providers: {summarizer.list_providers()}")
        print(f"   Available chunking: {summarizer.list_chunking_strategies()}")

        # Show what would be used for different scenarios
        print("\n   What would be used:")
        print("   - summarize_video('video_id'):")
        temp_provider = UnifiedProvider()
        print(
            f"     Provider: {temp_provider.config.name}, Model: {temp_provider.model}"
        )

        print("   - summarize_video('video_id', provider='openai'):")
        temp_provider = UnifiedProvider(provider="openai")
        print(
            f"     Provider: {temp_provider.config.name}, Model: {temp_provider.model}"
        )

        print("   - summarize_video('video_id', temperature=0.1):")
        temp_provider = UnifiedProvider(temperature=0.1)
        print(
            f"     Provider: {temp_provider.config.name}, Temperature: {temp_provider.temperature}"
        )

        # 9. YAML config support
        print("\n9. Testing YAML configuration support:")
        yaml_config_path = config_path.with_suffix(".yaml")

        yaml_content = """
default_provider: groq
default_temperature: 0.4
default_chunking_strategy: small

providers:
  groq:
    default_model: llama-3.1-70b-versatile
    temperature: 0.6
  openai:
    default_model: gpt-4o
"""

        with open(yaml_config_path, "w") as f:
            f.write(yaml_content)

        # Remove JSON config so YAML takes precedence
        config_path.unlink()

        reload_user_config()
        provider = UnifiedProvider()
        print(
            f"   YAML config: {provider.config.name}, {provider.model}, temp={provider.temperature}, chunking={provider.chunking_strategy.value}"
        )

        print("\n=== User Configuration Demo Complete ===")
        print("\nKey Features:")
        print("✓ Supports both JSON and YAML configuration files")
        print("✓ Located at ~/.config/tldwatch/config.json or config.yaml")
        print("✓ Provides defaults for provider, model, temperature, and chunking")
        print("✓ Provider-specific overrides supported")
        print("✓ Parameters can still override config values")
        print("✓ Graceful fallback to package defaults if no config")
        print("✓ CLI commands to create and view configuration")

    finally:
        # Restore original config if it existed
        if backup_path and backup_path.exists():
            if config_dir.exists():
                import shutil

                shutil.rmtree(config_dir)
            backup_path.rename(config_dir)


if __name__ == "__main__":
    try:
        asyncio.run(demo_user_config())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback

        traceback.print_exc()

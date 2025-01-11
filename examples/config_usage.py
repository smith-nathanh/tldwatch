"""
Example usage of the tldwatch configuration system.

This example demonstrates:
1. Loading and saving configuration
2. Getting and setting values
3. Updating multiple values
4. Using different providers and models
5. Resetting to defaults
"""

from tldwatch.core.config import Config


def main():
    # Load existing configuration or create with defaults
    config = Config.load()
    print("Initial configuration:")
    print(config)
    print()

    # Get individual values
    print("Current settings:")
    print(f"Provider: {config.get('provider')}")
    print(f"Model: {config['model']}")  # Dictionary-style access
    print(f"Temperature: {config.get('temperature', 0.7)}")  # With default value
    print()

    # Set individual values
    print("Updating individual settings...")
    config.set("temperature", 0.8)
    config["chunk_size"] = 5000  # Dictionary-style setting
    print(config)
    print()

    # Update multiple values
    print("Updating multiple settings...")
    config.update(
        {"provider": "groq", "model": "mixtral-8x7b-32768", "use_full_context": True}
    )
    print(config)
    print()

    # Provider-specific configuration
    print("Provider information:")
    print(f"Current provider: {config.current_provider}")
    print(f"Current model: {config.current_model}")
    print(f"Valid provider config: {config.validate_provider_config()}")
    print()

    # Reset to defaults
    print("Resetting to defaults...")
    config.reset()
    print(config)
    print()

    # Save changes
    print("Saving configuration...")
    config.save()
    print(f"Configuration saved to: {config.get_config_path()}")


if __name__ == "__main__":
    main()

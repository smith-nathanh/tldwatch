"""
Configuration example for tldwatch.

This example shows how to:
1. Create and manage configuration files
2. Use configuration with the library
3. Understanding configuration precedence
"""

import asyncio
import json
import os

from tldwatch import Summarizer
from tldwatch.core.user_config import get_user_config


def configuration_basics():
    """Show basic configuration management"""
    print("=== Configuration Basics ===")

    user_config = get_user_config()

    if user_config.has_config():
        config_path = user_config.get_config_path()
        print(f"✅ Config found at: {config_path}")
        print(f"   Default provider: {user_config.get_default_provider() or 'Not set'}")
        print(
            f"   Default temperature: {user_config.get_default_temperature() or 'Not set'}"
        )
        print(f"   Cache enabled: {user_config.is_cache_enabled()}")
    else:
        print("❌ No configuration file found")
        print("   Creating example config...")
        config_path = user_config.create_example_config()
        print(f"✅ Created config at: {config_path}")


def show_config_format():
    """Show what a config file looks like"""
    print("\n=== Configuration Format ===")

    example_config = {
        "default_provider": "openai",
        "default_temperature": 0.7,
        "default_chunking_strategy": "standard",
        "cache": {"enabled": True, "max_age_days": 30},
        "providers": {
            "openai": {"default_model": "gpt-4o-mini", "temperature": 0.7},
            "groq": {"default_model": "llama-3.1-8b-instant"},
        },
    }

    print("Example ~/.config/tldwatch/config.json:")
    print(json.dumps(example_config, indent=2))


async def using_config():
    """Show how configuration affects library usage"""
    print("\n=== Using Configuration ===")

    # Check if we have API keys
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")):
        print("⚠️  Set OPENAI_API_KEY or GROQ_API_KEY to run this example")
        return

    # Use defaults from config
    summarizer = Summarizer()
    print("Using default configuration from config file...")

    try:
        # This uses whatever is set in your config file
        summary = await summarizer.summarize("dQw4w9WgXcQ")
        print(f"Summary: {summary[:100]}...")

        # Override config with explicit parameters
        print("\nOverriding configuration...")
        summary = await summarizer.summarize(
            "dQw4w9WgXcQ",
            provider="groq" if os.environ.get("GROQ_API_KEY") else "openai",
            temperature=0.3,
        )
        print(f"Summary: {summary[:100]}...")

    except Exception as e:
        print(f"Error: {str(e)}")


def configuration_precedence():
    """Show configuration priority order"""
    print("\n=== Configuration Precedence ===")
    print("Priority order (highest to lowest):")
    print("1. Function parameters (summarizer.summarize(provider='openai'))")
    print("2. User config file (~/.config/tldwatch/config.json)")
    print("3. Built-in defaults")

    user_config = get_user_config()
    print("\nCurrent effective defaults:")
    print(f"  Provider: {user_config.get_default_provider() or 'openai (built-in)'}")
    print(f"  Temperature: {user_config.get_default_temperature() or '0.7 (built-in)'}")
    print(
        f"  Chunking: {user_config.get_default_chunking_strategy() or 'standard (built-in)'}"
    )


def main():
    """Run configuration examples"""
    print("TLDWatch Configuration Examples")
    print("=" * 35)

    configuration_basics()
    show_config_format()

    if os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY"):
        asyncio.run(using_config())
    else:
        print("\n⚠️  Set API keys to run library examples")

    configuration_precedence()

    print("\n=== CLI Configuration Commands ===")
    print("tldwatch --create-config    # Create example config")
    print("tldwatch --show-config      # Show current config")
    print("tldwatch --list-providers   # List available providers")


if __name__ == "__main__":
    main()

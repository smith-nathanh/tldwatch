#!/usr/bin/env python3
"""
Simplified CLI for tldwatch using the new unified provider system.
Much simpler than the original CLI with fewer options and cleaner interface.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from ..core.simple_summarizer import SimpleSummarizer
from ..core.user_config import get_user_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Simplified YouTube video summarizer using various LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  tldwatch-simple "https://youtube.com/watch?v=dQw4w9WgXcQ"
  
  # Use a specific provider
  tldwatch-simple "dQw4w9WgXcQ" --provider anthropic
  
  # Use a specific model and chunking strategy
  tldwatch-simple "video_id" --provider openai --model gpt-4o --chunking large
  
  # Submit entire transcript without chunking
  tldwatch-simple "video_url" --chunking none
  
  # Summarize direct text
  tldwatch-simple "Your text here..." --chunking none

Configuration:
  # Create user configuration file
  tldwatch-simple --create-config
  
  # Show current configuration
  tldwatch-simple --show-config
  
  # List available options
  tldwatch-simple --list-providers

User configuration file: ~/.config/tldwatch/config.json (or .yaml)
Available providers: openai, anthropic, google, groq, deepseek, cerebras, ollama
Available chunking strategies: none, standard, small, large
        """
    )
    
    # Input argument (optional for info commands)
    parser.add_argument(
        "input",
        nargs="?",
        help="YouTube URL, video ID, or direct text to summarize"
    )
    
    # Optional arguments
    parser.add_argument(
        "--provider", "-p",
        choices=SimpleSummarizer.list_providers(),
        help="LLM provider to use (uses user config default or 'openai' if not specified)"
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Specific model to use (uses provider default if not specified)"
    )
    
    parser.add_argument(
        "--chunking", "-c",
        choices=SimpleSummarizer.list_chunking_strategies(),
        help="Chunking strategy for long texts (uses user config default or 'standard' if not specified)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        help="Generation temperature 0.0-1.0 (uses user config default or 0.7 if not specified)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file (prints to stdout if not specified)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit"
    )
    
    parser.add_argument(
        "--show-defaults",
        action="store_true", 
        help="Show default models for each provider and exit"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create an example user configuration file and exit"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current user configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle info commands
    if args.list_providers:
        print("Available providers:")
        for provider in SimpleSummarizer.list_providers():
            default_model = SimpleSummarizer.get_default_model(provider)
            print(f"  {provider} (default model: {default_model})")
        return
    
    if args.show_defaults:
        print("Default models for each provider:")
        for provider in SimpleSummarizer.list_providers():
            default_model = SimpleSummarizer.get_default_model(provider)
            print(f"  {provider}: {default_model}")
        return
    
    if args.create_config:
        user_config = get_user_config()
        config_path = user_config.create_example_config()
        print(f"Created example configuration file at: {config_path}")
        print("Edit this file to customize your defaults.")
        return
    
    if args.show_config:
        user_config = get_user_config()
        if user_config.has_config():
            config_path = user_config.get_config_path()
            print(f"User configuration loaded from: {config_path}")
            print(f"Default provider: {user_config.get_default_provider() or 'Not set (will use openai)'}")
            print(f"Default temperature: {user_config.get_default_temperature() or 'Not set (will use 0.7)'}")
            print(f"Default chunking strategy: {user_config.get_default_chunking_strategy() or 'Not set (will use standard)'}")
            print("\nProvider-specific settings:")
            for provider in SimpleSummarizer.list_providers():
                model = user_config.get_default_model(provider)
                temp = user_config.get_default_temperature(provider)
                if model or temp:
                    print(f"  {provider}:")
                    if model:
                        print(f"    model: {model}")
                    if temp:
                        print(f"    temperature: {temp}")
        else:
            print("No user configuration file found.")
            print("Use --create-config to create an example configuration.")
        return
    
    # Check if input is required
    if args.input is None:
        print("Error: Input is required for summarization", file=sys.stderr)
        print("Use --help to see available options", file=sys.stderr)
        sys.exit(1)
    
    # Validate temperature
    if args.temperature is not None and not 0.0 <= args.temperature <= 1.0:
        print("Error: Temperature must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create summarizer and generate summary
        summarizer = SimpleSummarizer()
        
        # Create a temporary provider to get the actual values that will be used
        from ..core.providers.unified_provider import UnifiedProvider
        temp_provider = UnifiedProvider(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            chunking_strategy=args.chunking
        )
        
        print(f"Generating summary using {temp_provider.config.name}...", file=sys.stderr)
        print(f"Model: {temp_provider.model}", file=sys.stderr)
        print(f"Temperature: {temp_provider.temperature}", file=sys.stderr)
        print(f"Chunking strategy: {temp_provider.chunking_strategy.value}", file=sys.stderr)
        print("", file=sys.stderr)  # Empty line
        
        summary = await summarizer.summarize(
            video_input=args.input,
            provider=args.provider,
            model=args.model,
            chunking_strategy=args.chunking,
            temperature=args.temperature
        )
        
        # Output the summary
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to: {args.output}", file=sys.stderr)
        else:
            print(summary)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cli_entry_point():
    """Entry point for the simplified CLI"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_point()
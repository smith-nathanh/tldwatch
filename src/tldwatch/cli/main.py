#!/usr/bin/env python3
"""
Command-line interface for tldwatch using the unified provider system.
"""

import argparse
import asyncio
import logging
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..core.providers.unified_provider import UnifiedProvider
from ..core.proxy_config import (
    ProxyConfigError,
    create_generic_proxy,
    create_webshare_proxy,
)
from ..core.summarizer import Summarizer
from ..core.user_config import get_user_config

# Initialize rich console for pretty output
console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_proxy_config(args: argparse.Namespace):
    """Create proxy configuration from CLI arguments"""
    # Check for Webshare configuration
    webshare_username = args.webshare_username or os.environ.get(
        "WEBSHARE_PROXY_USERNAME"
    )
    webshare_password = args.webshare_password or os.environ.get(
        "WEBSHARE_PROXY_PASSWORD"
    )

    if webshare_username and webshare_password:
        try:
            return create_webshare_proxy(
                proxy_username=webshare_username, proxy_password=webshare_password
            )
        except ProxyConfigError as e:
            console.print(f"[red]Webshare proxy configuration error: {str(e)}[/red]")
            return None

    # Check for generic proxy configuration
    http_proxy = args.http_proxy or os.environ.get("HTTP_PROXY_URL")
    https_proxy = args.https_proxy or os.environ.get("HTTPS_PROXY_URL")

    if http_proxy or https_proxy:
        try:
            return create_generic_proxy(http_url=http_proxy, https_url=https_proxy)
        except ProxyConfigError as e:
            console.print(f"[red]Generic proxy configuration error: {str(e)}[/red]")
            return None

    return None


async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="YouTube video summarizer using various LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  tldwatch "https://youtube.com/watch?v=dQw4w9WgXcQ"
  
  # Use a specific provider
  tldwatch "dQw4w9WgXcQ" --provider anthropic
  
  # Use a specific model and chunking strategy
  tldwatch "video_id" --provider openai --model gpt-4o --chunking large
  
  # Submit entire transcript without chunking
  tldwatch "video_url" --chunking none
  
  # Summarize direct text
  tldwatch "Your text here..." --chunking none

Configuration:
  # Create user configuration file
  tldwatch --create-config
  
  # Show current configuration
  tldwatch --show-config
  
  # List available options
  tldwatch --list-providers

User configuration file: ~/.config/tldwatch/config.json (or .yaml)
Available providers: openai, anthropic, google, groq, deepseek, cerebras, ollama
Available chunking strategies: none, standard, small, large
        """,
    )

    # Input argument (optional for info commands)
    parser.add_argument(
        "input", nargs="?", help="YouTube URL, video ID, or direct text to summarize"
    )

    # Optional arguments
    parser.add_argument(
        "--provider",
        "-p",
        choices=Summarizer.list_providers(),
        help="LLM provider to use (uses user config default or 'openai' if not specified)",
    )

    parser.add_argument(
        "--model",
        "-m",
        help="Specific model to use (uses provider default if not specified)",
    )

    parser.add_argument(
        "--chunking",
        "-c",
        choices=Summarizer.list_chunking_strategies(),
        help="Chunking strategy for long texts (uses user config default or 'standard' if not specified)",
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Generation temperature 0.0-1.0 (uses user config default or 0.7 if not specified)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for this request (even if enabled in config)",
    )

    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration even if cached summary exists",
    )

    parser.add_argument(
        "--output", "-o", help="Output file (prints to stdout if not specified)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit",
    )

    parser.add_argument(
        "--show-defaults",
        action="store_true",
        help="Show default models for each provider and exit",
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create an example user configuration file and exit",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current user configuration and exit",
    )

    # Proxy configuration
    proxy_group = parser.add_argument_group("proxy options")
    proxy_group.add_argument(
        "--webshare-username",
        type=str,
        help="Webshare proxy username (can also use WEBSHARE_PROXY_USERNAME env var)",
    )
    proxy_group.add_argument(
        "--webshare-password",
        type=str,
        help="Webshare proxy password (can also use WEBSHARE_PROXY_PASSWORD env var)",
    )
    proxy_group.add_argument(
        "--http-proxy",
        type=str,
        help="HTTP proxy URL (e.g., http://user:pass@proxy.example.com:8080)",
    )
    proxy_group.add_argument(
        "--https-proxy",
        type=str,
        help="HTTPS proxy URL (e.g., https://user:pass@proxy.example.com:8080)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Handle info commands
    if args.list_providers:
        console.print(Panel("[bold]Available providers:[/bold]", border_style="blue"))
        for provider in Summarizer.list_providers():
            default_model = Summarizer.get_default_model(provider)
            console.print(
                f"  [cyan]{provider}[/cyan] (default model: [green]{default_model}[/green])"
            )
        return

    if args.show_defaults:
        console.print(
            Panel("[bold]Default models for each provider:[/bold]", border_style="blue")
        )
        for provider in Summarizer.list_providers():
            default_model = Summarizer.get_default_model(provider)
            console.print(f"  [cyan]{provider}[/cyan]: [green]{default_model}[/green]")
        return

    if args.create_config:
        user_config = get_user_config()
        config_path = user_config.create_example_config()
        console.print(
            f"[green]Created example configuration file at:[/green] {config_path}"
        )
        console.print("[yellow]Edit this file to customize your defaults.[/yellow]")
        return

    if args.show_config:
        user_config = get_user_config()
        if user_config.has_config():
            config_path = user_config.get_config_path()
            console.print(
                Panel(
                    f"[bold]User configuration loaded from:[/bold] {config_path}",
                    border_style="blue",
                )
            )
            console.print(
                f"Default provider: [cyan]{user_config.get_default_provider() or 'Not set (will use openai)'}[/cyan]"
            )
            console.print(
                f"Default temperature: [cyan]{user_config.get_default_temperature() or 'Not set (will use 0.7)'}[/cyan]"
            )
            console.print(
                f"Default chunking strategy: [cyan]{user_config.get_default_chunking_strategy() or 'Not set (will use standard)'}[/cyan]"
            )

            console.print("\n[bold]Provider-specific settings:[/bold]")
            for provider in Summarizer.list_providers():
                model = user_config.get_default_model(provider)
                temp = user_config.get_default_temperature(provider)
                if model or temp:
                    console.print(f"  [cyan]{provider}[/cyan]:")
                    if model:
                        console.print(f"    model: [green]{model}[/green]")
                    if temp:
                        console.print(f"    temperature: [green]{temp}[/green]")
        else:
            console.print("[yellow]No user configuration file found.[/yellow]")
            console.print(
                "[yellow]Use --create-config to create an example configuration.[/yellow]"
            )
        return

    # Check if input is required
    if args.input is None:
        console.print("[red]Error: Input is required for summarization[/red]")
        console.print("[yellow]Use --help to see available options[/yellow]")
        sys.exit(1)

    # Validate temperature
    if args.temperature is not None and not 0.0 <= args.temperature <= 1.0:
        console.print("[red]Error: Temperature must be between 0.0 and 1.0[/red]")
        sys.exit(1)

    try:
        # Create proxy configuration
        proxy_config = create_proxy_config(args)
        if proxy_config:
            console.print(f"[green]Using proxy configuration: {proxy_config}[/green]")

        # Create a temporary provider to get the actual values that will be used
        temp_provider = UnifiedProvider(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            chunking_strategy=args.chunking,
        )

        # Display configuration
        console.print(
            Panel(
                f"[bold]Provider:[/bold] [cyan]{temp_provider.config.name}[/cyan]\n"
                f"[bold]Model:[/bold] [green]{temp_provider.model}[/green]\n"
                f"[bold]Temperature:[/bold] {temp_provider.temperature}\n"
                f"[bold]Chunking strategy:[/bold] {temp_provider.chunking_strategy.value}",
                title="Configuration",
                border_style="blue",
            )
        )

        # Create summarizer and generate summary
        summarizer = Summarizer()

        # Determine cache settings
        use_cache = None  # Use config default
        if args.no_cache:
            use_cache = False
        elif args.force_regenerate:
            # Clear cache for this video first (both summary and transcript), then allow caching
            from ..utils.cache import clear_cache
            from ..utils.url_parser import extract_video_id, is_youtube_url

            user_config = get_user_config()
            video_id = None
            if is_youtube_url(args.input):
                video_id = extract_video_id(args.input)
            elif args.input and len(args.input) == 11:
                video_id = args.input

            if video_id:
                clear_cache(video_id=video_id, cache_dir=user_config.get_cache_dir())
                console.print(
                    f"[yellow]Cleared all cache (summary and transcript) for video {video_id}[/yellow]"
                )

        # Generate summary with progress indication
        with console.status(
            "[bold green]Generating summary...[/bold green]", spinner="dots"
        ):
            summary = await summarizer.summarize(
                video_input=args.input,
                provider=args.provider,
                model=args.model,
                chunking_strategy=args.chunking,
                temperature=args.temperature,
                use_cache=use_cache,
            )

        # Output the summary
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(summary)
            console.print(f"[green]Summary saved to:[/green] {args.output}")
        else:
            console.print(
                Panel(
                    Text(summary), title="Summary", border_style="green", expand=False
                )
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


def cli_entry():
    """Entry point for the CLI"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli_entry()

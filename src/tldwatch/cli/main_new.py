"""
Command-line interface for tldwatch with improved chunking options.
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

from rich.console import Console

from ..core.chunking import ChunkingConfig, ChunkingStrategy
from ..core.config import Config
from ..core.summarizer_new import Summarizer, SummarizerError
from ..core.proxy_config import create_webshare_proxy, create_generic_proxy, ProxyConfigError

# Initialize rich console for pretty output
console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description="Get quick summaries of YouTube videos", prog="tldwatch"
    )

    # Input options - mutually exclusive group for video_id/url/stdin
    input_group = parser.add_mutually_exclusive_group()  # No longer required by default
    input_group.add_argument("--video-id", help="YouTube video ID")
    input_group.add_argument("url", nargs="?", help="YouTube video URL")
    input_group.add_argument(
        "--stdin", action="store_true", help="Read input from stdin"
    )

    # Output options
    parser.add_argument("--out", type=str, help="Output file path must be json file")

    # Provider configuration
    parser.add_argument(
        "--provider",
        type=str,
        choices=[
            "openai",
            "groq",
            "anthropic",
            "google",
            "cerebras",
            "deepseek",
            "ollama",
        ],
        help="LLM provider to use (defaults to config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use with the provider (defaults to provider default)",
    )

    # Chunking options
    chunking_group = parser.add_argument_group("chunking options")
    chunking_group.add_argument(
        "--chunk-size", type=int, help="Size of text chunks for processing"
    )
    chunking_group.add_argument(
        "--chunk-overlap", type=int, help="Overlap between chunks"
    )
    chunking_group.add_argument(
        "--chunking-strategy",
        type=str,
        choices=[s.value for s in ChunkingStrategy],
        help="Chunking strategy to use",
    )
    chunking_group.add_argument(
        "--interactive-chunking",
        action="store_true",
        help="Interactively select chunking strategy based on transcript length",
    )

    # Additional options
    parser.add_argument(
        "--temperature", type=float, help="Temperature for generation (0.0 to 1.0)"
    )
    parser.add_argument(
        "--full-context", action="store_true", help="Use model's full context window"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save current settings as default configuration",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current configuration settings and config file location",
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

    return parser


def create_proxy_config(args: argparse.Namespace):
    """Create proxy configuration from CLI arguments"""
    # Check for Webshare configuration
    webshare_username = args.webshare_username or os.environ.get("WEBSHARE_PROXY_USERNAME")
    webshare_password = args.webshare_password or os.environ.get("WEBSHARE_PROXY_PASSWORD")
    
    if webshare_username and webshare_password:
        try:
            return create_webshare_proxy(
                proxy_username=webshare_username,
                proxy_password=webshare_password
            )
        except ProxyConfigError as e:
            console.print(f"[red]Webshare proxy configuration error: {str(e)}[/red]")
            return None
    
    # Check for generic proxy configuration
    http_proxy = args.http_proxy or os.environ.get("HTTP_PROXY_URL")
    https_proxy = args.https_proxy or os.environ.get("HTTPS_PROXY_URL")
    
    if http_proxy or https_proxy:
        try:
            return create_generic_proxy(
                http_url=http_proxy,
                https_url=https_proxy
            )
        except ProxyConfigError as e:
            console.print(f"[red]Generic proxy configuration error: {str(e)}[/red]")
            return None
    
    return None


def create_chunking_config(args: argparse.Namespace, config: Config) -> Optional[ChunkingConfig]:
    """Create chunking configuration from CLI arguments and config"""
    # Check if any chunking options are provided
    if not (args.chunk_size or args.chunk_overlap or args.chunking_strategy):
        return None  # Let the summarizer determine the default

    # Get values from args or config
    chunk_size = args.chunk_size or config.get("chunk_size", 4000)
    chunk_overlap = args.chunk_overlap or config.get("chunk_overlap", 200)
    
    # Get strategy
    strategy_value = args.chunking_strategy or config.get("chunking_strategy", "standard")
    try:
        strategy = ChunkingStrategy(strategy_value)
    except ValueError:
        console.print(f"[yellow]Warning: Invalid chunking strategy '{strategy_value}', using 'standard'[/yellow]")
        strategy = ChunkingStrategy.STANDARD
    
    return ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )


async def run_summarizer(
    video_id: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    chunking_config: Optional[ChunkingConfig] = None,
    temperature: Optional[float] = None,
    use_full_context: bool = False,
    interactive: bool = False,
    proxy_config = None,
) -> Summarizer:
    """Run the summarizer with progress indication"""
    config = Config.load()

    # Override config with CLI arguments
    provider = provider or config.get("provider", "openai")
    model = model or config.get("model")
    temperature = temperature or config.get("temperature", 0.7)

    # Print provider and model info as a persistent message
    console.print()  # Add a blank line for spacing
    console.print(f"Provider: {provider}")
    console.print(f"Model: {model}")
    console.print()  # Add a blank line for spacing

    summarizer = Summarizer(
        provider=provider,
        model=model,
        temperature=temperature,
        chunking_config=chunking_config,
        use_full_context=use_full_context,
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
        proxy_config=proxy_config,
        interactive=interactive,
    )

    # Use live display instead of Progress
    with console.status("Generating summary...", spinner="dots"):
        try:
            await summarizer.get_summary(video_id=video_id)
            return summarizer
        except SummarizerError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            sys.exit(1)


def save_config(args: argparse.Namespace) -> None:
    """Save current settings to config file"""
    config = Config.load()

    # Update config with CLI arguments
    if args.provider:
        config["provider"] = args.provider
    if args.model:
        config["model"] = args.model
    if args.chunk_size:
        config["chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        config["chunk_overlap"] = args.chunk_overlap
    if args.chunking_strategy:
        config["chunking_strategy"] = args.chunking_strategy
    if args.temperature:
        config["temperature"] = args.temperature

    config.save()
    console.print("[green]Configuration saved successfully[/green]")


def check_environment(args) -> None:
    """Check for required environment variables"""
    required_vars = {
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }

    config = Config.load()
    provider = args.provider if args.provider else config.get("provider", "openai")

    if env_var := required_vars.get(provider):
        if not os.environ.get(env_var):
            console.print(f"[yellow]Warning: {env_var} not set in environment[/yellow]")


async def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Check environment variables
    check_environment(args)

    if args.print_config:
        config = Config.load()
        console.print("[bold]Current Configuration:[/bold]")
        console.print(config)
        console.print(f"[bold]Config file location:[/bold] {Config.get_config_path()}")
        sys.exit(0)

    # Check output is json
    if args.out and not args.out.endswith(".json"):
        console.print("[red]Error: Output file must be a JSON file[/red]")
        sys.exit(1)

    # Handle save config first
    if args.save_config:
        save_config(args)
        if not args.video_id and not args.url and not args.stdin:
            return

    # Validate input requirements if not just saving config
    if not args.save_config and not (args.video_id or args.url or args.stdin):
        parser.error("one of the arguments --video-id url --stdin is required")

    # Initialize summarizer early for input validation
    config = Config.load()
    provider = args.provider or config.get("provider", "openai")
    model = args.model or config.get("model")
    temperature = args.temperature or config.get("temperature", 0.7)
    
    # Create chunking configuration
    chunking_config = create_chunking_config(args, config)

    # Create proxy configuration
    proxy_config = create_proxy_config(args) or config.proxy_config
    if proxy_config:
        console.print(f"[green]Using proxy configuration: {proxy_config}[/green]")

    summarizer = Summarizer(
        provider=provider,
        model=model,
        temperature=temperature,
        chunking_config=chunking_config,
        use_full_context=args.full_context,
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
        proxy_config=proxy_config,
        interactive=args.interactive_chunking,
    )

    try:
        # Get stdin content if needed
        stdin_content = None
        if args.stdin:
            if sys.stdin.isatty():
                console.print("[red]Error: No input provided on stdin[/red]")
                sys.exit(1)
            stdin_content = sys.stdin.read().strip()

        # Validate input using Summarizer's method
        try:
            video_id = summarizer.validate_input(
                video_id=args.video_id, url=args.url, stdin_content=stdin_content
            )
        except SummarizerError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            sys.exit(1)

        # Generate summary with progress indication
        with console.status("Generating summary...", spinner="dots"):
            try:
                await summarizer.get_summary(video_id=video_id)
            except SummarizerError as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                sys.exit(1)

        # Handle output
        if args.out:
            await summarizer.export_summary(args.out)
            console.print(f"[green]Summary saved to {args.out}[/green]")
        else:
            console.print(summarizer.summary)
            console.print()  # Add a blank line for spacing

    finally:
        await summarizer.close()

    sys.exit(0)


def cli_entry() -> None:
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
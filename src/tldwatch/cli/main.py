import argparse
import asyncio
import os
import sys
from typing import Optional, Tuple

from rich.console import Console

from ..core.config import Config
from ..core.summarizer import Summarizer, SummarizerError
from ..utils.url_parser import extract_video_id, is_youtube_url

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
        choices=["openai", "groq", "anthropic", "cerebras", "deepseek", "ollama"],
        help="LLM provider to use (defaults to config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use with the provider (defaults to provider default)",
    )

    # Additional options
    parser.add_argument(
        "--chunk-size", type=int, help="Size of text chunks for processing"
    )
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

    return parser


def get_input_source(args: argparse.Namespace) -> str:
    """Determine and validate the input source"""
    if args.video_id:
        return args.video_id
    elif args.url:
        if not is_youtube_url(args.url):
            console.print("[red]Error: Invalid YouTube URL[/red]")
            sys.exit(1)
        video_id = extract_video_id(args.url)
        if not video_id:
            console.print("[red]Error: Could not extract video ID from URL[/red]")
            sys.exit(1)
        return video_id
    elif args.stdin:
        # Read from stdin
        if sys.stdin.isatty():
            console.print("[red]Error: No input provided on stdin[/red]")
            sys.exit(1)
        content = sys.stdin.read().strip()
        if is_youtube_url(content):
            return extract_video_id(content) or ""
        return content

    return ""


async def run_summarizer(
    video_id: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    temperature: Optional[float] = None,
    use_full_context: bool = False,
) -> Tuple[str, Optional[Summarizer]]:
    """Run the summarizer with progress indication"""
    config = Config.load()

    # Override config with CLI arguments
    provider = provider or config.get("provider", "openai")
    model = model or config.get("model")
    chunk_size = chunk_size or config.get("chunk_size", 4000)
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
        chunk_size=chunk_size,
        use_full_context=use_full_context,
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
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
    if args.temperature:
        config["temperature"] = args.temperature

    config.save()
    console.print("[green]Configuration saved successfully[/green]")


def check_environment(args) -> None:
    """Check for required environment variables"""
    required_vars = {
        "openai": "OPENAI_API_KEY",
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

    # Get input source
    video_id = get_input_source(args)

    # Run summarizer
    summarizer = await run_summarizer(
        video_id,
        provider=args.provider,
        model=args.model,
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        use_full_context=args.full_context,
    )

    if args.out:
        await summarizer.export_summary(args.out)
        console.print(f"[green]Summary saved to {args.out}[/green]")
    else:
        # Write to stdout
        console.print(summarizer.summary)
        console.print()  # Add a blank line for spacing
    sys.exit(0)  # Explicitly exit after completion


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

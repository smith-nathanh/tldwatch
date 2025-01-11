import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video-id", help="YouTube video ID")
    input_group.add_argument("url", nargs="?", help="YouTube video URL")
    input_group.add_argument(
        "--stdin", action="store_true", help="Read input from stdin"
    )

    # Output options
    parser.add_argument("--out", type=str, help="Output file path (defaults to stdout)")

    # Provider configuration
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "groq", "cerebras", "ollama"],
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
) -> str:
    """Run the summarizer with progress indication"""
    config = Config.load()

    # Override config with CLI arguments
    provider = provider or config.get("provider", "openai")
    model = model or config.get("model")
    chunk_size = chunk_size or config.get("chunk_size", 4000)
    temperature = temperature or config.get("temperature", 0.7)

    summarizer = Summarizer(
        provider=provider,
        model=model,
        temperature=temperature,
        chunk_size=chunk_size,
        use_full_context=use_full_context,
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="Generating summary...", total=None)
        try:
            return await summarizer.get_summary(video_id=video_id)
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


def write_output(summary: str, output_file: Optional[str]) -> None:
    """Write summary to specified output or stdout"""
    if output_file:
        output_path = Path(output_file)

        # Check if file exists
        if output_path.exists() and not Confirm.ask(
            f"File {output_file} exists. Overwrite?"
        ):
            return

        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the summary
        with open(output_path, "w") as f:
            if output_file.endswith(".json"):
                json.dump({"summary": summary}, f, indent=2)
            else:
                f.write(summary)
        console.print(f"[green]Summary saved to {output_file}[/green]")
    else:
        # Write to stdout
        console.print(summary)


def check_environment() -> None:
    """Check for required environment variables"""
    required_vars = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }

    config = Config.load()
    provider = config.get("provider", "openai")

    if env_var := required_vars.get(provider):
        if not os.environ.get(env_var):
            console.print(f"[yellow]Warning: {env_var} not set in environment[/yellow]")


async def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Check environment variables
    check_environment()

    # Save config if requested
    if args.save_config:
        save_config(args)
        if not args.video_id and not args.url and not args.stdin:
            return

    # Get input source
    video_id = get_input_source(args)

    # Run summarizer
    summary = await run_summarizer(
        video_id,
        provider=args.provider,
        model=args.model,
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        use_full_context=args.full_context,
    )

    # Write output
    write_output(summary, args.out)


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

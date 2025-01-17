#!/usr/bin/env python3
"""
Interactive demo script for tldwatch CLI usage.
This script demonstrates various CLI features with real examples.
"""

import os
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

console = Console()


def get_tldwatch_command() -> str:
    """
    Get the appropriate command to run tldwatch.
    Returns installed 'tldwatch' if available, otherwise uses development path.
    """
    # Check if tldwatch is installed
    if shutil.which("tldwatch"):
        return "tldwatch"

    # Fall back to development version
    return "python -m tldwatch.cli.main"


def run_command(command: str, description: str = None) -> None:
    """Run a command and display its output"""
    if description:
        console.print(f"\n[bold blue]{description}[/bold blue]")

    # Replace 'tldwatch' with appropriate command
    tldwatch_cmd = get_tldwatch_command()
    command = command.replace("tldwatch", tldwatch_cmd)

    console.print(f"[dim]$ {command}[/dim]")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            env=os.environ,
        )
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(f"[yellow]{result.stderr}[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error: Command failed with exit code {e.returncode}[/red]")
        if e.stdout:
            console.print(e.stdout)
        if e.stderr:
            console.print(f"[red]{e.stderr}[/red]")


def check_environment() -> bool:
    """Check if required environment variables are set"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI",
        "GROQ_API_KEY": "Groq",
        "CEREBRAS_API_KEY": "Cerebras",
        "DEEPSEEK_API_KEY": "DeepSeek",
    }

    missing = [name for var, name in required_vars.items() if not os.environ.get(var)]

    if missing:
        console.print(
            f"[yellow]Warning: API keys missing for: {', '.join(missing)}[/yellow]"
        )
        console.print("Some examples may not work without the required API keys.")
        console.print("Make sure to set the environment variables by:")
        console.print("1. Sourcing your .env file: source .env")
        console.print("2. Or exporting directly: export OPENAI_API_KEY=your-key-here")
        return False
    return True


def main():
    console.print("[bold]tldwatch CLI Demo[/bold]")
    console.print("This script will demonstrate various CLI features.")

    # Check environment
    check_environment()

    # Create output directory
    output_dir = Path("tldwatch_examples")
    output_dir.mkdir(exist_ok=True)

    # Basic usage
    if Confirm.ask("\nDemonstrate basic usage?"):
        run_command(
            "tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc",
            "Basic usage with YouTube URL",
        )

        run_command("tldwatch --video-id QAgR4uQ15rc", "Using video ID directly")

    # Output options
    if Confirm.ask("\nDemonstrate output options?"):
        run_command(
            f"tldwatch --video-id QAgR4uQ15rc --out {output_dir}/summary.txt",
            "Saving summary to text file",
        )

        run_command(
            f"tldwatch --video-id QAgR4uQ15rc --out {output_dir}/summary.json",
            "Saving summary as JSON",
        )

    # Provider selection
    if Confirm.ask("\nTry different providers?"):
        providers = {
            "openai": "gpt-4o",
            "groq": "mixtral-8x7b-32768",
            "ollama": "mistral",
        }

        provider = Prompt.ask(
            "Which provider?", choices=list(providers.keys()), default="openai"
        )

        run_command(
            f"tldwatch --video-id QAgR4uQ15rc --provider {provider} "
            f"--model {providers[provider]}",
            f"Using {provider.title()} provider",
        )

    # Advanced options
    if Confirm.ask("\nTry advanced options?"):
        run_command(
            "tldwatch --video-id QAgR4uQ15rc --full-context "
            "--chunk-size 6000 --temperature 0.8",
            "Using advanced options",
        )

    # Configuration
    if Confirm.ask("\nDemonstrate configuration?"):
        run_command(
            "tldwatch --save-config --provider openai --model gpt-4",
            "Saving default configuration",
        )

    console.print("\n[bold green]Demo completed![/bold green]")
    console.print(f"Example outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()

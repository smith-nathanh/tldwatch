# TLDWatch Examples

This directory contains focused examples showing how to use TLDWatch.

## Files

### `quickstart.py`
The fastest way to get started with TLDWatch. Run this first to verify your setup.

### `simple_example.py`
Core functionality examples showing:
- Basic video summarization
- Different input methods (URL, video ID, direct text)
- Provider selection
- Parameter customization

### `config_usage.py`
Configuration management examples:
- Creating and using config files
- Understanding configuration precedence
- Provider-specific settings

### `cli_examples.txt`
Command-line interface examples with common CLI commands.

## Setup

Before running examples, set up your API keys:

```bash
# Option 1: OpenAI
export OPENAI_API_KEY="your-openai-key"

# Option 2: Groq (free tier available)
export GROQ_API_KEY="your-groq-key"

# Option 3: Other providers
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

## Running Examples

```bash
# Start with quickstart
python quickstart.py

# Try the main example
python simple_example.py

# Learn about configuration
python config_usage.py
```

## CLI Usage

```bash
# Basic usage
tldwatch "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# With options
tldwatch "dQw4w9WgXcQ" --provider groq --output summary.txt

# See all options
tldwatch --help
```

# TLDWatch Simplified Interface

This document describes the new, greatly simplified interface for TLDWatch that consolidates all provider implementations into a unified system.

## Overview

The new interface eliminates the complexity of managing different providers, rate limits, context windows, and chunking strategies. Instead, it provides a clean, simple API that "just works" for most use cases.

## Key Simplifications

### 1. Unified Provider System
- **Before**: Separate classes for each provider with custom retry logic, rate limiting, and request handling
- **After**: Single `UnifiedProvider` class that handles all providers with consistent behavior

### 2. Simplified Configuration
- **Before**: Complex `config.yaml` with rate limits, context windows, and model-specific settings
- **After**: Simple config with just default models for each provider

### 3. Generic Chunking
- **Before**: Complex chunking strategies based on model context windows
- **After**: Simple, generic chunking that works for any model

### 4. Clean Interface
- **Before**: Many required parameters and complex initialization
- **After**: Minimal required parameters with sensible defaults

## Quick Start

### Installation
```bash
pip install tldwatch
```

### Basic Usage

```python
import asyncio
from tldwatch import summarize_video

# Simplest possible usage - uses your configured defaults
summary = await summarize_video("https://youtube.com/watch?v=dQw4w9WgXcQ")
print(summary)
```

### User Configuration

TLDWatch supports user configuration files to set your preferred defaults:

```bash
# Create an example configuration file
tldwatch-simple --create-config

# View your current configuration
tldwatch-simple --show-config
```

Configuration file location: `~/.config/tldwatch/config.json` (or `.yaml`)

Example configuration:
```json
{
  "default_provider": "anthropic",
  "default_temperature": 0.3,
  "default_chunking_strategy": "large",
  "providers": {
    "openai": {
      "default_model": "gpt-4o",
      "temperature": 0.8
    },
    "anthropic": {
      "default_model": "claude-3-5-sonnet-20241022",
      "temperature": 0.2
    }
  }
}
```

With this configuration:
- `summarize_video("video_id")` uses Anthropic Claude with temperature 0.2
- `summarize_video("video_id", provider="openai")` uses OpenAI GPT-4o with temperature 0.8
- You can still override any setting: `summarize_video("video_id", temperature=0.9)`

### With Options

```python
from tldwatch import SimpleSummarizer

summarizer = SimpleSummarizer()

summary = await summarizer.summarize(
    "dQw4w9WgXcQ",  # YouTube URL, video ID, or direct text
    provider="anthropic",  # openai, anthropic, google, groq, deepseek, cerebras, ollama
    model="claude-3-5-sonnet-20241022",  # optional - uses provider default if not specified
    chunking_strategy="standard",  # none, standard, small, large
    temperature=0.7
)
```

## CLI Usage

### Simple CLI
```bash
# Basic usage (uses your configured defaults)
tldwatch-simple "https://youtube.com/watch?v=dQw4w9WgXcQ"

# With options
tldwatch-simple "video_id" --provider anthropic --chunking large

# Submit entire transcript without chunking
tldwatch-simple "video_url" --chunking none

# Configuration management
tldwatch-simple --create-config    # Create example config file
tldwatch-simple --show-config      # Show current configuration
tldwatch-simple --list-providers   # List available providers
tldwatch-simple --show-defaults    # Show default models
```

## API Reference

### SimpleSummarizer

The main class for the simplified interface.

```python
class SimpleSummarizer:
    async def summarize(
        self,
        video_input: str,  # YouTube URL, video ID, or direct text
        provider: Optional[str] = None,  # Provider (uses user config default)
        model: Optional[str] = None,  # Model (uses user/provider default)
        chunking_strategy: Optional[str] = None,  # Chunking (uses user config default)
        temperature: Optional[float] = None  # Temperature (uses user config default)
    ) -> str:
        """Generate a summary"""
```

### Convenience Function

```python
async def summarize_video(
    video_input: str,
    provider: Optional[str] = None,  # Uses user config default
    model: Optional[str] = None,  # Uses user/provider default
    chunking_strategy: Optional[str] = None,  # Uses user config default
    temperature: Optional[float] = None  # Uses user config default
) -> str:
    """Quick summarization function"""
```

## Providers

### Supported Providers
- **openai**: OpenAI GPT models (default: gpt-4o-mini)
- **anthropic**: Anthropic Claude models (default: claude-3-5-sonnet-20241022)
- **google**: Google Gemini models (default: gemini-1.5-flash)
- **groq**: Groq models (default: llama-3.1-8b-instant)
- **deepseek**: DeepSeek models (default: deepseek-chat)
- **cerebras**: Cerebras models (default: llama3.1-8b)
- **ollama**: Local Ollama models (default: llama3.1:8b)

### Environment Variables
Set the appropriate API key for your chosen provider:
- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic
- `GEMINI_API_KEY` for Google
- `GROQ_API_KEY` for Groq
- `DEEPSEEK_API_KEY` for DeepSeek
- `CEREBRAS_API_KEY` for Cerebras
- No API key needed for Ollama (local)

## Chunking Strategies

### Available Strategies
- **none**: Submit entire transcript without chunking
- **standard**: Balanced approach with ~4000 character chunks (default)
- **small**: Smaller ~2000 character chunks for detailed processing
- **large**: Larger ~8000 character chunks for context preservation

### How Chunking Works
1. Text is split at sentence boundaries when possible
2. If multiple chunks are created, each is summarized separately
3. If the combined summaries are still long, they're summarized again
4. Simple rate limiting (1 second between requests) prevents API issues

## Migration from Legacy Interface

### Old Way
```python
from tldwatch import Summarizer
from tldwatch.core.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o-mini",
    temperature=0.7,
    rate_limit_config=RateLimitConfig(...),
    use_full_context=False
)

summarizer = Summarizer(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    chunk_size=4000,
    chunk_overlap=200,
    use_full_context=False
)

summary = await summarizer.summarize_video("video_id")
```

### New Way
```python
from tldwatch import summarize_video

# That's it!
summary = await summarize_video("video_id")

# Or with options:
summary = await summarize_video(
    "video_id",
    provider="openai",
    model="gpt-4o-mini",
    chunking_strategy="standard",
    temperature=0.7
)
```

## Examples

See `examples/simple_usage.py` for comprehensive examples of the new interface.

## Benefits

1. **Simplicity**: Minimal configuration required
2. **Consistency**: All providers work the same way
3. **Reliability**: Generic retry logic and rate limiting
4. **Flexibility**: Easy to switch between providers and models
5. **Maintainability**: Much less code to maintain and debug

## Backward Compatibility

The legacy `Summarizer` class is still available for backward compatibility, but new projects should use the simplified interface.

## Configuration File

The new `config.yaml` is much simpler:

```yaml
# Simplified provider configuration
providers:
  anthropic:
    default_model: "claude-3-5-sonnet-20241022"
  google:
    default_model: "gemini-1.5-flash"
  openai:
    default_model: "gpt-4o-mini"
  # ... etc
```

No more complex rate limits, context windows, or model-specific configurations to manage!
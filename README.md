# tldwatch

A Python library for generating summaries of YouTube video transcripts (or any text) using the LLM provider of your choice.

## Features

- Generate summaries from YouTube videos using video IDs or URLs
- Process any text by passing it directly
- Support for multiple AI providers:
  - OpenAI
  - Google
  - Anthropic
  - Groq
  - Cerebras
  - DeepSeek
  - Ollama (local models)
- Simple command-line interface with rich text formatting
- Clean Python API for programmatic usage
- Smart chunking strategies for long transcripts
- Proxy support to avoid IP blocking (Webshare integration)

## Installation

```bash
pip install tldwatch
```

## Quick Start

### Command Line Usage

```bash
# Summarize using YouTube URL
tldwatch "https://www.youtube.com/watch?v=QAgR4uQ15rc"

# Using video ID directly
tldwatch "QAgR4uQ15rc"

# Process direct text
tldwatch "Your text to summarize..."

# Save summary to file
tldwatch "QAgR4uQ15rc" --output summary.txt

# Use specific provider/model
tldwatch "QAgR4uQ15rc" --provider groq --model mixtral-8x7b-32768

# Use specific chunking strategy
tldwatch "QAgR4uQ15rc" --chunking large
```

### Python Library Usage

```python
import asyncio
from tldwatch import Summarizer, summarize_video

async def main():
    # Quick usage with convenience function
    summary = await summarize_video("https://www.youtube.com/watch?v=QAgR4uQ15rc")
    print(summary)
    
    # More control with Summarizer class
    summarizer = Summarizer()
    
    # From YouTube URL
    summary = await summarizer.summarize("https://www.youtube.com/watch?v=QAgR4uQ15rc")
    
    # From video ID with specific provider and model
    summary = await summarizer.summarize(
        "QAgR4uQ15rc",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        chunking_strategy="large",
        temperature=0.5
    )
    
    # From direct text
    summary = await summarizer.summarize("Your text to summarize...")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Environment Variables

Set up your API keys:

```bash
# Required for respective providers
export OPENAI_API_KEY="your-key-here"
export GROQ_API_KEY="your-key-here"
export CEREBRAS_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Optional, for YouTube metadata enrichment
export YOUTUBE_API_KEY="your-key-here"
```

### User Configuration

tldwatch uses a simple user configuration system that automatically loads your saved settings.

#### Configuration Location
- Linux/Mac: `~/.config/tldwatch/config.json` or `~/.config/tldwatch/config.yaml`
- Or uses `XDG_CONFIG_HOME` if set

#### Creating and Viewing Configuration

```bash
# Create example configuration file
tldwatch --create-config

# View current configuration
tldwatch --show-config

# List available providers
tldwatch --list-providers

# Show default models
tldwatch --show-defaults
```

#### Configuration Precedence
Settings are applied in this order (highest to lowest priority):
1. Command line arguments / function parameters
2. User's saved config file
3. Built-in defaults

#### Provider Configuration

Each provider has default models, but you can specify alternatives:

```python
# OpenAI
summarizer = Summarizer()
summary = await summarizer.summarize("video_id", provider="openai", model="gpt-4o")

# Anthropic
summary = await summarizer.summarize("video_id", provider="anthropic", model="claude-3-5-sonnet-20241022")

# Groq
summary = await summarizer.summarize("video_id", provider="groq", model="mixtral-8x7b-32768")

# Ollama (local)
summary = await summarizer.summarize("video_id", provider="ollama", model="llama3.1:8b")
```

### Chunking Strategies

For long transcripts, tldwatch provides different chunking strategies:

```python
# Available strategies
from tldwatch import ChunkingStrategy

# Use specific strategy
summary = await summarizer.summarize("video_id", chunking_strategy="large")
summary = await summarizer.summarize("video_id", chunking_strategy=ChunkingStrategy.LARGE)

# Strategy options:
# - "none": Submit entire transcript (if it fits in context window)
# - "standard": Default balanced approach
# - "small": Smaller chunks for detailed processing
# - "large": Larger chunks for better context preservation
```

## Proxy Configuration (Avoiding IP Blocking)

YouTube may block IP addresses that make too many requests. Use proxy configuration to avoid this:

### Webshare (Recommended)

```bash
# Set up Webshare credentials
export WEBSHARE_PROXY_USERNAME="your_username"
export WEBSHARE_PROXY_PASSWORD="your_password"

# Use with CLI
tldwatch "https://www.youtube.com/watch?v=QAgR4uQ15rc"
```

```python
# Use with library
from tldwatch import Summarizer, create_webshare_proxy

proxy_config = create_webshare_proxy(
    proxy_username="your_username",
    proxy_password="your_password"
)

# The proxy configuration will be used automatically
summarizer = Summarizer()
summary = await summarizer.summarize("https://www.youtube.com/watch?v=QAgR4uQ15rc")
```

### Generic Proxies

```python
from tldwatch import create_generic_proxy

proxy_config = create_generic_proxy(
    http_url="http://user:pass@proxy.example.com:8080",
    https_url="https://user:pass@proxy.example.com:8080"
)
```

## Error Handling

```python
from tldwatch import Summarizer

try:
    summarizer = Summarizer()
    summary = await summarizer.summarize("https://www.youtube.com/watch?v=...")
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Summarization error: {e}")
    if "blocked" in str(e).lower():
        print("Consider using proxy configuration")
```

## Development

Clone and set up for development:

```bash
git clone https://github.com/yourusername/tldwatch.git
cd tldwatch
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
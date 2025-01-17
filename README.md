# tldwatch

A Python library for generating summaries of YouTube video transcripts (or any transcript) using a selection of AI providers.

## Features

- Generate summaries from YouTube videos using video IDs or URLs
- Process other video transcripts text directly
- Support for multiple AI providers:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - Groq (Mixtral, LLaMA)
  - Cerebras (Llama)
  - DeepSeek (deepseek-chat)
  - Ollama (local models)
- Command-line interface for quick summaries
- Python library for programmatic usage
- Configurable chunking for long transcripts
- Rate limiting and error handling
- Export summaries to JSON with metadata

## Installation

```bash
pip install tldwatch
```

## Quick Start

### Command Line Usage

```bash
# Summarize using YouTube URL
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc

# Using video ID directly
tldwatch --video-id QAgR4uQ15rc

# Process a local transcript
cat transcript.txt | tldwatch --stdin

# Save summary to file
tldwatch --video-id QAgR4uQ15rc --out summary.txt

# Use specific provider/model
tldwatch --video-id QAgR4uQ15rc --provider groq --model mixtral-8x7b-32768
```

### Python Library Usage

```python
import asyncio
from tldwatch import Summarizer

async def main():
    # Initialize summarizer
    summarizer = Summarizer(
        provider="openai",
        model="gpt-4o"
    )
    
    # Get summary from video ID
    summary = await summarizer.get_summary(
        video_id="QAgR4uQ15rc"
    )
    print(summary)
    
    # Or from YouTube URL
    summary = await summarizer.get_summary(
        url="https://www.youtube.com/watch?v=QAgR4uQ15rc"
    )
    
    # Or from direct transcript input
    summary = await summarizer.get_summary(
        transcript_text="Your transcript text here..."
    )
    
    # Export summary
    await summarizer.export_summary("summary.json")

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

# Optional, for YouTube metadata enrichment
export YOUTUBE_API_KEY="your-key-here"
```

### Configuration Management

tldwatch uses a persistent configuration system that automatically loads your saved settings.

#### Configuration Location
- Linux/Mac: `~/.config/tldwatch/config.json`
- Or uses `XDG_CONFIG_HOME` if set

#### Setting Configuration

You can save your preferred settings permanently:
```bash
# Save default provider and model
tldwatch --save-config --provider groq --model mixtral-8x7b-32768

# Save with additional settings
tldwatch --save-config --provider openai --model gpt-4o --temperature 0.8 --chunk-size 6000
```

#### Configuration Precedence
Settings are applied in this order (highest to lowest priority):
1. Command line arguments
2. User's saved config file
3. Built-in defaults

For example:
```bash
# Uses your saved config settings
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc

# Overrides saved config just for this run
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc --provider openai --model gpt-4o
```

#### Provider Configuration

Each provider has default models, but you can specify alternatives:

```python
# OpenAI
summarizer = Summarizer(provider="openai", model="gpt-4o")

# Groq
summarizer = Summarizer(provider="groq", model="mixtral-8x7b-32768")

# Cerebras
summarizer = Summarizer(provider="cerebras", model="llama3.1-8b")

# Ollama (local)
summarizer = Summarizer(provider="ollama", model="mistral")
```

### Advanced Configuration

```python
summarizer = Summarizer(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,           # Control output randomness
    chunk_size=4000,          # Size of text chunks
    chunk_overlap=200,        # Overlap between chunks
    use_full_context=True,    # Use model's full context window
    youtube_api_key="..."     # For metadata enrichment
)
```

## Processing Long Transcripts

For long transcripts, tldwatch automatically handles chunking:

1. If `use_full_context=True` and the transcript fits in the model's context window, it processes the entire transcript at once.
2. Otherwise, it splits the transcript into chunks with overlap, summarizes each chunk, and then combines the summaries.

## Error Handling

```python
from tldwatch import Summarizer, SummarizerError

try:
    summarizer = Summarizer()
    summary = await summarizer.get_summary(video_id="...")
except ValueError as e:
    print(f"Invalid input: {e}")
except SummarizerError as e:
    print(f"Summarization error: {e}")
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
# TLDWatch: New Architecture

## Consolidated Providers and Improved Chunking

TLDWatch has been updated with a new architecture that consolidates providers and improves chunking strategies. This update makes it easier to add new providers, simplifies the codebase, and provides more flexible chunking options for different transcript lengths.

### Key Improvements

#### 1. Consolidated Provider Architecture

The provider implementation has been completely redesigned to use a unified request handler and factory pattern. This reduces code duplication and makes it easier to add new providers.

- **Unified Request Handler**: All API requests are now handled by a single request handler class, which standardizes error handling, retries, and response parsing.
- **Provider Factory**: Providers are now instantiated through a factory, which simplifies provider selection and configuration.
- **Standardized API**: All providers now implement the same interface, making it easier to switch between providers.

#### 2. Improved Chunking Strategies

The chunking system has been redesigned to be more flexible and adaptable to different transcript lengths.

- **Multiple Chunking Strategies**: Choose from standard, small, large, adaptive, paragraph, or semantic chunking strategies.
- **Interactive Mode**: In CLI mode, you can now interactively select a chunking strategy based on the transcript length.
- **Automatic Adaptation**: The default strategy automatically adapts to the transcript length, using larger chunks for shorter transcripts and smaller chunks for longer ones.

#### 3. CLI Enhancements

The command-line interface has been enhanced with new options for chunking and better feedback.

- **Chunking Options**: Specify chunking strategy, chunk size, and overlap from the command line.
- **Interactive Mode**: Use `--interactive-chunking` to select a chunking strategy interactively.
- **Better Feedback**: More detailed progress information and error messages.

### Getting Started

To use the new architecture, update your imports to use the new modules:

```python
# Old imports
from tldwatch.core.summarizer import Summarizer
from tldwatch.core.providers.openai import OpenAIProvider

# New imports
from tldwatch.core.summarizer_new import Summarizer
from tldwatch.core.providers import ProviderFactory
from tldwatch.core.chunking import ChunkingConfig, ChunkingStrategy
```

See the [Migration Guide](MIGRATION_GUIDE.md) for detailed instructions on updating your code.

### Example Usage

#### CLI Usage

```bash
# Basic usage (same as before)
tldwatch https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Use interactive chunking
tldwatch https://www.youtube.com/watch?v=dQw4w9WgXcQ --interactive-chunking

# Specify a chunking strategy
tldwatch https://www.youtube.com/watch?v=dQw4w9WgXcQ --chunking-strategy adaptive

# Customize chunk size and overlap
tldwatch https://www.youtube.com/watch?v=dQw4w9WgXcQ --chunk-size 6000 --chunk-overlap 500
```

#### Python API Usage

```python
import asyncio
from tldwatch.core.summarizer_new import Summarizer
from tldwatch.core.chunking import ChunkingConfig, ChunkingStrategy

async def main():
    # Basic usage (same as before)
    summarizer = Summarizer(provider="openai")
    summary = await summarizer.get_summary(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(summary)
    
    # With custom chunking configuration
    chunking_config = ChunkingConfig(
        chunk_size=6000,
        chunk_overlap=500,
        strategy=ChunkingStrategy.ADAPTIVE
    )
    
    summarizer = Summarizer(
        provider="openai",
        chunking_config=chunking_config
    )
    
    summary = await summarizer.get_summary(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(summary)

if __name__ == "__main__":
    asyncio.run(main())
```

### Available Chunking Strategies

- `standard`: Default strategy with balanced chunk size
- `small`: Smaller chunks for more granular processing
- `large`: Larger chunks for better context preservation
- `adaptive`: Adapts chunk size based on transcript length
- `paragraph`: Chunks based on paragraph boundaries
- `semantic`: Chunks based on semantic boundaries (experimental)

### Adding New Providers

Adding a new provider is now much simpler. See the [Migration Guide](MIGRATION_GUIDE.md) for detailed instructions.
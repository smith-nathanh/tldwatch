# Migration Guide: Consolidated Providers and Improved Chunking

This guide will help you migrate from the previous version of tldwatch to the new version with consolidated providers and improved chunking strategies.

## Overview of Changes

1. **Consolidated Provider Architecture**
   - Unified request handling across all providers
   - Simplified provider implementation with less boilerplate
   - Factory pattern for provider instantiation
   - Standardized error handling and response parsing

2. **Improved Chunking Strategies**
   - Flexible chunking strategies based on transcript length
   - Interactive chunking strategy selection
   - Better handling of different transcript lengths
   - Improved chunk boundary detection

3. **CLI Enhancements**
   - New chunking strategy options
   - Interactive mode for selecting chunking strategy
   - Better error messages and feedback

## API Changes

### Provider Usage

#### Before:

```python
from tldwatch.core.providers.openai import OpenAIProvider

provider = OpenAIProvider(model="gpt-4", temperature=0.7)
summary = await provider.generate_summary(text)
```

#### After:

```python
from tldwatch.core.providers import ProviderFactory

provider = ProviderFactory.create_provider(
    provider_name="openai",
    model="gpt-4",
    temperature=0.7
)
summary = await provider.generate_summary(text)
```

### Chunking Configuration

#### Before:

```python
from tldwatch.core.summarizer import Summarizer

summarizer = Summarizer(
    provider="openai",
    chunk_size=4000,
    chunk_overlap=200
)
```

#### After:

```python
from tldwatch.core.chunking import ChunkingConfig, ChunkingStrategy
from tldwatch.core.summarizer_new import Summarizer

# Option 1: Let the summarizer determine the best chunking strategy
summarizer = Summarizer(provider="openai")

# Option 2: Specify a chunking configuration
chunking_config = ChunkingConfig(
    chunk_size=4000,
    chunk_overlap=200,
    strategy=ChunkingStrategy.STANDARD
)

summarizer = Summarizer(
    provider="openai",
    chunking_config=chunking_config
)

# Option 3: Use interactive mode (CLI only)
summarizer = Summarizer(
    provider="openai",
    interactive=True
)
```

## CLI Usage

### New Chunking Options

```bash
# Use interactive chunking strategy selection
tldwatch https://www.youtube.com/watch?v=dQw4w9WgXcQ --interactive-chunking

# Specify a chunking strategy
tldwatch https://www.youtube.com/watch?v=dQw4w9WgXcQ --chunking-strategy adaptive

# Customize chunk size and overlap
tldwatch https://www.youtube.com/watch?v=dQw4w9WgXcQ --chunk-size 6000 --chunk-overlap 500
```

### Available Chunking Strategies

- `standard`: Default strategy with balanced chunk size
- `small`: Smaller chunks for more granular processing
- `large`: Larger chunks for better context preservation
- `adaptive`: Adapts chunk size based on transcript length
- `paragraph`: Chunks based on paragraph boundaries
- `semantic`: Chunks based on semantic boundaries

## Implementation Details

### For Developers Adding New Providers

To add a new provider, you need to:

1. Create a new provider class that inherits from `BaseProvider`
2. Implement the required abstract methods
3. Register the provider with the factory

Example:

```python
from typing import Any, Dict, Optional

from .base_provider import BaseProvider, RateLimitConfig

class MyNewProvider(BaseProvider):
    """My new provider implementation"""
    
    API_BASE = "https://api.myprovider.com/v1"
    RESPONSE_KEY_PATH = ("response", "text")
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Initialize with default model if not specified
        if model is None:
            model = "default-model"
            
        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )
    
    def _get_provider_name(self) -> str:
        return "myprovider"
        
    def _default_rate_limit_config(self) -> RateLimitConfig:
        return RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000,
            max_retries=3,
            retry_delay=1.0,
        )
        
    def count_tokens(self, text: str) -> int:
        # Implement token counting for your provider
        return len(text) // 4  # Simple approximation
        
    @property
    def max_concurrent_requests(self) -> int:
        return 10
        
    @property
    def context_window(self) -> int:
        return 8192
        
    def _prepare_request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            **kwargs,
        }
```

Then register your provider in `provider_registry.py`:

```python
from .myprovider import MyNewProvider

def register_all_providers() -> None:
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "myprovider": MyNewProvider,  # Add your provider here
    }
    
    for name, provider_class in providers.items():
        ProviderFactory.register_provider(name, provider_class)
```

## Troubleshooting

### Common Issues

1. **Provider not found**: Make sure the provider is registered in `provider_registry.py`
2. **API key not found**: Check that the environment variable is set correctly
3. **Chunking strategy not working**: Verify that the chunking strategy is supported and properly configured

### Getting Help

If you encounter any issues with the migration, please open an issue on GitHub with the following information:

- The version of tldwatch you're using
- The code that's causing the issue
- Any error messages you're seeing
- Steps to reproduce the issue
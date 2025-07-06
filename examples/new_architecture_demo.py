"""
Demo script for the new architecture with consolidated providers and improved chunking.

This script demonstrates how to use the new provider factory and chunking strategies.
"""

import asyncio
import os
import sys
from typing import Optional

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tldwatch.core.chunking import ChunkingConfig, ChunkingStrategy
from src.tldwatch.core.providers import ProviderFactory
from src.tldwatch.core.summarizer_new import Summarizer


async def demo_provider_factory():
    """Demonstrate the provider factory"""
    print("\n=== Provider Factory Demo ===\n")
    
    # Get available providers
    from src.tldwatch.core.providers.provider_factory import ProviderFactory
    available_providers = ProviderFactory.get_available_providers()
    print(f"Available providers: {', '.join(available_providers.keys())}")
    
    # Create a provider using the factory
    provider = ProviderFactory.create_provider(
        provider_name="openai",
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    print(f"Created provider: {provider.__class__.__name__}")
    print(f"Model: {provider.model}")
    print(f"Context window: {provider.context_window} tokens")
    print(f"Max concurrent requests: {provider.max_concurrent_requests}")
    
    # Clean up
    await provider.close()


async def demo_chunking_strategies():
    """Demonstrate different chunking strategies"""
    print("\n=== Chunking Strategies Demo ===\n")
    
    # Sample text (a short transcript)
    text = """
    Welcome to this demonstration of chunking strategies. In this video, we'll explore
    how different chunking strategies affect the summarization process. Chunking is the
    process of breaking down a long text into smaller, manageable pieces that can be
    processed by language models with limited context windows.
    
    The standard chunking strategy balances chunk size and overlap to provide a good
    general-purpose approach. The small chunking strategy uses smaller chunks with less
    overlap, which can be useful for very detailed processing. The large chunking strategy
    uses larger chunks with more overlap, which can help preserve context across chunks.
    
    The adaptive chunking strategy automatically adjusts the chunk size based on the length
    of the text. For shorter texts, it uses larger chunks to preserve context. For longer
    texts, it uses smaller chunks to ensure efficient processing.
    
    The paragraph chunking strategy tries to split the text at paragraph boundaries, which
    can help preserve the natural structure of the text. The semantic chunking strategy
    attempts to split the text based on semantic boundaries, which can help ensure that
    related concepts stay together.
    
    In conclusion, choosing the right chunking strategy depends on the specific requirements
    of your summarization task. For general-purpose summarization, the standard or adaptive
    strategies are usually a good choice. For more specialized tasks, one of the other
    strategies might be more appropriate.
    """
    
    # Clean the text
    text = text.strip()
    
    # Create chunking configurations for different strategies
    chunking_configs = {
        "standard": ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            strategy=ChunkingStrategy.STANDARD
        ),
        "small": ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            strategy=ChunkingStrategy.SMALL
        ),
        "large": ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            strategy=ChunkingStrategy.LARGE
        ),
        "adaptive": ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            strategy=ChunkingStrategy.ADAPTIVE
        ),
        "paragraph": ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            strategy=ChunkingStrategy.PARAGRAPH
        ),
    }
    
    # Import the chunking module
    from src.tldwatch.core.chunking import split_text
    
    # Demonstrate each chunking strategy
    for name, config in chunking_configs.items():
        chunks = split_text(text, config)
        print(f"\n{name.upper()} CHUNKING STRATEGY:")
        print(f"- Configuration: size={config.chunk_size}, overlap={config.chunk_overlap}")
        print(f"- Number of chunks: {len(chunks)}")
        print(f"- Average chunk length: {sum(len(c) for c in chunks) / len(chunks):.1f} characters")
        print(f"- First chunk preview: {chunks[0][:100]}...")


async def demo_summarizer(url: Optional[str] = None):
    """Demonstrate the new summarizer with different chunking strategies"""
    print("\n=== Summarizer Demo ===\n")
    
    # Use a default URL if none provided
    if not url:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    
    print(f"Summarizing: {url}")
    
    # Create a summarizer with default settings
    summarizer = Summarizer(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.7,
    )
    
    try:
        # Generate a summary
        summary = await summarizer.get_summary(url=url)
        
        print("\nSUMMARY:")
        print(summary)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await summarizer.close()


async def main():
    """Run the demo"""
    print("=== New Architecture Demo ===")
    print("This script demonstrates the new architecture with consolidated providers and improved chunking.")
    
    # Demo the provider factory
    await demo_provider_factory()
    
    # Demo chunking strategies
    await demo_chunking_strategies()
    
    # Demo the summarizer (optional)
    # Uncomment the line below to run the summarizer demo
    # await demo_summarizer()


if __name__ == "__main__":
    asyncio.run(main())
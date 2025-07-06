#!/usr/bin/env python3
"""
Migration demo showing the difference between the old complex interface
and the new simplified interface.
"""

import asyncio
import os


def show_old_way():
    """Show how complex the old interface was"""
    print("=== OLD WAY (Complex) ===")
    print("""
# Old way required lots of imports and configuration
from tldwatch import Summarizer
from tldwatch.core.providers.openai import OpenAIProvider
from tldwatch.core.providers.base import RateLimitConfig
from tldwatch.core.chunking import ChunkingStrategy, ChunkingConfig

# Complex rate limit configuration
rate_limit_config = RateLimitConfig(
    requests_per_minute=3500,
    tokens_per_minute=180000,
    max_retries=3,
    retry_delay=1.0,
    timeout=120.0
)

# Complex chunking configuration
chunking_config = ChunkingConfig(
    chunk_size=4000,
    chunk_overlap=200,
    strategy=ChunkingStrategy.STANDARD,
    min_chunk_size=500,
    max_chunk_size=8000
)

# Complex provider initialization
provider = OpenAIProvider(
    model="gpt-4o-mini",
    temperature=0.7,
    rate_limit_config=rate_limit_config,
    use_full_context=False
)

# Complex summarizer initialization
summarizer = Summarizer(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    chunk_size=4000,
    chunk_overlap=200,
    use_full_context=False,
    youtube_api_key=None,
    proxy_config=None
)

# Finally, summarize
summary = await summarizer.summarize_video("video_id")
""")


def show_new_way():
    """Show how simple the new interface is"""
    print("=== NEW WAY (Simple) ===")
    print("""
# New way - just one import and one line!
from tldwatch import summarize_video

# That's it!
summary = await summarize_video("video_id")

# Or with options:
summary = await summarize_video(
    "video_id",
    provider="openai",           # Choose provider
    model="gpt-4o-mini",        # Optional - uses default if not specified
    chunking_strategy="standard", # Simple chunking options
    temperature=0.7
)

# Or using the class interface:
from tldwatch import SimpleSummarizer

summarizer = SimpleSummarizer()
summary = await summarizer.summarize("video_id", provider="anthropic")
""")


async def demo_new_interface():
    """Demonstrate the new interface with actual code"""
    print("=== LIVE DEMO OF NEW INTERFACE ===")
    
    # Import the new interface
    try:
        from tldwatch import SimpleSummarizer, summarize_video, ChunkingStrategy
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Show available options
    summarizer = SimpleSummarizer()
    print(f"✓ Available providers: {summarizer.list_providers()}")
    print(f"✓ Available chunking strategies: {summarizer.list_chunking_strategies()}")
    
    # Show default models
    print("\n✓ Default models:")
    for provider in summarizer.list_providers():
        try:
            default_model = summarizer.get_default_model(provider)
            print(f"   {provider}: {default_model}")
        except Exception as e:
            print(f"   {provider}: Error - {e}")
    
    # Test text processing (without API calls)
    print("\n✓ Testing text processing:")
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """ * 10  # Make it longer
    
    # Test chunking strategies
    for strategy in ["none", "standard", "small", "large"]:
        try:
            # Create provider to test chunking (without making API calls)
            from tldwatch.core.providers.unified_provider import UnifiedProvider, ChunkingStrategy
            provider = UnifiedProvider("openai", chunking_strategy=ChunkingStrategy(strategy))
            chunks = provider.chunk_text(sample_text)
            print(f"   {strategy}: {len(chunks)} chunks (text length: {len(sample_text)} chars)")
        except Exception as e:
            print(f"   {strategy}: Error - {e}")
    
    # Show what an actual API call would look like (but don't make it)
    print("\n✓ Example API call (not executed):")
    print("""
    # This would work if you have an API key set:
    summary = await summarize_video(
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        provider="openai",
        chunking_strategy="standard"
    )
    """)
    
    # Check for API keys
    print("\n✓ API key status:")
    api_keys = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "GROQ_API_KEY", "DEEPSEEK_API_KEY", "CEREBRAS_API_KEY"
    ]
    
    for key in api_keys:
        status = "✓ Set" if os.getenv(key) else "✗ Not set"
        print(f"   {key}: {status}")


def show_benefits():
    """Show the benefits of the new approach"""
    print("=== BENEFITS OF NEW APPROACH ===")
    benefits = [
        "🎯 Simplicity: One import, one function call",
        "🔧 No complex configuration needed",
        "🚀 Sensible defaults for everything",
        "🔄 Easy to switch between providers",
        "📦 Unified interface for all providers",
        "🛡️ Built-in error handling and retries",
        "📏 Generic chunking that works everywhere",
        "🧹 Much less code to maintain",
        "📚 Easier to learn and use",
        "🔒 Backward compatible with old interface"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")


async def main():
    """Main demo function"""
    print("TLDWatch Interface Migration Demo")
    print("=" * 50)
    
    show_old_way()
    print("\n" + "=" * 50 + "\n")
    
    show_new_way()
    print("\n" + "=" * 50 + "\n")
    
    await demo_new_interface()
    print("\n" + "=" * 50 + "\n")
    
    show_benefits()
    print("\n" + "=" * 50)
    
    print("\nMigration complete! 🎉")
    print("The new interface is ready to use.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
"""
Examples of using tldwatch as an imported library.

This file demonstrates various ways to use tldwatch programmatically,
including different input methods, configuration options, and error handling.
"""

import asyncio
from pathlib import Path

from tldwatch import Summarizer, summarize_video


async def basic_usage():
    """Basic usage examples"""
    print("\n=== Basic Usage Examples ===")

    # Initialize with defaults
    summarizer = Summarizer()

    # Summarize using video ID
    print("Processing video ID...")
    summary = await summarizer.summarize("QAgR4uQ15rc")
    print(f"Summary from video ID: {summary[:200]}...")

    # Summarize using URL
    print("\nProcessing YouTube URL...")
    summary = await summarizer.summarize("https://www.youtube.com/watch?v=QAgR4uQ15rc")
    print(f"Summary from URL: {summary[:200]}...")

    # Process direct text
    print("\nProcessing direct text...")
    text = "This is a sample text that needs to be summarized. It contains information about various topics and should be condensed into a shorter, more digestible format. The text is long enough to demonstrate the summarization capabilities."
    summary = await summarizer.summarize(text)
    print(f"Summary from direct text: {summary[:200]}...")


async def provider_examples():
    """Examples with different providers"""
    print("\n=== Provider Examples ===")

    # Using Groq
    print("Using Groq provider...")
    summary = await summarize_video(
        "QAgR4uQ15rc", provider="groq", model="llama-3.1-8b-instant", temperature=0.7
    )
    print(f"Groq Summary: {summary[:200]}...")

    # Using OpenAI with specific model
    print("\nUsing OpenAI provider...")
    summary = await summarize_video(
        "QAgR4uQ15rc", provider="openai", model="gpt-4o-mini", temperature=0.5
    )
    print(f"OpenAI Summary: {summary[:200]}...")

    # Using Anthropic
    print("\nUsing Anthropic provider...")
    summary = await summarize_video(
        "QAgR4uQ15rc",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        temperature=0.3,
    )
    print(f"Anthropic Summary: {summary[:200]}...")


async def chunking_examples():
    """Examples with different chunking strategies"""
    print("\n=== Chunking Strategy Examples ===")

    summarizer = Summarizer()

    # No chunking (submit entire transcript)
    print("Using 'none' chunking strategy...")
    summary = await summarizer.summarize("QAgR4uQ15rc", chunking_strategy="none")
    print(f"No chunking Summary: {summary[:200]}...")

    # Large chunks for better context
    print("\nUsing 'large' chunking strategy...")
    summary = await summarizer.summarize("QAgR4uQ15rc", chunking_strategy="large")
    print(f"Large chunks Summary: {summary[:200]}...")

    # Small chunks for detailed processing
    print("\nUsing 'small' chunking strategy...")
    summary = await summarizer.summarize("QAgR4uQ15rc", chunking_strategy="small")
    print(f"Small chunks Summary: {summary[:200]}...")


async def batch_processing():
    """Example of batch processing multiple videos"""
    print("\n=== Batch Processing Example ===")

    video_ids = ["QAgR4uQ15rc", "dQw4w9WgXcQ"]  # Using real video IDs

    summarizer = Summarizer(provider="openai", model="gpt-4o-mini")
    output_dir = Path("summaries")
    output_dir.mkdir(exist_ok=True)

    for video_id in video_ids:
        try:
            print(f"Processing {video_id}...")
            summary = await summarizer.summarize(video_id)

            # Save to file
            output_file = output_dir / f"{video_id}_summary.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Saved summary to {output_file}")

        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")


async def cache_examples():
    """Examples with caching"""
    print("\n=== Caching Examples ===")

    summarizer = Summarizer()

    # First call - will cache the result
    print("First call (will cache)...")
    summary1 = await summarizer.summarize("QAgR4uQ15rc", use_cache=True)
    print(f"First summary: {summary1[:200]}...")

    # Second call - will use cache
    print("\nSecond call (will use cache)...")
    summary2 = await summarizer.summarize("QAgR4uQ15rc", use_cache=True)
    print(f"Second summary: {summary2[:200]}...")
    print(f"Results identical: {summary1 == summary2}")

    # Force regeneration
    print("\nForced regeneration (bypass cache)...")
    summary3 = await summarizer.summarize("QAgR4uQ15rc", use_cache=False)
    print(f"Force regenerated: {summary3[:200]}...")


async def error_handling():
    """Example of error handling"""
    print("\n=== Error Handling Examples ===")

    summarizer = Summarizer()

    # Handle invalid video ID
    try:
        await summarizer.summarize("invalid_video_id")
    except Exception as e:
        print(f"Expected error for invalid video ID: {str(e)}")

    # Handle short text
    try:
        await summarizer.summarize("short")
    except ValueError as e:
        print(f"Expected error for short text: {str(e)}")

    # Handle network issues gracefully
    try:
        # This might fail due to network issues
        await summarizer.summarize("nonexistent_video_id")
    except Exception as e:
        print(f"Network/API error handled: {str(e)}")


async def main():
    """Run all examples"""
    print("TLDWatch Library Usage Examples")
    print("=" * 40)

    await basic_usage()
    await provider_examples()
    await chunking_examples()
    await batch_processing()
    await cache_examples()
    await error_handling()

    print("\n=== All Examples Complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Process failed: {str(e)}")

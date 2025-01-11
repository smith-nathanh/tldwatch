"""
Examples of using tldwatch as an imported library.

This file demonstrates various ways to use tldwatch programmatically,
including different input methods, configuration options, and error handling.
"""

import asyncio
import os
from pathlib import Path

from tldwatch import Summarizer


async def basic_usage():
    """Basic usage examples"""
    print("\n=== Basic Usage Examples ===")

    # Initialize with defaults (OpenAI provider)
    summarizer = Summarizer()

    # Get summary using video ID
    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
    print("Summary from video ID:", summary)

    # Get summary using URL
    summary = await summarizer.get_summary(
        url="https://www.youtube.com/watch?v=QAgR4uQ15rc"
    )
    print("Summary from URL:", summary)

    # Process direct transcript input
    transcript = "This is a sample transcript that needs to be summarized..."
    summary = await summarizer.get_summary(transcript_text=transcript)
    print("Summary from direct transcript:", summary)


async def provider_examples():
    """Examples with different providers"""
    print("\n=== Provider Examples ===")

    # Using Groq
    groq_summarizer = Summarizer(
        provider="groq", model="mixtral-8x7b-32768", temperature=0.7
    )
    summary = await groq_summarizer.get_summary(video_id="QAgR4uQ15rc")
    print("Groq Summary:", summary)

    # Using OpenAI with GPT-4
    openai_summarizer = Summarizer(provider="openai", model="gpt-4", temperature=0.5)
    summary = await openai_summarizer.get_summary(video_id="QAgR4uQ15rc")
    print("OpenAI Summary:", summary)

    # Using local Ollama
    ollama_summarizer = Summarizer(provider="ollama", model="mistral", temperature=0.7)
    summary = await ollama_summarizer.get_summary(video_id="QAgR4uQ15rc")
    print("Ollama Summary:", summary)


async def advanced_configuration():
    """Examples with advanced configuration"""
    print("\n=== Advanced Configuration Examples ===")

    # Using full context window
    summarizer = Summarizer(
        provider="groq",
        model="mixtral-8x7b-32768",
        use_full_context=True,
        chunk_size=8000,  # Larger chunks
        chunk_overlap=400,  # More overlap
        temperature=0.8,
    )

    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
    print("Full Context Summary:", summary)


async def batch_processing():
    """Example of batch processing multiple videos"""
    print("\n=== Batch Processing Example ===")

    video_ids = ["QAgR4uQ15rc", "QkGwxtALTLU", "another_video_id"]

    summarizer = Summarizer(provider="groq")
    output_dir = Path("summaries")
    output_dir.mkdir(exist_ok=True)

    for video_id in video_ids:
        try:
            summary = await summarizer.get_summary(video_id=video_id)

            # Export to file
            output_file = output_dir / f"{video_id}_summary.json"
            summarizer.export_summary(str(output_file))
            print(f"Processed {video_id}")

        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")


async def error_handling():
    """Example of proper error handling"""
    print("\n=== Error Handling Example ===")

    summarizer = Summarizer()

    try:
        # Try with invalid video ID
        summary = await summarizer.get_summary(video_id="invalid_video_id")
    except ValueError as e:
        print(f"Invalid input error: {str(e)}")
    except Exception as e:
        print(f"General error: {str(e)}")


async def metadata_enrichment():
    """Example using YouTube API for metadata enrichment"""
    print("\n=== Metadata Enrichment Example ===")

    # Initialize with YouTube API key
    summarizer = Summarizer(youtube_api_key=os.environ.get("YOUTUBE_API_KEY"))

    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")

    # Export with metadata
    summarizer.export_summary("summary_with_metadata.json")
    print("Exported summary with metadata")


async def custom_prompt_handling():
    """Example of processing transcripts with specific context"""
    print("\n=== Custom Prompt Handling Example ===")

    # Read multiple transcripts
    transcripts = [
        "First transcript content...",
        "Second transcript content...",
        "Third transcript content...",
    ]

    summarizer = Summarizer(provider="openai")
    summaries = []

    for transcript in transcripts:
        summary = await summarizer.get_summary(transcript_text=transcript)
        summaries.append(summary)

    # Combine summaries if needed
    combined = "\n\n".join(summaries)
    print("Processed multiple transcripts:", combined)


def sync_wrapper():
    """
    Example of using the async API in synchronous code.
    Useful when you can't use async/await directly.
    """
    print("\n=== Synchronous Usage Example ===")

    def get_summary_sync(video_id: str) -> str:
        """Synchronous wrapper for get_summary"""
        summarizer = Summarizer()
        return asyncio.run(summarizer.get_summary(video_id=video_id))

    # Use synchronously
    summary = get_summary_sync("QAgR4uQ15rc")
    print("Sync Summary:", summary)


async def main():
    """Run all examples"""
    # Basic examples
    await basic_usage()

    # Provider examples
    await provider_examples()

    # Advanced configuration
    await advanced_configuration()

    # Batch processing
    await batch_processing()

    # Error handling
    await error_handling()

    # Metadata enrichment
    await metadata_enrichment()

    # Custom prompt handling
    await custom_prompt_handling()

    # Synchronous usage
    sync_wrapper()


if __name__ == "__main__":
    asyncio.run(main())

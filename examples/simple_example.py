"""
Simple, focused example showing the core tldwatch functionality.
This example demonstrates the most common use cases.
"""

import asyncio
import os

from tldwatch import Summarizer, summarize_video


async def main():
    """Demonstrate core tldwatch functionality"""
    print("TLDWatch Simple Example")
    print("=" * 30)

    # Check if we have API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_groq = bool(os.environ.get("GROQ_API_KEY"))

    if not (has_openai or has_groq):
        print("⚠️  No API keys found!")
        print("Set OPENAI_API_KEY or GROQ_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return

    # Example video ID (a short video for testing)
    video_id = "dQw4w9WgXcQ"  # Never Gonna Give You Up

    try:
        # Method 1: Quick usage with convenience function
        print("\n1. Quick usage with convenience function:")
        summary = await summarize_video(video_id)
        print(f"   Summary: {summary[:100]}...")

        # Method 2: Using Summarizer class with options
        print("\n2. Using Summarizer class with options:")
        summarizer = Summarizer()

        # Choose provider based on available API keys
        provider = "openai" if has_openai else "groq"

        summary = await summarizer.summarize(
            video_id, provider=provider, chunking_strategy="standard", temperature=0.7
        )
        print(f"   Provider: {provider}")
        print(f"   Summary: {summary[:100]}...")

        # Method 3: From YouTube URL
        print("\n3. From YouTube URL:")
        url = f"https://www.youtube.com/watch?v={video_id}"
        summary = await summarizer.summarize(url)
        print(f"   Summary: {summary[:100]}...")

        # Method 4: Direct text summarization
        print("\n4. Direct text summarization:")
        text = """
        This is a longer piece of text that demonstrates the direct text
        summarization capability. You can pass any text directly to the
        summarizer instead of a YouTube video. This is useful for summarizing
        articles, documents, or any other text content you have.
        """
        summary = await summarizer.summarize(text.strip())
        print(f"   Summary: {summary[:100]}...")

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("This might be due to:")
        print("- Invalid API key")
        print("- Network connectivity issues")
        print("- Video not available")


if __name__ == "__main__":
    asyncio.run(main())

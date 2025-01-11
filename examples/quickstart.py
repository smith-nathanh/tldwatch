"""
Quickstart example for tldwatch library usage.
Shows the most common ways to use the library.
"""

import asyncio

from tldwatch import Summarizer


async def main():
    # Initialize summarizer (uses OpenAI by default)
    summarizer = Summarizer(
        provider="openai",  # or "groq", "cerebras", "ollama"
        model="gpt-4o",  # optional, uses provider default if not specified
    )

    # 1. Summarize from YouTube video ID
    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
    print("\nSummary from video ID:", summary)

    # 2. Summarize from YouTube URL
    summary = await summarizer.get_summary(
        url="https://www.youtube.com/watch?v=QAgR4uQ15rc"
    )
    print("\nSummary from URL:", summary)

    # 3. Summarize direct transcript input
    transcript = """
    This is a sample transcript.
    It can be any text you want to summarize.
    The summarizer will process it using the configured provider.
    """

    summary = await summarizer.get_summary(transcript_text=transcript)
    print("\nSummary from transcript:", summary)

    # 4. Export summary to file
    summarizer.export_summary("summary.json")


if __name__ == "__main__":
    asyncio.run(main())

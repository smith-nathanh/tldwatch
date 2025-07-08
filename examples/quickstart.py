"""
Quickstart example for tldwatch - the simplest way to get started.
"""

import asyncio
import os

from tldwatch import Summarizer, summarize_video


async def main():
    """Quick start examples"""
    print("TLDWatch Quickstart")
    print("=" * 20)

    # Check for API keys
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")):
        print("❌ Missing API key!")
        print("Set one of these environment variables:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export GROQ_API_KEY='your-key-here'")
        return

    try:
        # Example 1: Convenience function (simplest)
        print("\n1. Using convenience function:")
        summary = await summarize_video("dQw4w9WgXcQ")
        print(f"   {summary[:100]}...")

        # Example 2: Summarizer class (more control)
        print("\n2. Using Summarizer class:")
        summarizer = Summarizer()
        summary = await summarizer.summarize(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            provider="openai" if os.environ.get("OPENAI_API_KEY") else "groq",
        )
        print(f"   {summary[:100]}...")

        print("\n✅ Success! You're ready to use TLDWatch.")
        print("\nNext steps:")
        print("- Try: tldwatch --create-config")
        print("- See: examples/simple_example.py")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

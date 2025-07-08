#!/usr/bin/env python3
"""
Cache usage example for TLDWatch.
Demonstrates how to use the caching functionality programmatically.
"""

import asyncio
import logging

from tldwatch import Summarizer, clear_cache, get_cache, get_cache_stats


async def main():
    """Demonstrate cache functionality"""
    print("TLDWatch Cache Usage Example")
    print("=" * 40)

    # Enable logging to see cache activity
    logging.basicConfig(level=logging.INFO)

    # Create a summarizer
    summarizer = Summarizer()

    # Use a sample video ID (Rick Astley - Never Gonna Give You Up)
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    print("\n1. Generating summary (will be cached)...")
    summary1 = await summarizer.summarize(
        video_url, provider="openai", model="gpt-4o-mini", use_cache=True
    )
    print(f"Summary length: {len(summary1)} characters")

    print("\n2. Retrieving same summary (should use cache)...")
    summary2 = await summarizer.summarize(
        video_url, provider="openai", model="gpt-4o-mini", use_cache=True
    )
    print(f"Summary length: {len(summary2)} characters")
    print(f"Summaries identical: {summary1 == summary2}")

    print("\n3. Cache statistics:")
    stats = get_cache_stats()
    print(f"  Cache location: {stats['cache_dir']}")
    print(f"  Total cached videos: {stats['total_videos']}")
    print(f"  Total cache size: {stats['total_size_mb']} MB")

    print(
        "\n4. Generating summary with different model (will create new cache entry)..."
    )
    summary3 = await summarizer.summarize(
        video_url,
        provider="openai",
        model="gpt-4o-mini",  # Same model but different parameters
        temperature=0.5,  # Different temperature
        use_cache=True,
    )
    print(f"Summary length: {len(summary3)} characters")
    print(f"Different from first summary: {summary1 != summary3}")

    print("\n5. Updated cache statistics:")
    stats = get_cache_stats()
    print(f"  Total cached videos: {stats['total_videos']}")
    print(f"  Total cache size: {stats['total_size_mb']} MB")

    print("\n6. Listing cached entries:")
    cache = get_cache()
    cached_videos = cache.list_cached_videos()
    for video_id in cached_videos:
        entry = cache.get_cached_summary(video_id)
        if entry:
            print(
                f"  {video_id}: {entry.provider}/{entry.model} ({entry.chunking_strategy})"
            )

    print("\n7. Generating summary without cache...")
    summary4 = await summarizer.summarize(
        video_url,
        provider="openai",
        model="gpt-4o-mini",
        use_cache=False,  # Disable cache
    )
    print(f"Summary length: {len(summary4)} characters")
    print(f"Identical to cached version: {summary1 == summary4}")

    print("\n8. Cache management example:")
    response = input("Clear cache for this video? (y/N): ")
    if response.lower() in ["y", "yes"]:
        video_id = "dQw4w9WgXcQ"  # Extract from URL
        removed = clear_cache(video_id=video_id)
        print(f"Cleared {removed} cache entries")

        # Final stats
        stats = get_cache_stats()
        print(f"  Remaining cached videos: {stats['total_videos']}")

    print("\nCache usage example completed!")


if __name__ == "__main__":
    asyncio.run(main())

"""
Cache management CLI for TLDWatch.
Provides command-line utilities for managing the summary cache.
"""

import argparse
import sys

from ..core.user_config import get_user_config
from ..utils.cache import clear_cache, get_cache, get_cache_stats


def cmd_list_cache(args) -> None:
    """List all cached videos"""
    user_config = get_user_config()
    cache = get_cache(user_config.get_cache_dir())

    cached_videos = cache.list_cached_videos()

    if not cached_videos:
        print("No cached entries found.")
        return

    print(f"Found {len(cached_videos)} cached videos:")
    for video_id in cached_videos:
        has_summary = cache.has_cached_summary(video_id)
        has_transcript = cache.has_cached_transcript(video_id)

        cache_types = []
        if has_summary:
            cache_types.append("summary")
        if has_transcript:
            cache_types.append("transcript")

        cache_info = f"({', '.join(cache_types)})"

        if has_summary:
            entry = cache.get_cached_summary(video_id)
            if entry:
                print(
                    f"  {video_id} - {entry.provider}/{entry.model} - {entry.chunking_strategy} {cache_info}"
                )
                if entry.video_metadata and entry.video_metadata.get("title"):
                    print(f"    Title: {entry.video_metadata['title']}")
            else:
                print(f"  {video_id} - (error reading summary cache) {cache_info}")
        else:
            print(f"  {video_id} - {cache_info}")


def cmd_clear_cache(args) -> None:
    """Clear cache entries"""
    user_config = get_user_config()

    if args.video_id:
        # Clear specific video
        removed = clear_cache(
            video_id=args.video_id, cache_dir=user_config.get_cache_dir()
        )
        if removed > 0:
            print(
                f"Cleared cache (summaries and transcripts) for video {args.video_id}"
            )
        else:
            print(f"No cache found for video {args.video_id}")
    else:
        # Clear all cache
        if not args.force:
            response = input(
                "Are you sure you want to clear ALL cached summaries and transcripts? (y/N): "
            )
            if response.lower() not in ["y", "yes"]:
                print("Cancelled.")
                return

        removed = clear_cache(cache_dir=user_config.get_cache_dir())
        print(f"Cleared {removed} cached files (summaries and transcripts)")


def cmd_cache_stats(args) -> None:
    """Show cache statistics"""
    user_config = get_user_config()
    stats = get_cache_stats(cache_dir=user_config.get_cache_dir())

    print("Cache Statistics:")
    print(f"  Location: {stats['cache_dir']}")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Cached summaries: {stats['cached_summaries']}")
    print(f"  Cached transcripts: {stats['cached_transcripts']}")
    print(
        f"  Total size: {stats['total_size_mb']} MB ({stats['total_size_bytes']} bytes)"
    )


def cmd_show_cache_entry(args) -> None:
    """Show detailed information about a cached entry"""
    user_config = get_user_config()
    cache = get_cache(user_config.get_cache_dir())

    entry = cache.get_cached_summary(args.video_id)
    transcript = cache.get_cached_transcript(args.video_id)

    if not entry and not transcript:
        print(f"No cached entries found for video {args.video_id}")
        return

    print(f"Cache Entry for {args.video_id}:")

    if entry:
        print("  Summary Cache:")
        print(f"    Provider: {entry.provider}")
        print(f"    Model: {entry.model}")
        print(f"    Chunking Strategy: {entry.chunking_strategy}")
        print(f"    Temperature: {entry.temperature}")
        print(f"    Cached: {entry.timestamp}")

        if entry.video_metadata:
            print("    Video Metadata:")
            for key, value in entry.video_metadata.items():
                if key != "html":  # Skip HTML content as it's verbose
                    print(f"      {key}: {value}")

        if args.show_summary:
            print("\n  Summary:")
            print(f"    {entry.summary}")
    else:
        print("  Summary Cache: Not found")

    if transcript:
        print(f"\n  Transcript Cache: Found ({len(transcript)} characters)")
        if args.show_summary:  # Reuse this flag for showing transcript content
            print(f"    {transcript[:500]}{'...' if len(transcript) > 500 else ''}")
    else:
        print("  Transcript Cache: Not found")


def cmd_cleanup_cache(args) -> None:
    """Clean up old cache entries"""
    user_config = get_user_config()
    cache = get_cache(user_config.get_cache_dir())

    max_age_days = args.max_age_days or user_config.get_cache_max_age_days()
    removed = cache.cleanup_old_cache(max_age_days)

    print(f"Cleaned up {removed} cache entries older than {max_age_days} days")


def create_cache_parser() -> argparse.ArgumentParser:
    """Create argument parser for cache management commands"""
    parser = argparse.ArgumentParser(
        prog="tldwatch-cache", description="Manage TLDWatch summary cache"
    )

    subparsers = parser.add_subparsers(dest="command", help="Cache management commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List cached summaries")
    list_parser.set_defaults(func=cmd_list_cache)

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cached summaries")
    clear_parser.add_argument("--video-id", help="Clear cache for specific video ID")
    clear_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    clear_parser.set_defaults(func=cmd_clear_cache)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.set_defaults(func=cmd_cache_stats)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show cached entry details")
    show_parser.add_argument("video_id", help="Video ID to show")
    show_parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Include full summary and transcript text",
    )
    show_parser.set_defaults(func=cmd_show_cache_entry)

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old cache entries")
    cleanup_parser.add_argument(
        "--max-age-days",
        type=int,
        help="Maximum age in days (uses config default if not specified)",
    )
    cleanup_parser.set_defaults(func=cmd_cleanup_cache)

    return parser


def main() -> None:
    """Main entry point for cache management CLI"""
    parser = create_cache_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

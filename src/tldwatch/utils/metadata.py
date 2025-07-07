"""
YouTube video metadata fetcher.
Provides functionality to fetch basic video metadata for caching.
"""

import logging
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


async def fetch_video_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch basic metadata for a YouTube video.

    This uses YouTube's oembed API which doesn't require an API key but
    provides limited metadata (title, author, thumbnail).

    Args:
        video_id: YouTube video ID

    Returns:
        Dictionary with video metadata or None if fetch fails
    """
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract useful metadata
                    metadata = {
                        "title": data.get("title"),
                        "author_name": data.get("author_name"),
                        "author_url": data.get("author_url"),
                        "thumbnail_url": data.get("thumbnail_url"),
                        "width": data.get("width"),
                        "height": data.get("height"),
                        "html": data.get("html"),
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    }

                    logger.debug(
                        f"Fetched metadata for video {video_id}: {metadata['title']}"
                    )
                    return metadata
                else:
                    logger.warning(
                        f"Failed to fetch metadata for {video_id}: HTTP {response.status}"
                    )
                    return None

    except Exception as e:
        logger.warning(f"Error fetching metadata for video {video_id}: {e}")
        return None


def extract_metadata_from_transcript_api(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from YouTube transcript API response if available.
    This is a fallback when oembed fails.

    Args:
        video_id: YouTube video ID

    Returns:
        Basic metadata dictionary or None
    """
    # This would need to be implemented if we want to extract metadata
    # from the transcript API response, but it's quite limited
    return {"url": f"https://www.youtube.com/watch?v={video_id}", "video_id": video_id}

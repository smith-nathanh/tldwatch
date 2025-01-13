"""Utilities for parsing YouTube URLs and extracting video IDs."""

import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

# Common YouTube domain patterns
YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "m.youtube.com",
}


def is_youtube_url(url: str) -> bool:
    """
    Check if a URL is a valid YouTube URL.

    Args:
        url: URL to check

    Returns:
        True if URL is a valid YouTube URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        return (
            parsed.netloc in YOUTUBE_DOMAINS
            and (
                parsed.path == "/watch"  # Standard watch URLs
                or "/watch/" in parsed.path  # Some mobile URLs
                or parsed.netloc == "youtu.be"
            )  # Short URLs
        )
    except Exception:
        return False


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the video ID from a YouTube URL.

    Handles various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID&feature=shared
    - https://www.youtube.com/watch?v=VIDEO_ID&list=PLAYLIST_ID
    - https://m.youtube.com/watch?v=VIDEO_ID

    Args:
        url: YouTube URL

    Returns:
        Video ID if found, None otherwise
    """
    if not url:
        return None

    try:
        # Handle youtu.be short URLs
        parsed = urlparse(url)
        if parsed.netloc == "youtu.be":
            # Video ID is in the path
            video_id = parsed.path.lstrip("/")
            return video_id if _is_valid_video_id(video_id) else None

        # Handle standard youtube.com URLs
        if parsed.netloc in YOUTUBE_DOMAINS:
            # Parse query parameters
            query_params = parse_qs(parsed.query)

            # Get video ID from 'v' parameter
            if "v" in query_params:
                video_id = query_params["v"][0]
                return video_id if _is_valid_video_id(video_id) else None

        return None
    except Exception:
        return None


def _is_valid_video_id(video_id: str) -> bool:
    """
    Check if a string matches YouTube video ID format.

    YouTube video IDs are typically 11 characters long and
    contain only alphanumeric characters, underscores, and hyphens.

    Args:
        video_id: String to check

    Returns:
        True if string matches video ID format, False otherwise
    """
    if not video_id:
        return False

    # YouTube video IDs are typically 11 characters
    if len(video_id) != 11:
        return False

    # Video IDs contain only alphanumeric chars, underscores, and hyphens
    pattern = r"^[A-Za-z0-9_-]+$"
    return bool(re.match(pattern, video_id))

"""
Cache management for TLDWatch summaries.
Provides functionality to cache video summaries with metadata about the generation settings.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached summary entry with metadata"""

    video_id: str
    summary: str
    provider: str
    model: str
    chunking_strategy: str
    temperature: float
    timestamp: float
    video_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create instance from dictionary"""
        return cls(**data)


@dataclass
class TranscriptCacheEntry:
    """Represents a cached transcript entry"""

    video_id: str
    transcript: str
    timestamp: float
    video_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptCacheEntry":
        """Create instance from dictionary"""
        return cls(**data)


class SummaryCache:
    """
    Manages caching of video summaries and transcripts with metadata.

    Cache structure:
    - Default location: ~/.cache/tldwatch/summaries/
    - Each video gets its own JSON file: {video_id}_cache.json
    - Files contain summary, generation metadata, and video metadata
    - Transcripts are cached separately: {video_id}_transcript.json
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Custom cache directory path. If None, uses default location.
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "tldwatch" / "summaries"

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized cache at: {self.cache_dir}")

    def _get_cache_file(self, video_id: str) -> Path:
        """Get the cache file path for a video ID"""
        return self.cache_dir / f"{video_id}_cache.json"

    def _get_transcript_cache_file(self, video_id: str) -> Path:
        """Get the transcript cache file path for a video ID"""
        return self.cache_dir / f"{video_id}_transcript.json"

    def has_cached_transcript(self, video_id: str) -> bool:
        """
        Check if a transcript exists in cache.

        Args:
            video_id: YouTube video ID

        Returns:
            True if cached transcript exists
        """
        transcript_file = self._get_transcript_cache_file(video_id)
        return transcript_file.exists()

    def get_cached_transcript(self, video_id: str) -> Optional[str]:
        """
        Retrieve cached transcript for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Transcript text if found, None otherwise
        """
        transcript_file = self._get_transcript_cache_file(video_id)
        if not transcript_file.exists():
            return None

        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)

            entry = TranscriptCacheEntry.from_dict(transcript_data)
            logger.debug(f"Retrieved cached transcript for {video_id}")
            return entry.transcript

        except Exception as e:
            logger.warning(f"Error reading cached transcript for {video_id}: {e}")
            return None

    def cache_transcript(
        self,
        video_id: str,
        transcript: str,
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache a transcript.

        Args:
            video_id: YouTube video ID
            transcript: Transcript text
            video_metadata: Optional video metadata (title, description, etc.)
        """
        transcript_entry = TranscriptCacheEntry(
            video_id=video_id,
            transcript=transcript,
            timestamp=time.time(),
            video_metadata=video_metadata,
        )

        transcript_file = self._get_transcript_cache_file(video_id)

        try:
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump(transcript_entry.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Cached transcript for {video_id}")

        except Exception as e:
            logger.error(f"Error caching transcript for {video_id}: {e}")

    def has_cached_summary(
        self,
        video_id: str,
        provider: str = None,
        model: str = None,
        chunking_strategy: str = None,
        temperature: float = None,
    ) -> bool:
        """
        Check if a summary exists in cache with optional parameter matching.

        Args:
            video_id: YouTube video ID
            provider: Optional provider filter
            model: Optional model filter
            chunking_strategy: Optional chunking strategy filter
            temperature: Optional temperature filter

        Returns:
            True if matching cached summary exists
        """
        cache_file = self._get_cache_file(video_id)
        if not cache_file.exists():
            return False

        # If no specific parameters requested, just check if any cache exists
        if all(
            param is None for param in [provider, model, chunking_strategy, temperature]
        ):
            return True

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            entry = CacheEntry.from_dict(cache_data)

            # Check parameter matches
            if provider and entry.provider != provider:
                return False
            if model and entry.model != model:
                return False
            if chunking_strategy and entry.chunking_strategy != chunking_strategy:
                return False
            if temperature is not None and abs(entry.temperature - temperature) > 0.01:
                return False

            return True

        except Exception as e:
            logger.warning(f"Error reading cache for {video_id}: {e}")
            return False

    def get_cached_summary(self, video_id: str) -> Optional[CacheEntry]:
        """
        Retrieve cached summary for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            CacheEntry if found, None otherwise
        """
        cache_file = self._get_cache_file(video_id)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            entry = CacheEntry.from_dict(cache_data)
            logger.debug(f"Retrieved cached summary for {video_id}")
            return entry

        except Exception as e:
            logger.warning(f"Error reading cached summary for {video_id}: {e}")
            return None

    def cache_summary(
        self,
        video_id: str,
        summary: str,
        provider: str,
        model: str,
        chunking_strategy: str,
        temperature: float,
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache a summary with generation metadata.

        Args:
            video_id: YouTube video ID
            summary: Generated summary text
            provider: LLM provider used
            model: Specific model used
            chunking_strategy: Chunking strategy used
            temperature: Generation temperature used
            video_metadata: Optional video metadata (title, description, etc.)
        """
        cache_entry = CacheEntry(
            video_id=video_id,
            summary=summary,
            provider=provider,
            model=model,
            chunking_strategy=chunking_strategy,
            temperature=temperature,
            timestamp=time.time(),
            video_metadata=video_metadata,
        )

        cache_file = self._get_cache_file(video_id)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_entry.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(
                f"Cached summary for {video_id} (provider={provider}, model={model})"
            )

        except Exception as e:
            logger.error(f"Error caching summary for {video_id}: {e}")

    def clear_cache(self, video_id: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            video_id: If provided, clear only this video's cache. If None, clear all cache.

        Returns:
            Number of files removed
        """
        removed_count = 0

        if video_id:
            # Clear specific video cache (both summary and transcript)
            cache_file = self._get_cache_file(video_id)
            transcript_file = self._get_transcript_cache_file(video_id)

            for file_to_remove in [cache_file, transcript_file]:
                if file_to_remove.exists():
                    try:
                        file_to_remove.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.error(f"Error clearing cache file {file_to_remove}: {e}")

            if removed_count > 0:
                logger.info(
                    f"Cleared cache for video {video_id} ({removed_count} files)"
                )
        else:
            # Clear all cache (both summaries and transcripts)
            try:
                for cache_file in self.cache_dir.glob("*_cache.json"):
                    cache_file.unlink()
                    removed_count += 1
                for transcript_file in self.cache_dir.glob("*_transcript.json"):
                    transcript_file.unlink()
                    removed_count += 1
                logger.info(f"Cleared all cache ({removed_count} files)")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

        return removed_count

    def list_cached_videos(self) -> List[str]:
        """
        List all video IDs that have cached summaries.

        Returns:
            List of video IDs
        """
        video_ids = set()
        try:
            # Get video IDs from summary cache files
            for cache_file in self.cache_dir.glob("*_cache.json"):
                # Extract video ID from filename (remove "_cache.json" suffix)
                video_id = cache_file.stem.replace("_cache", "")
                video_ids.add(video_id)

            # Get video IDs from transcript cache files
            for transcript_file in self.cache_dir.glob("*_transcript.json"):
                # Extract video ID from filename (remove "_transcript.json" suffix)
                video_id = transcript_file.stem.replace("_transcript", "")
                video_ids.add(video_id)
        except Exception as e:
            logger.error(f"Error listing cached videos: {e}")

        return sorted(list(video_ids))

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cached_videos = self.list_cached_videos()
        total_size = 0
        summary_count = 0
        transcript_count = 0

        try:
            for cache_file in self.cache_dir.glob("*_cache.json"):
                total_size += cache_file.stat().st_size
                summary_count += 1
            for transcript_file in self.cache_dir.glob("*_transcript.json"):
                total_size += transcript_file.stat().st_size
                transcript_count += 1
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")

        return {
            "cache_dir": str(self.cache_dir),
            "total_videos": len(cached_videos),
            "cached_summaries": summary_count,
            "cached_transcripts": transcript_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def cleanup_old_cache(self, max_age_days: int = 30) -> int:
        """
        Remove cache entries older than specified days.

        Args:
            max_age_days: Maximum age in days for cache entries

        Returns:
            Number of files removed
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_count = 0

        try:
            # Clean up summary cache files
            for cache_file in self.cache_dir.glob("*_cache.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)

                    timestamp = cache_data.get("timestamp", 0)
                    if timestamp < cutoff_time:
                        cache_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old cache file: {cache_file}")

                except Exception as e:
                    logger.warning(f"Error checking cache file {cache_file}: {e}")

            # Clean up transcript cache files
            for transcript_file in self.cache_dir.glob("*_transcript.json"):
                try:
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        transcript_data = json.load(f)

                    timestamp = transcript_data.get("timestamp", 0)
                    if timestamp < cutoff_time:
                        transcript_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old transcript file: {transcript_file}")

                except Exception as e:
                    logger.warning(
                        f"Error checking transcript file {transcript_file}: {e}"
                    )

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache entries")

        return removed_count


# Global cache instance
_cache_instance = None


def get_cache(cache_dir: Optional[str] = None) -> SummaryCache:
    """Get the global cache instance"""
    global _cache_instance
    if _cache_instance is None or (
        cache_dir and str(_cache_instance.cache_dir) != cache_dir
    ):
        _cache_instance = SummaryCache(cache_dir)
    return _cache_instance


def clear_cache(video_id: Optional[str] = None, cache_dir: Optional[str] = None) -> int:
    """
    Convenience function to clear cache.

    Args:
        video_id: If provided, clear only this video's cache. If None, clear all cache.
        cache_dir: Optional custom cache directory

    Returns:
        Number of files removed
    """
    cache = get_cache(cache_dir)
    return cache.clear_cache(video_id)


def get_cache_stats(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get cache statistics.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        Dictionary with cache statistics
    """
    cache = get_cache(cache_dir)
    return cache.get_cache_stats()


def has_cached_transcript(video_id: str, cache_dir: Optional[str] = None) -> bool:
    """
    Convenience function to check if a transcript is cached.

    Args:
        video_id: YouTube video ID
        cache_dir: Optional custom cache directory

    Returns:
        True if cached transcript exists
    """
    cache = get_cache(cache_dir)
    return cache.has_cached_transcript(video_id)


def get_cached_transcript(
    video_id: str, cache_dir: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to get a cached transcript.

    Args:
        video_id: YouTube video ID
        cache_dir: Optional custom cache directory

    Returns:
        Cached transcript text if found, None otherwise
    """
    cache = get_cache(cache_dir)
    return cache.get_cached_transcript(video_id)

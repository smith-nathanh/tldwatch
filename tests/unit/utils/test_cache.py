"""
Unit tests for the cache functionality.
Tests summary and transcript caching with various scenarios.
"""

import json
import time
from pathlib import Path

from tldwatch.utils.cache import CacheEntry, SummaryCache, TranscriptCacheEntry


class TestCacheEntry:
    """Test CacheEntry dataclass functionality."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
            timestamp=time.time(),
            video_metadata={"title": "Test Video"},
        )

        assert entry.video_id == "test123"
        assert entry.summary == "Test summary"
        assert entry.provider == "openai"
        assert entry.model == "gpt-4o"

    def test_cache_entry_to_dict(self):
        """Test converting cache entry to dictionary."""
        timestamp = time.time()
        entry = CacheEntry(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
            timestamp=timestamp,
            video_metadata={"title": "Test Video"},
        )

        data = entry.to_dict()
        assert data["video_id"] == "test123"
        assert data["summary"] == "Test summary"
        assert data["timestamp"] == timestamp

    def test_cache_entry_from_dict(self):
        """Test creating cache entry from dictionary."""
        timestamp = time.time()
        data = {
            "video_id": "test123",
            "summary": "Test summary",
            "provider": "openai",
            "model": "gpt-4o",
            "chunking_strategy": "standard",
            "temperature": 0.7,
            "timestamp": timestamp,
            "video_metadata": {"title": "Test Video"},
        }

        entry = CacheEntry.from_dict(data)
        assert entry.video_id == "test123"
        assert entry.summary == "Test summary"
        assert entry.timestamp == timestamp


class TestTranscriptCacheEntry:
    """Test TranscriptCacheEntry dataclass functionality."""

    def test_transcript_cache_entry_creation(self):
        """Test creating a transcript cache entry."""
        entry = TranscriptCacheEntry(
            video_id="test123",
            transcript="Test transcript text",
            timestamp=time.time(),
            video_metadata={"title": "Test Video"},
        )

        assert entry.video_id == "test123"
        assert entry.transcript == "Test transcript text"

    def test_transcript_cache_entry_serialization(self):
        """Test transcript cache entry serialization."""
        timestamp = time.time()
        entry = TranscriptCacheEntry(
            video_id="test123",
            transcript="Test transcript",
            timestamp=timestamp,
            video_metadata={"title": "Test Video"},
        )

        data = entry.to_dict()
        recreated = TranscriptCacheEntry.from_dict(data)

        assert recreated.video_id == entry.video_id
        assert recreated.transcript == entry.transcript
        assert recreated.timestamp == entry.timestamp


class TestSummaryCache:
    """Test SummaryCache functionality."""

    def test_cache_initialization_default_path(self):
        """Test cache initialization with default path."""
        cache = SummaryCache()
        expected_path = Path.home() / ".cache" / "tldwatch" / "summaries"
        assert cache.cache_dir == expected_path

    def test_cache_initialization_custom_path(self, temp_cache_dir):
        """Test cache initialization with custom path."""
        cache = SummaryCache(cache_dir=temp_cache_dir)
        assert cache.cache_dir == Path(temp_cache_dir)

    def test_cache_summary(self, temp_cache_dir):
        """Test caching a summary."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        cache.cache_summary(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
            video_metadata={"title": "Test Video"},
        )

        cache_file = cache._get_cache_file("test123")
        assert cache_file.exists()

        with open(cache_file, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["video_id"] == "test123"
        assert data[0]["summary"] == "Test summary"

    def test_has_cached_summary(self, temp_cache_dir):
        """Test checking if summary exists in cache."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        # Should not exist initially
        assert not cache.has_cached_summary(
            video_id="test123",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        # Cache a summary
        cache.cache_summary(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        # Should exist now
        assert cache.has_cached_summary(
            video_id="test123",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

    def test_get_cached_summary(self, temp_cache_dir):
        """Test retrieving cached summary."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        # Cache a summary
        cache.cache_summary(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
            video_metadata={"title": "Test Video"},
        )

        # Retrieve the summary
        entry = cache.get_cached_summary(
            video_id="test123",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        assert entry is not None
        assert entry.summary == "Test summary"
        assert entry.provider == "openai"

    def test_cache_transcript(self, temp_cache_dir):
        """Test caching a transcript."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        cache.cache_transcript(
            video_id="test123",
            transcript="Test transcript",
            video_metadata={"title": "Test Video"},
        )

        transcript_file = cache._get_transcript_cache_file("test123")
        assert transcript_file.exists()

        with open(transcript_file, "r") as f:
            data = json.load(f)

        assert data["video_id"] == "test123"
        assert data["transcript"] == "Test transcript"

    def test_has_cached_transcript(self, temp_cache_dir):
        """Test checking if transcript exists in cache."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        # Should not exist initially
        assert not cache.has_cached_transcript("test123")

        # Cache a transcript
        cache.cache_transcript(video_id="test123", transcript="Test transcript")

        # Should exist now
        assert cache.has_cached_transcript("test123")

    def test_get_cached_transcript(self, temp_cache_dir):
        """Test retrieving cached transcript."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        # Cache a transcript
        cache.cache_transcript(
            video_id="test123",
            transcript="Test transcript",
            video_metadata={"title": "Test Video"},
        )

        # Retrieve the transcript
        transcript = cache.get_cached_transcript("test123")

        assert transcript is not None
        assert transcript == "Test transcript"

    def test_multiple_summaries_same_video(self, temp_cache_dir):
        """Test caching multiple summaries for the same video with different parameters."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        # Cache summary with OpenAI
        cache.cache_summary(
            video_id="test123",
            summary="OpenAI summary",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        # Cache summary with Anthropic
        cache.cache_summary(
            video_id="test123",
            summary="Anthropic summary",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            chunking_strategy="large",
            temperature=0.5,
        )

        # Both should exist
        openai_entry = cache.get_cached_summary(
            video_id="test123",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        anthropic_entry = cache.get_cached_summary(
            video_id="test123",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            chunking_strategy="large",
            temperature=0.5,
        )

        assert openai_entry.summary == "OpenAI summary"
        assert anthropic_entry.summary == "Anthropic summary"

    def test_clear_video_cache(self, temp_cache_dir):
        """Test clearing cache for a specific video."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        # Cache summary and transcript
        cache.cache_summary(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        cache.cache_transcript(video_id="test123", transcript="Test transcript")

        # Verify they exist
        assert cache.has_cached_summary(
            video_id="test123",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )
        assert cache.has_cached_transcript("test123")

        # Clear cache for this video
        removed_count = cache.clear_cache("test123")
        assert removed_count > 0

        # Should no longer exist
        assert not cache.has_cached_summary(
            video_id="test123",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )
        assert not cache.has_cached_transcript("test123")

    def test_get_cache_stats(self, temp_cache_dir):
        """Test getting cache statistics."""
        cache = SummaryCache(cache_dir=temp_cache_dir)

        # Cache some data
        cache.cache_summary(
            video_id="test123",
            summary="Test summary 1",
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        cache.cache_summary(
            video_id="test456",
            summary="Test summary 2",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            chunking_strategy="large",
            temperature=0.5,
        )

        cache.cache_transcript(video_id="test123", transcript="Test transcript")

        stats = cache.get_cache_stats()

        assert stats["total_videos"] == 2
        assert stats["cached_summaries"] == 2
        assert stats["cached_transcripts"] == 1
        assert "total_size_mb" in stats

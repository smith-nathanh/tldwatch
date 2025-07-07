"""
Tests for the caching functionality.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tldwatch.core.summarizer import Summarizer
from tldwatch.utils.cache import (
    CacheEntry,
    SummaryCache,
    TranscriptCacheEntry,
    clear_cache,
    get_cache,
    get_cache_stats,
    get_cached_transcript,
    has_cached_transcript,
)


class TestCacheEntry:
    """Test CacheEntry dataclass"""

    def test_to_dict(self):
        """Test converting cache entry to dictionary"""
        entry = CacheEntry(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4",
            chunking_strategy="standard",
            temperature=0.7,
            timestamp=1234567890.0,
            video_metadata={"title": "Test Video"},
        )

        result = entry.to_dict()
        assert result["video_id"] == "test123"
        assert result["summary"] == "Test summary"
        assert result["provider"] == "openai"
        assert result["video_metadata"]["title"] == "Test Video"

    def test_from_dict(self):
        """Test creating cache entry from dictionary"""
        data = {
            "video_id": "test123",
            "summary": "Test summary",
            "provider": "openai",
            "model": "gpt-4",
            "chunking_strategy": "standard",
            "temperature": 0.7,
            "timestamp": 1234567890.0,
            "video_metadata": {"title": "Test Video"},
        }

        entry = CacheEntry.from_dict(data)
        assert entry.video_id == "test123"
        assert entry.summary == "Test summary"
        assert entry.provider == "openai"
        assert entry.video_metadata["title"] == "Test Video"


class TestTranscriptCacheEntry:
    """Test TranscriptCacheEntry dataclass"""

    def test_to_dict(self):
        """Test converting transcript cache entry to dictionary"""
        entry = TranscriptCacheEntry(
            video_id="test123",
            transcript="This is a test transcript with lots of text content.",
            timestamp=1234567890.0,
            video_metadata={"title": "Test Video"},
        )

        result = entry.to_dict()
        assert result["video_id"] == "test123"
        assert (
            result["transcript"]
            == "This is a test transcript with lots of text content."
        )
        assert result["timestamp"] == 1234567890.0
        assert result["video_metadata"]["title"] == "Test Video"

    def test_from_dict(self):
        """Test creating transcript cache entry from dictionary"""
        data = {
            "video_id": "test123",
            "transcript": "This is a test transcript with lots of text content.",
            "timestamp": 1234567890.0,
            "video_metadata": {"title": "Test Video"},
        }

        entry = TranscriptCacheEntry.from_dict(data)
        assert entry.video_id == "test123"
        assert (
            entry.transcript == "This is a test transcript with lots of text content."
        )
        assert entry.timestamp == 1234567890.0
        assert entry.video_metadata["title"] == "Test Video"


class TestSummaryCache:
    """Test SummaryCache functionality"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create cache instance with temporary directory"""
        return SummaryCache(temp_cache_dir)

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that cache initialization creates directory"""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        SummaryCache(str(cache_dir))
        assert cache_dir.exists()

    def test_cache_summary(self, cache):
        """Test caching a summary"""
        cache.cache_summary(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4",
            chunking_strategy="standard",
            temperature=0.7,
            video_metadata={"title": "Test Video"},
        )

        # Check that cache file was created
        cache_file = cache._get_cache_file("test123")
        assert cache_file.exists()

        # Check file contents
        with open(cache_file, "r") as f:
            data = json.load(f)

        assert data["video_id"] == "test123"
        assert data["summary"] == "Test summary"
        assert data["provider"] == "openai"

    def test_get_cached_summary(self, cache):
        """Test retrieving cached summary"""
        # First cache a summary
        cache.cache_summary(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4",
            chunking_strategy="standard",
            temperature=0.7,
        )

        # Then retrieve it
        entry = cache.get_cached_summary("test123")
        assert entry is not None
        assert entry.video_id == "test123"
        assert entry.summary == "Test summary"
        assert entry.provider == "openai"

    def test_get_nonexistent_cache(self, cache):
        """Test retrieving non-existent cache entry"""
        entry = cache.get_cached_summary("nonexistent")
        assert entry is None

    def test_has_cached_summary(self, cache):
        """Test checking if cache exists"""
        # Should not exist initially
        assert not cache.has_cached_summary("test123")

        # Cache a summary
        cache.cache_summary(
            video_id="test123",
            summary="Test summary",
            provider="openai",
            model="gpt-4",
            chunking_strategy="standard",
            temperature=0.7,
        )

        # Should exist now
        assert cache.has_cached_summary("test123")

        # Test parameter matching
        assert cache.has_cached_summary("test123", provider="openai")
        assert cache.has_cached_summary("test123", model="gpt-4")
        assert not cache.has_cached_summary("test123", provider="anthropic")

    def test_clear_specific_cache(self, cache):
        """Test clearing specific video cache"""
        # Cache two summaries
        cache.cache_summary("test1", "Summary 1", "openai", "gpt-4", "standard", 0.7)
        cache.cache_summary("test2", "Summary 2", "openai", "gpt-4", "standard", 0.7)

        # Clear one
        removed = cache.clear_cache("test1")
        assert removed == 1
        assert not cache.has_cached_summary("test1")
        assert cache.has_cached_summary("test2")

    def test_clear_all_cache(self, cache):
        """Test clearing all cache"""
        # Cache multiple summaries
        cache.cache_summary("test1", "Summary 1", "openai", "gpt-4", "standard", 0.7)
        cache.cache_summary("test2", "Summary 2", "openai", "gpt-4", "standard", 0.7)
        cache.cache_summary("test3", "Summary 3", "openai", "gpt-4", "standard", 0.7)

        # Clear all
        removed = cache.clear_cache()
        assert removed == 3
        assert not cache.has_cached_summary("test1")
        assert not cache.has_cached_summary("test2")
        assert not cache.has_cached_summary("test3")

    def test_list_cached_videos(self, cache):
        """Test listing cached videos"""
        # Should be empty initially
        assert cache.list_cached_videos() == []

        # Cache some summaries
        cache.cache_summary("video1", "Summary 1", "openai", "gpt-4", "standard", 0.7)
        cache.cache_summary("video2", "Summary 2", "openai", "gpt-4", "standard", 0.7)

        # Should list videos
        cached = cache.list_cached_videos()
        assert len(cached) == 2
        assert "video1" in cached
        assert "video2" in cached

    def test_get_cache_stats(self, cache):
        """Test cache statistics"""
        # Cache a summary
        cache.cache_summary(
            "test123", "Test summary", "openai", "gpt-4", "standard", 0.7
        )

        stats = cache.get_cache_stats()
        assert stats["total_videos"] == 1
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] > 0
        assert "cache_dir" in stats

    def test_cleanup_old_cache(self, cache):
        """Test cleaning up old cache entries"""
        # Create old cache entry
        old_time = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
        cache.cache_summary(
            "old_video", "Old summary", "openai", "gpt-4", "standard", 0.7
        )

        # Manually modify timestamp to be old
        cache_file = cache._get_cache_file("old_video")
        with open(cache_file, "r") as f:
            data = json.load(f)
        data["timestamp"] = old_time
        with open(cache_file, "w") as f:
            json.dump(data, f)

        # Create new cache entry
        cache.cache_summary(
            "new_video", "New summary", "openai", "gpt-4", "standard", 0.7
        )

        # Cleanup old entries (30 days)
        removed = cache.cleanup_old_cache(30)
        assert removed == 1
        assert not cache.has_cached_summary("old_video")
        assert cache.has_cached_summary("new_video")

    def test_cache_transcript(self, cache):
        """Test caching a transcript"""
        transcript_text = "This is a test transcript with lots of text content."
        cache.cache_transcript(
            video_id="test123",
            transcript=transcript_text,
            video_metadata={"title": "Test Video"},
        )

        # Check that transcript cache file was created
        transcript_file = cache._get_transcript_cache_file("test123")
        assert transcript_file.exists()

        # Check file contents
        with open(transcript_file, "r") as f:
            data = json.load(f)

        assert data["video_id"] == "test123"
        assert data["transcript"] == transcript_text
        assert data["video_metadata"]["title"] == "Test Video"

    def test_get_cached_transcript(self, cache):
        """Test retrieving cached transcript"""
        transcript_text = "This is a test transcript with lots of text content."

        # First cache a transcript
        cache.cache_transcript(
            video_id="test123",
            transcript=transcript_text,
            video_metadata={"title": "Test Video"},
        )

        # Then retrieve it
        cached_transcript = cache.get_cached_transcript("test123")
        assert cached_transcript is not None
        assert cached_transcript == transcript_text

    def test_get_nonexistent_transcript(self, cache):
        """Test retrieving non-existent transcript cache entry"""
        transcript = cache.get_cached_transcript("nonexistent")
        assert transcript is None

    def test_has_cached_transcript(self, cache):
        """Test checking if transcript cache exists"""
        # Should not exist initially
        assert not cache.has_cached_transcript("test123")

        # Cache a transcript
        cache.cache_transcript(
            video_id="test123",
            transcript="Test transcript content",
        )

        # Should exist now
        assert cache.has_cached_transcript("test123")

    def test_clear_specific_cache_with_transcript(self, cache):
        """Test clearing specific video cache including transcript"""
        # Cache both summary and transcript
        cache.cache_summary("test1", "Summary 1", "openai", "gpt-4", "standard", 0.7)
        cache.cache_transcript("test1", "Transcript 1")
        cache.cache_summary("test2", "Summary 2", "openai", "gpt-4", "standard", 0.7)

        # Clear one video (should remove both summary and transcript)
        removed = cache.clear_cache("test1")
        assert removed == 2  # Both summary and transcript files
        assert not cache.has_cached_summary("test1")
        assert not cache.has_cached_transcript("test1")
        assert cache.has_cached_summary("test2")

    def test_clear_all_cache_with_transcripts(self, cache):
        """Test clearing all cache including transcripts"""
        # Cache multiple summaries and transcripts
        cache.cache_summary("test1", "Summary 1", "openai", "gpt-4", "standard", 0.7)
        cache.cache_transcript("test1", "Transcript 1")
        cache.cache_summary("test2", "Summary 2", "openai", "gpt-4", "standard", 0.7)
        cache.cache_transcript("test2", "Transcript 2")

        # Clear all
        removed = cache.clear_cache()
        assert removed == 4  # 2 summaries + 2 transcripts
        assert not cache.has_cached_summary("test1")
        assert not cache.has_cached_transcript("test1")
        assert not cache.has_cached_summary("test2")
        assert not cache.has_cached_transcript("test2")

    def test_list_cached_videos_with_transcripts(self, cache):
        """Test listing cached videos including transcript-only videos"""
        # Should be empty initially
        assert cache.list_cached_videos() == []

        # Cache some summaries and transcripts
        cache.cache_summary("video1", "Summary 1", "openai", "gpt-4", "standard", 0.7)
        cache.cache_transcript("video2", "Transcript 2")  # transcript only
        cache.cache_summary("video3", "Summary 3", "openai", "gpt-4", "standard", 0.7)
        cache.cache_transcript("video3", "Transcript 3")  # both

        # Should list all videos
        cached = cache.list_cached_videos()
        assert len(cached) == 3
        assert "video1" in cached
        assert "video2" in cached
        assert "video3" in cached

    def test_get_cache_stats_with_transcripts(self, cache):
        """Test cache statistics with transcripts"""
        # Cache summaries and transcripts
        cache.cache_summary("test1", "Test summary", "openai", "gpt-4", "standard", 0.7)
        cache.cache_transcript("test1", "Test transcript content")
        cache.cache_transcript("test2", "Another transcript")  # transcript only

        stats = cache.get_cache_stats()
        assert stats["total_videos"] == 2
        assert stats["cached_summaries"] == 1
        assert stats["cached_transcripts"] == 2
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] > 0
        assert "cache_dir" in stats

    # ...existing code...


class TestCacheIntegration:
    """Test cache integration with Summarizer"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.mark.asyncio
    async def test_summarizer_uses_cache(self, temp_cache_dir):
        """Test that summarizer uses cache when enabled"""
        with patch("tldwatch.core.user_config.get_user_config") as mock_config:
            # Mock user config to use temp cache dir
            mock_config.return_value.is_cache_enabled.return_value = True
            mock_config.return_value.get_cache_dir.return_value = temp_cache_dir

            with patch(
                "tldwatch.core.summarizer.YouTubeTranscriptApi"
            ) as mock_transcript:
                # Mock transcript API
                mock_transcript.get_transcript.return_value = [
                    {"text": "This is a test transcript."}
                ]

                with patch(
                    "tldwatch.core.providers.unified_provider.UnifiedProvider"
                ) as mock_provider:
                    # Mock provider
                    mock_instance = mock_provider.return_value
                    mock_instance.config.name = "openai"
                    mock_instance.model = "gpt-4"
                    mock_instance.chunking_strategy.value = "standard"
                    mock_instance.temperature = 0.7
                    mock_instance.generate_summary = AsyncMock(
                        return_value="Generated summary"
                    )

                    summarizer = Summarizer()

                    # First call should generate summary
                    summary1 = await summarizer.summarize("dQw4w9WgXcQ", use_cache=True)
                    assert summary1 == "Generated summary"
                    assert mock_instance.generate_summary.call_count == 1

                    # Second call with same parameters should use cache
                    summary2 = await summarizer.summarize("dQw4w9WgXcQ", use_cache=True)
                    assert summary2 == "Generated summary"
                    assert (
                        mock_instance.generate_summary.call_count == 1
                    )  # Should not be called again

    @pytest.mark.asyncio
    async def test_summarizer_uses_transcript_cache(self, temp_cache_dir):
        """Test that summarizer uses transcript cache when available"""
        with patch("tldwatch.core.user_config.get_user_config") as mock_config:
            # Mock user config to use temp cache dir
            mock_config.return_value.is_cache_enabled.return_value = True
            mock_config.return_value.get_cache_dir.return_value = temp_cache_dir

            with patch(
                "tldwatch.core.summarizer.YouTubeTranscriptApi"
            ) as mock_transcript:
                # Mock transcript API
                mock_transcript.get_transcript.return_value = [
                    {"text": "This is a test transcript."}
                ]

                with patch(
                    "tldwatch.core.providers.unified_provider.UnifiedProvider"
                ) as mock_provider:
                    # Mock provider
                    mock_instance = mock_provider.return_value
                    mock_instance.config.name = "openai"
                    mock_instance.model = "gpt-4"
                    mock_instance.chunking_strategy.value = "standard"
                    mock_instance.temperature = 0.7
                    mock_instance.generate_summary = AsyncMock(
                        return_value="Generated summary"
                    )

                    summarizer = Summarizer()

                    # First call should fetch transcript from YouTube and cache it
                    summary1 = await summarizer.summarize("dQw4w9WgXcQ", use_cache=True)
                    assert summary1 == "Generated summary"
                    assert mock_transcript.get_transcript.call_count == 1

                    # Second call should use cached transcript, not fetch from YouTube
                    summary2 = await summarizer.summarize("dQw4w9WgXcQ", use_cache=True)
                    assert summary2 == "Generated summary"
                    # Transcript should not be fetched again due to summary cache
                    assert mock_transcript.get_transcript.call_count == 1

                    # Test with different parameters that require new summary but cached transcript
                    summary3 = await summarizer.summarize(
                        "dQw4w9WgXcQ", use_cache=True, temperature=0.5
                    )
                    assert summary3 == "Generated summary"
                    # Transcript should still not be fetched due to transcript cache
                    assert mock_transcript.get_transcript.call_count == 1

    def test_global_cache_functions(self, temp_cache_dir):
        """Test global cache utility functions"""
        # Test get_cache_stats
        stats = get_cache_stats(temp_cache_dir)
        assert stats["total_videos"] == 0

        # Create a cache entry
        cache = get_cache(temp_cache_dir)
        cache.cache_summary(
            "test123", "Test summary", "openai", "gpt-4", "standard", 0.7
        )
        cache.cache_transcript("test123", "Test transcript content")

        # Test updated stats
        stats = get_cache_stats(temp_cache_dir)
        assert stats["total_videos"] == 1
        assert stats["cached_summaries"] == 1
        assert stats["cached_transcripts"] == 1

        # Test transcript utility functions
        assert has_cached_transcript("test123", temp_cache_dir)
        assert not has_cached_transcript("nonexistent", temp_cache_dir)

        transcript = get_cached_transcript("test123", temp_cache_dir)
        assert transcript == "Test transcript content"

        # Test clear_cache function
        removed = clear_cache("test123", temp_cache_dir)
        assert removed == 2  # Both summary and transcript

        stats = get_cache_stats(temp_cache_dir)
        assert stats["total_videos"] == 0

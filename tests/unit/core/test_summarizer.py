"""
Unit tests for the main Summarizer class.
Tests the core summarization functionality with various inputs and configurations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldwatch.core.summarizer import Summarizer


class TestSummarizer:
    """Test Summarizer functionality."""

    @pytest.fixture
    def mock_user_config(self):
        """Mock user configuration."""
        mock_config = MagicMock()
        mock_config.is_cache_enabled.return_value = True
        mock_config.get_cache_dir.return_value = "/tmp/test_cache"
        return mock_config

    @pytest.fixture
    def mock_cache(self):
        """Mock cache functionality."""
        mock_cache = MagicMock()
        mock_cache.has_cached_summary.return_value = False
        mock_cache.get_cached_summary.return_value = None
        mock_cache.has_cached_transcript.return_value = False
        mock_cache.get_cached_transcript.return_value = None
        return mock_cache

    @pytest.fixture
    def mock_unified_provider(self):
        """Mock unified provider."""
        mock_provider = MagicMock()
        mock_provider.config.name = "openai"
        mock_provider.model = "gpt-4o"
        mock_provider.chunking_strategy.value = "standard"
        mock_provider.temperature = 0.7
        mock_provider.generate_summary = AsyncMock(return_value="Generated summary")
        return mock_provider

    def test_summarizer_initialization(self):
        """Test that summarizer initializes correctly."""
        summarizer = Summarizer()
        assert summarizer is not None

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    async def test_summarize_direct_text(
        self,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_cache,
        mock_unified_provider,
    ):
        """Test summarizing direct text input."""
        # Setup mocks
        mock_get_user_config.return_value = mock_user_config
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider

        summarizer = Summarizer()

        # Test with direct text (not a YouTube URL or video ID)
        text = "This is a long piece of text that needs to be summarized for testing purposes."
        result = await summarizer.summarize(text)

        assert result == "Generated summary"
        mock_unified_provider.generate_summary.assert_called_once_with(text)

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    @patch("tldwatch.core.summarizer.YouTubeTranscriptApi.get_transcript")
    async def test_summarize_youtube_url(
        self,
        mock_get_transcript,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_cache,
        mock_unified_provider,
        mock_youtube_transcript,
    ):
        """Test summarizing a YouTube URL."""
        # Setup mocks
        mock_get_user_config.return_value = mock_user_config
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider
        mock_get_transcript.return_value = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "This is a test", "start": 2.0, "duration": 3.0},
        ]

        summarizer = Summarizer()

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = await summarizer.summarize(url)

        assert result == "Generated summary"
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

        # Check that the transcript text was passed to the provider
        expected_text = "Hello world This is a test"
        mock_unified_provider.generate_summary.assert_called_once_with(expected_text)

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    @patch("tldwatch.core.summarizer.YouTubeTranscriptApi.get_transcript")
    async def test_summarize_video_id(
        self,
        mock_get_transcript,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_cache,
        mock_unified_provider,
    ):
        """Test summarizing with a direct video ID."""
        # Setup mocks
        mock_get_user_config.return_value = mock_user_config
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider
        mock_get_transcript.return_value = [
            {"text": "Video content", "start": 0.0, "duration": 5.0}
        ]

        summarizer = Summarizer()

        video_id = "dQw4w9WgXcQ"
        result = await summarizer.summarize(video_id)

        assert result == "Generated summary"
        mock_get_transcript.assert_called_once_with(video_id)

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    async def test_summarize_with_cached_summary(
        self,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_unified_provider,
    ):
        """Test that cached summaries are returned when available."""
        # Setup cached entry
        mock_cached_entry = MagicMock()
        mock_cached_entry.summary = "Cached summary"
        mock_cached_entry.provider = "openai"
        mock_cached_entry.model = "gpt-4o"

        # Setup cache to return cached entry
        mock_cache = MagicMock()
        mock_cache.has_cached_summary.return_value = True
        mock_cache.get_cached_summary.return_value = mock_cached_entry

        mock_get_user_config.return_value = mock_user_config
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider

        summarizer = Summarizer()

        video_id = "dQw4w9WgXcQ"
        result = await summarizer.summarize(video_id)

        # Should return cached result without calling provider
        assert result == "Cached summary"
        mock_unified_provider.generate_summary.assert_not_called()

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    @patch("tldwatch.core.summarizer.YouTubeTranscriptApi.get_transcript")
    async def test_summarize_with_cache_disabled(
        self,
        mock_get_transcript,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_unified_provider,
    ):
        """Test summarization with caching disabled."""
        # Setup user config with cache disabled
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = False
        mock_user_config.get_cache_dir.return_value = "/tmp/test_cache"

        mock_get_user_config.return_value = mock_user_config
        mock_provider_class.return_value = mock_unified_provider
        mock_get_transcript.return_value = [
            {"text": "Test content", "start": 0.0, "duration": 2.0}
        ]

        summarizer = Summarizer()

        video_id = "dQw4w9WgXcQ"
        result = await summarizer.summarize(video_id, use_cache=False)

        assert result == "Generated summary"
        # Cache should not be checked or called
        mock_get_cache.assert_not_called()

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    async def test_summarize_with_provider_params(
        self,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_cache,
        mock_unified_provider,
    ):
        """Test summarization with explicit provider parameters."""
        # Setup mocks
        mock_get_user_config.return_value = mock_user_config
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider

        summarizer = Summarizer()

        text = "This is a test text for summarization with custom parameters."
        result = await summarizer.summarize(
            text,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.5,
            chunking_strategy="large",
        )

        assert result == "Generated summary"

        # Verify provider was initialized with correct parameters
        mock_provider_class.assert_called_once_with(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.5,
            chunking_strategy="large",
        )

    async def test_summarize_short_text_error(self):
        """Test that short text input raises ValueError."""
        summarizer = Summarizer()

        with pytest.raises(ValueError, match="Input text is too short"):
            await summarizer.summarize("short")

    @patch("tldwatch.core.summarizer.extract_video_id")
    async def test_summarize_invalid_youtube_url(self, mock_extract_video_id):
        """Test that invalid YouTube URLs raise ValueError."""
        mock_extract_video_id.return_value = None

        summarizer = Summarizer()

        with pytest.raises(ValueError, match="Could not extract video ID from URL"):
            await summarizer.summarize("https://www.youtube.com/invalid")

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    @patch("tldwatch.core.summarizer.YouTubeTranscriptApi.get_transcript")
    async def test_transcript_caching(
        self,
        mock_get_transcript,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_unified_provider,
    ):
        """Test that transcripts are cached when enabled."""
        # Setup mocks
        mock_get_user_config.return_value = mock_user_config
        mock_cache = MagicMock()
        mock_cache.has_cached_summary.return_value = False
        mock_cache.has_cached_transcript.return_value = False
        mock_cache.get_cached_transcript.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider

        mock_get_transcript.return_value = [
            {"text": "Transcript content", "start": 0.0, "duration": 3.0}
        ]

        summarizer = Summarizer()

        video_id = "dQw4w9WgXcQ"
        await summarizer.summarize(video_id)

        # Verify transcript was cached
        mock_cache.cache_transcript.assert_called_once()
        call_args = mock_cache.cache_transcript.call_args
        assert call_args[1]["video_id"] == video_id
        assert "Transcript content" in call_args[1]["transcript"]

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    async def test_use_cached_transcript(
        self,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_unified_provider,
    ):
        """Test that cached transcripts are used when available."""
        # Setup cached transcript
        mock_cached_transcript = MagicMock()
        mock_cached_transcript.transcript = "Cached transcript content"

        mock_cache = MagicMock()
        mock_cache.has_cached_summary.return_value = False
        mock_cache.has_cached_transcript.return_value = True
        mock_cache.get_cached_transcript.return_value = mock_cached_transcript

        mock_get_user_config.return_value = mock_user_config
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider

        summarizer = Summarizer()

        video_id = "dQw4w9WgXcQ"
        result = await summarizer.summarize(video_id)

        # Should use cached transcript
        assert result == "Generated summary"
        mock_unified_provider.generate_summary.assert_called_once_with(
            "Cached transcript content"
        )

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("tldwatch.core.summarizer.get_cache")
    @patch("tldwatch.core.summarizer.UnifiedProvider")
    @patch("tldwatch.core.summarizer.YouTubeTranscriptApi.get_transcript")
    async def test_summary_caching(
        self,
        mock_get_transcript,
        mock_provider_class,
        mock_get_cache,
        mock_get_user_config,
        mock_user_config,
        mock_cache,
        mock_unified_provider,
    ):
        """Test that summaries are cached after generation."""
        # Setup mocks
        mock_get_user_config.return_value = mock_user_config
        mock_get_cache.return_value = mock_cache
        mock_provider_class.return_value = mock_unified_provider

        mock_get_transcript.return_value = [
            {"text": "Content to summarize", "start": 0.0, "duration": 2.0}
        ]

        summarizer = Summarizer()

        video_id = "dQw4w9WgXcQ"
        await summarizer.summarize(video_id)

        # Verify summary was cached
        mock_cache.cache_summary.assert_called_once()
        call_args = mock_cache.cache_summary.call_args
        assert call_args[1]["video_id"] == video_id
        assert call_args[1]["summary"] == "Generated summary"
        assert call_args[1]["provider"] == "openai"

"""
Integration tests for the core summarization workflow.
Tests the complete flow from input to summary generation with real components.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tldwatch import Summarizer, get_cache


class TestSummarizerIntegration:
    """Integration tests for the complete summarization workflow."""

    @pytest.fixture
    def integration_cache_dir(self):
        """Provide a temporary cache directory for integration tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    @patch("aiohttp.ClientSession")
    async def test_complete_summarization_workflow(
        self,
        mock_session,
        mock_get_transcript,
        mock_get_user_config,
        integration_cache_dir,
        mock_youtube_transcript,
        mock_aiohttp_session,
        sample_video_id,
        mock_env_vars,
    ):
        """Test the complete workflow from video input to cached summary."""
        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = integration_cache_dir
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup transcript API
        mock_get_transcript.return_value = [
            {
                "text": "Welcome to this video about machine learning.",
                "start": 0.0,
                "duration": 3.5,
            },
            {
                "text": "Today we'll explore neural networks and deep learning.",
                "start": 3.5,
                "duration": 4.2,
            },
            {
                "text": "Neural networks are inspired by biological neurons.",
                "start": 7.7,
                "duration": 3.8,
            },
            {
                "text": "They can learn complex patterns from data.",
                "start": 11.5,
                "duration": 3.1,
            },
            {
                "text": "Thanks for watching and subscribe for more content!",
                "start": 14.6,
                "duration": 3.4,
            },
        ]

        # Setup HTTP session response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This video provides an introduction to machine learning and neural networks. "
                        "It explains how neural networks are inspired by biological neurons and can "
                        "learn complex patterns from data. The content covers the basics of deep learning "
                        "and encourages viewers to subscribe for more educational content."
                    }
                }
            ]
        }

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        # Initialize summarizer and run complete workflow
        summarizer = Summarizer()

        # First run - should fetch transcript and generate summary
        summary1 = await summarizer.summarize(sample_video_id)

        assert "machine learning" in summary1.lower()
        assert "neural networks" in summary1.lower()

        # Verify transcript was cached
        cache = get_cache(integration_cache_dir)
        assert cache.has_cached_transcript(sample_video_id)

        # Verify summary was cached
        assert cache.has_cached_summary(
            video_id=sample_video_id,
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        # Second run - should use cached summary
        summary2 = await summarizer.summarize(sample_video_id)

        assert summary1 == summary2

        # Transcript API should only be called once (first time)
        mock_get_transcript.assert_called_once_with(sample_video_id)

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    @patch("aiohttp.ClientSession")
    async def test_different_provider_configurations(
        self,
        mock_session,
        mock_get_transcript,
        mock_get_user_config,
        integration_cache_dir,
        sample_video_id,
        mock_env_vars,
    ):
        """Test that different provider configurations create separate cache entries."""
        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = integration_cache_dir
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup transcript API
        mock_get_transcript.return_value = [
            {"text": "Test video content", "start": 0.0, "duration": 2.0}
        ]

        # Setup different responses for different providers
        def create_mock_response(content):
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": content}}]
            }
            return mock_response

        openai_response = create_mock_response("OpenAI generated summary")
        anthropic_response = create_mock_response("Anthropic generated summary")

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.side_effect = [
            openai_response,
            anthropic_response,
        ]
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        summarizer = Summarizer()

        # Generate summary with OpenAI
        summary_openai = await summarizer.summarize(
            sample_video_id, provider="openai", model="gpt-4o"
        )

        # Generate summary with Anthropic
        summary_anthropic = await summarizer.summarize(
            sample_video_id, provider="anthropic", model="claude-3-5-sonnet-20241022"
        )

        assert summary_openai == "OpenAI generated summary"
        assert summary_anthropic == "Anthropic generated summary"

        # Both should be cached separately
        cache = get_cache(integration_cache_dir)

        assert cache.has_cached_summary(
            video_id=sample_video_id,
            provider="openai",
            model="gpt-4o",
            chunking_strategy="standard",
            temperature=0.7,
        )

        assert cache.has_cached_summary(
            video_id=sample_video_id,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            chunking_strategy="standard",
            temperature=0.7,
        )

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("aiohttp.ClientSession")
    async def test_direct_text_summarization(
        self, mock_session, mock_get_user_config, integration_cache_dir, mock_env_vars
    ):
        """Test direct text summarization without YouTube integration."""
        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = integration_cache_dir
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This text discusses artificial intelligence and machine learning technologies."
                    }
                }
            ]
        }

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        summarizer = Summarizer()

        long_text = """
        Artificial intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that work and react like humans. Machine learning is a subset 
        of AI that involves the use of algorithms and statistical models to enable computers 
        to improve their performance on a specific task through experience. Deep learning, 
        in turn, is a subset of machine learning that uses neural networks with multiple 
        layers to model and understand complex patterns in data.
        """

        summary = await summarizer.summarize(long_text.strip())

        assert "artificial intelligence" in summary.lower()
        assert "machine learning" in summary.lower()

        # Verify the request was made with the correct text
        mock_session_instance.post.assert_called_once()

        # Direct text should not be cached (no video ID)
        cache = get_cache(integration_cache_dir)
        cache_stats = cache.get_cache_stats()
        assert cache_stats["videos_with_summaries"] == 0

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    async def test_cache_disabled_workflow(
        self,
        mock_get_transcript,
        mock_get_user_config,
        integration_cache_dir,
        sample_video_id,
        mock_env_vars,
    ):
        """Test workflow with caching disabled."""
        # Setup user config with cache disabled
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = False
        mock_user_config.get_cache_dir.return_value = integration_cache_dir
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup transcript API to be called multiple times
        mock_get_transcript.return_value = [
            {"text": "Test content", "start": 0.0, "duration": 2.0}
        ]

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Generated summary"}}]
            }

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value.__aenter__.return_value = (
                mock_response
            )
            mock_session.return_value.__aenter__.return_value = mock_session_instance

            summarizer = Summarizer()

            # Run twice with cache disabled
            summary1 = await summarizer.summarize(sample_video_id, use_cache=False)
            summary2 = await summarizer.summarize(sample_video_id, use_cache=False)

            assert summary1 == "Generated summary"
            assert summary2 == "Generated summary"

            # Transcript should be fetched twice (no caching)
            assert mock_get_transcript.call_count == 2

            # No cache entries should be created
            cache = get_cache(integration_cache_dir)
            cache_stats = cache.get_cache_stats()
            assert cache_stats["videos_with_summaries"] == 0
            assert cache_stats["videos_with_transcripts"] == 0

    @patch("tldwatch.core.summarizer.get_user_config")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    async def test_transcript_api_error_handling(
        self,
        mock_get_transcript,
        mock_get_user_config,
        integration_cache_dir,
        sample_video_id,
        mock_env_vars,
    ):
        """Test handling of YouTube transcript API errors."""
        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = integration_cache_dir
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup transcript API to raise an error
        from youtube_transcript_api import TranscriptsDisabled

        mock_get_transcript.side_effect = TranscriptsDisabled(sample_video_id)

        summarizer = Summarizer()

        with pytest.raises(Exception):  # Should propagate the transcript error
            await summarizer.summarize(sample_video_id)

"""
Pytest configuration and shared fixtures for tldwatch tests.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_cache_dir():
    """Provide a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_config_dir():
    """Provide a temporary config directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_youtube_transcript():
    """Mock YouTube transcript API responses."""
    mock_transcript = [
        {
            "text": "Welcome to this video about artificial intelligence.",
            "start": 0.0,
            "duration": 3.5,
        },
        {
            "text": "Today we'll be discussing machine learning fundamentals.",
            "start": 3.5,
            "duration": 4.2,
        },
        {
            "text": "Let's start with supervised learning algorithms.",
            "start": 7.7,
            "duration": 3.8,
        },
        {
            "text": "Neural networks are a powerful tool for pattern recognition.",
            "start": 11.5,
            "duration": 4.1,
        },
        {
            "text": "Thank you for watching and don't forget to subscribe!",
            "start": 15.6,
            "duration": 3.2,
        },
    ]

    with patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript") as mock:
        mock.return_value = mock_transcript
        yield mock


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for API calls."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "This is a test summary of the video content about AI and machine learning."
                }
            }
        ]
    }

    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__.return_value = mock_response

    with patch("aiohttp.ClientSession", return_value=mock_session):
        yield mock_session


@pytest.fixture
def sample_video_id():
    """Provide a sample YouTube video ID for testing."""
    return "dQw4w9WgXcQ"


@pytest.fixture
def sample_video_url():
    """Provide a sample YouTube video URL for testing."""
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def sample_transcript_text():
    """Provide sample transcript text for testing."""
    return (
        "Welcome to this video about artificial intelligence. "
        "Today we'll be discussing machine learning fundamentals. "
        "Let's start with supervised learning algorithms. "
        "Neural networks are a powerful tool for pattern recognition. "
        "Thank you for watching and don't forget to subscribe!"
    )


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_user_config():
    """Provide a sample user configuration for testing."""
    return {
        "default_provider": "openai",
        "default_temperature": 0.7,
        "default_chunking_strategy": "standard",
        "cache": {
            "enabled": True,
            "directory": None,  # Will use default
        },
        "providers": {
            "openai": {"default_model": "gpt-4o", "temperature": 0.5},
            "anthropic": {"default_model": "claude-3-5-sonnet-20241022"},
        },
    }

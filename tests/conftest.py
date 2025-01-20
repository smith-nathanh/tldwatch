# conftest.py
import os
from typing import Dict, Type
from unittest.mock import patch

import pytest

from tldwatch.core.providers.anthropic import AnthropicProvider
from tldwatch.core.providers.base import BaseProvider
from tldwatch.core.providers.cerebras import CerebrasProvider
from tldwatch.core.providers.deepseek import DeepSeekProvider
from tldwatch.core.providers.groq import GroqProvider
from tldwatch.core.providers.ollama import OllamaProvider
from tldwatch.core.providers.openai import OpenAIProvider

# Map of provider names to their classes
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "cerebras": CerebrasProvider,
    "deepseek": DeepSeekProvider,
    "ollama": OllamaProvider,
}


@pytest.fixture(autouse=True)
def mock_ollama_models():
    """Mock Ollama's available models check"""
    with patch(
        "tldwatch.core.providers.ollama.OllamaProvider._get_available_models"
    ) as mock:
        mock.return_value = ["llama3.1:8b"]
        yield


@pytest.fixture
def mock_config():
    """Configuration fixture for all tests"""
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "chunk_size": 4000,
        "chunk_overlap": 200,
        "temperature": 0.7,
        "use_full_context": False,
    }


@pytest.fixture
def sample_transcript():
    """Sample transcript data for summarization tests"""
    return (
        "This is a sample transcript with multiple sentences. "
        "It contains enough content to test chunking. "
        "The content should be meaningful enough to generate summaries. "
        "It should also be long enough to test context windows. "
    ) * 25  # Makes it long enough to test chunking


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for API keys. Automatically used in all tests."""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-proj-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
            "GROQ_API_KEY": "gsk-test-key",
            "CEREBRAS_API_KEY": "csk-test-key",
            "DEEPSEEK_API_KEY": "sk-test-key",
            "YOUTUBE_API_KEY": "yt-test-key",
        },
    ):
        yield


@pytest.fixture
def provider_instance(request):
    """Create provider instance for testing"""
    provider_name = request.param
    provider_class = PROVIDERS[provider_name]

    # Create instance with default configuration
    instance = provider_class(temperature=0.7)
    return instance


@pytest.fixture
def mock_youtube_api():
    """Mock YouTube API responses"""
    return {
        "items": [
            {
                "snippet": {
                    "title": "Test Video",
                    "description": "Test Description",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "channelTitle": "Test Channel",
                },
                "statistics": {"viewCount": "1000", "likeCount": "100"},
            }
        ]
    }


@pytest.fixture
def mock_successful_completion():
    """Mock successful API completion response - matches provider response processing"""
    return {
        "openai": "Summary",  # OpenAI provider extracts content from response
        "anthropic": "Summary",  # Anthropic provider extracts text from response
        "groq": "Summary",  # Groq provider extracts content
        "cerebras": "Summary",  # Cerebras returns generated text directly
        "deepseek": "Summary",  # DeepSeek provider extracts content
        "ollama": "Summary",  # Ollama returns response directly
    }


@pytest.fixture
def mock_auth_error():
    """Mock authentication error responses for each provider"""
    return {
        "openai": {"error": {"code": "invalid_api_key", "message": "Invalid API key"}},
        "anthropic": {
            "error": {"type": "authentication_error", "message": "Invalid API key"}
        },
        "groq": {
            "error": {"type": "authentication_error", "message": "Invalid API key"}
        },
        "cerebras": {"error": "Authentication failed", "status_code": 401},
        "deepseek": {
            "error": {"code": "invalid_api_key", "message": "Invalid API key"}
        },
        "ollama": {"error": "Authentication error", "status_code": 401},
    }


@pytest.fixture
def mock_rate_limit_error():
    """Mock rate limit error responses"""
    return {
        "openai": {
            "error": {"code": "rate_limit_exceeded", "message": "Rate limit exceeded"}
        },
        "anthropic": {
            "error": {"type": "rate_limit_error", "message": "Too many requests"}
        },
        "groq": {
            "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}
        },
        "cerebras": {"error": "Too many requests", "status_code": 429},
        "deepseek": {
            "error": {"code": "rate_limit_exceeded", "message": "Rate limit exceeded"}
        },
        "ollama": {"error": "Too many requests", "status_code": 429},
    }


@pytest.fixture
def mock_network_error():
    """Mock network error responses for timeout/connection issues"""
    return {
        provider: {"error": "Connection error", "status_code": 500}
        for provider in [
            "openai",
            "anthropic",
            "groq",
            "cerebras",
            "deepseek",
            "ollama",
        ]
    }

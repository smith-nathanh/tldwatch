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

# Provider-specific test configurations
PROVIDER_TEST_CONFIG = {
    "openai": {
        "default_model": "gpt-4o-mini",
        "context_window": 128000,
        "rate_limit": 3500,
    },
    "anthropic": {
        "default_model": "claude-3-sonnet",
        "context_window": 200000,
        "rate_limit": 5000,
    },
    "groq": {
        "default_model": "mixtral-8x7b-32768",
        "context_window": 32768,
        "rate_limit": 1000,
    },
    "cerebras": {
        "default_model": "llama3.1-8b",
        "context_window": 8192,
        "rate_limit": 1000,
    },
}


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


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for API keys"""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-test-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
            "GROQ_API_KEY": "gsk-test-key",
            "CEREBRAS_API_KEY": "cb-test-key",
            "DEEPSEEK_API_KEY": "ds-test-key",
            "YOUTUBE_API_KEY": "yt-test-key",
        },
    ):
        yield


@pytest.fixture
def provider_instance(request, mock_env_vars):
    """Create provider instance for testing"""
    provider_name = request.param  # We're getting the provider name string
    provider_class = PROVIDERS[provider_name]  # Look up the class using the name
    config = PROVIDER_TEST_CONFIG.get(provider_name, {})

    return provider_class(
        model=config.get("default_model", "test-model"), temperature=0.7
    )


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
    """Mock successful API completion response"""
    return {
        "openai": {
            "choices": [{"message": {"content": "Summary"}}],
            "usage": {"total_tokens": 100},
        },
        "anthropic": {
            "content": [{"text": "Summary"}],
            "usage": {"input_tokens": 50, "output_tokens": 50},
        },
        "groq": {
            "choices": [{"message": {"content": "Summary"}}],
            "usage": {"total_tokens": 100},
        },
        "cerebras": {"generated_text": "Summary", "token_count": 100},
        "deepseek": {
            "choices": [{"message": {"content": "Summary"}}],
            "usage": {"total_tokens": 100},
        },
        "ollama": {
            "response": "Summary",
            "total_duration": 1000,
            "load_duration": 100,
        },
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
            "error": {"code": "rate_limit_exceeded", "message": "Rate limit exceeded"}
        },
        "cerebras": {"error": "Too many requests", "status_code": 429},
    }


@pytest.fixture
def mock_auth_error():
    """Mock authentication error responses"""
    return {
        "openai": {"error": {"code": "invalid_api_key", "message": "Invalid API key"}},
        "anthropic": {
            "error": {"type": "authentication_error", "message": "Invalid API key"}
        },
        # Add other providers' auth error formats
    }


@pytest.fixture
def mock_network_error():
    """Mock network error responses for timeout/connection issues"""
    return {
        provider: {"error": "Connection error", "status_code": 500}
        for provider in PROVIDERS.keys()
    }

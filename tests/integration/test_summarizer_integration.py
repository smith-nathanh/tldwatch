import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from tldwatch.core.summarizer import Summarizer

# Test video ID that's unlikely to be taken down (e.g. a popular educational video)
TEST_VIDEO_ID = "jNQXAC9IVRw"  # "Me at the zoo" - First YouTube video
TEST_VIDEO_URL = f"https://www.youtube.com/watch?v={TEST_VIDEO_ID}"


def has_api_key(provider: str) -> bool:
    """Check if API key is available for the given provider"""
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None,  # Local provider doesn't need API key
    }
    env_var = env_vars.get(provider.lower())
    return env_var is None or os.environ.get(env_var) is not None


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "chunk_size": 4000,
        }
        json.dump(config, f)
        f.flush()
        yield Path(f.name)
        os.unlink(f.name)


@pytest_asyncio.fixture
async def summarizer():
    """Create a summarizer instance with test configuration"""
    summarizer = Summarizer(
        provider="openai", model="gpt-4o-mini", temperature=0.7, chunk_size=4000
    )
    yield summarizer
    await summarizer.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not has_api_key("openai"), reason="OpenAI API key not available")
async def test_end_to_end_video_summary(summarizer):
    """Test complete flow from video ID to summary"""
    summary = await summarizer.get_summary(video_id=TEST_VIDEO_ID)

    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


@pytest.mark.asyncio
@pytest.mark.skipif(not has_api_key("openai"), reason="OpenAI API key not available")
async def test_url_to_summary_flow(summarizer):
    """Test summarization using a YouTube URL"""
    summary = await summarizer.get_summary(url=TEST_VIDEO_URL)

    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


@pytest.mark.asyncio
@pytest.mark.skipif(not has_api_key("openai"), reason="OpenAI API key not available")
async def test_chunked_processing(summarizer):
    """Test processing of a longer video that requires chunking"""
    # Use a longer video that will definitely need chunking
    LONG_VIDEO_ID = "9bZkp7q19f0"  # "Gangnam Style" - Long enough to test chunking

    summary = await summarizer.get_summary(video_id=LONG_VIDEO_ID)

    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


@pytest.mark.asyncio
@pytest.mark.skipif(not has_api_key("openai"), reason="OpenAI API key not available")
async def test_export_summary(summarizer, tmp_path):
    """Test exporting summary to a file"""
    await summarizer.get_summary(video_id=TEST_VIDEO_ID)

    export_path = tmp_path / "test_summary.json"
    await summarizer.export_summary(str(export_path))

    assert export_path.exists()
    with open(export_path) as f:
        data = json.load(f)
        assert "summary" in data
        assert "transcript" in data
        assert "provider" in data
        assert "model" in data


@pytest.mark.asyncio
@pytest.mark.skipif(not has_api_key("openai"), reason="OpenAI API key not available")
async def test_direct_transcript_input(summarizer):
    """Test summarization of directly provided transcript text"""
    test_transcript = """
    This is a test transcript.
    It contains multiple lines of text.
    This should be summarized by the model.
    The summary should capture the key points.
    """

    summary = await summarizer.get_summary(transcript_text=test_transcript)

    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not has_api_key("openai") or not has_api_key("anthropic"),
    reason="API keys not available",
)
async def test_provider_switching():
    """Test switching between different providers"""
    providers = ["openai", "anthropic"]  # Add other providers as needed

    for provider in providers:
        if not has_api_key(provider):
            continue
        summarizer = Summarizer(provider=provider, temperature=0.7, chunk_size=4000)
        try:
            summary = await summarizer.get_summary(video_id=TEST_VIDEO_ID)
            assert summary is not None
            assert len(summary) > 0
        finally:
            await summarizer.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not has_api_key("openai"), reason="OpenAI API key not available")
async def test_rate_limiting(summarizer):
    """Test rate limiting behavior"""
    # Make multiple concurrent requests
    tasks = [summarizer.get_summary(video_id=TEST_VIDEO_ID) for _ in range(5)]

    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    # Check that we got some successful results
    successful = [s for s in summaries if isinstance(s, str)]
    assert len(successful) > 0


@pytest.mark.asyncio
async def test_error_handling(summarizer):
    """Test error handling for various scenarios"""
    # Test invalid video ID
    with pytest.raises(Exception):
        await summarizer.get_summary(video_id="invalid_id")

    # Test invalid URL
    with pytest.raises(Exception):
        await summarizer.get_summary(url="https://youtube.com/invalid")

    # Test empty transcript
    with pytest.raises(Exception):
        await summarizer.get_summary(transcript_text="")


@pytest.mark.asyncio
async def test_summarizer_with_mock_provider(monkeypatch):
    """Test summarizer with a mocked provider to verify test framework works"""
    from unittest.mock import AsyncMock, MagicMock

    # Mock the provider to return fake responses
    mock_provider = MagicMock()
    mock_provider.generate_summary = AsyncMock(return_value="This is a fake summary")
    mock_provider.max_concurrent_requests = 5
    mock_provider.rate_limit_config = MagicMock()
    mock_provider.rate_limit_config.requests_per_minute = 60
    mock_provider.rate_limit_config.max_retries = 3
    mock_provider.rate_limit_config.retry_delay = 1.0
    mock_provider.close = AsyncMock()

    # Create summarizer with mocked provider
    summarizer = Summarizer(
        provider="openai", model="gpt-4o-mini", temperature=0.7, chunk_size=4000
    )
    summarizer.provider = mock_provider

    try:
        # Test direct transcript input
        test_transcript = """
        This is a test transcript.
        It contains multiple lines of text.
        This should be summarized by the model.
        The summary should capture the key points.
        """

        summary = await summarizer.get_summary(transcript_text=test_transcript)

        assert summary is not None
        assert len(summary) > 0
        assert isinstance(summary, str)
        assert "fake summary" in summary.lower()

        # Verify the provider was called
        mock_provider.generate_summary.assert_called()

    finally:
        await summarizer.close()

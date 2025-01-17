import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from tldwatch.core.summarizer import Summarizer

# Test video ID that's unlikely to be taken down (e.g. a popular educational video)
TEST_VIDEO_ID = "jNQXAC9IVRw"  # "Me at the zoo" - First YouTube video
TEST_VIDEO_URL = f"https://www.youtube.com/watch?v={TEST_VIDEO_ID}"


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


@pytest.fixture
async def summarizer():
    """Create a summarizer instance with test configuration"""
    summarizer = Summarizer(
        provider="openai", model="gpt-4o-mini", temperature=0.7, chunk_size=4000
    )
    yield summarizer
    await summarizer.close()


@pytest.mark.asyncio
async def test_end_to_end_video_summary(summarizer):
    """Test complete flow from video ID to summary"""
    summary = await summarizer.get_summary(video_id=TEST_VIDEO_ID)

    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


@pytest.mark.asyncio
async def test_url_to_summary_flow(summarizer):
    """Test summarization using a YouTube URL"""
    summary = await summarizer.get_summary(url=TEST_VIDEO_URL)

    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


@pytest.mark.asyncio
async def test_chunked_processing(summarizer):
    """Test processing of a longer video that requires chunking"""
    # Use a longer video that will definitely need chunking
    LONG_VIDEO_ID = "9bZkp7q19f0"  # "Gangnam Style" - Long enough to test chunking

    summary = await summarizer.get_summary(video_id=LONG_VIDEO_ID)

    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


@pytest.mark.asyncio
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
async def test_provider_switching():
    """Test switching between different providers"""
    providers = ["openai", "anthropic"]  # Add other providers as needed

    for provider in providers:
        summarizer = Summarizer(provider=provider, temperature=0.7, chunk_size=4000)
        try:
            summary = await summarizer.get_summary(video_id=TEST_VIDEO_ID)
            assert summary is not None
            assert len(summary) > 0
        finally:
            await summarizer.close()


@pytest.mark.asyncio
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

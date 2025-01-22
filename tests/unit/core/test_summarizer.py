from unittest.mock import patch

import pytest

from tldwatch.core.summarizer import Summarizer


@pytest.fixture
def summarizer(mock_config, mock_env_vars):
    return Summarizer(**mock_config)


def test_transcript_chunking(summarizer, sample_transcript):
    """Test text chunking logic"""
    chunks = summarizer._split_into_chunks(
        sample_transcript, chunk_size=200, overlap=50
    )
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)

    # Test overlap
    if len(chunks) > 1:
        overlap = set(chunks[0][-50:]).intersection(set(chunks[1][:50]))
        assert len(overlap) > 0


@pytest.mark.parametrize(
    "mock_config", list(Summarizer.PROVIDERS.keys()), indirect=True
)
@pytest.mark.asyncio
async def test_chunk_processing(mock_config, mock_successful_completion):
    summarizer = Summarizer(**mock_config)
    chunk = "This is a test chunk."
    with patch.object(summarizer.provider, "generate_summary") as mock_generate:
        mock_generate.return_value = mock_successful_completion[mock_config["provider"]]
        result = await summarizer._process_chunk(chunk, 0)
        assert result == "Summary"


@pytest.mark.parametrize(
    "mock_config", list(Summarizer.PROVIDERS.keys()), indirect=True
)
@pytest.mark.asyncio
async def test_summary_combination(mock_config):
    summarizer = Summarizer(**mock_config)
    summarizer.transcript = "Test transcript"
    with patch.object(summarizer.provider, "generate_summary") as mock_generate:
        mock_generate.return_value = "Combined summary"
        result = await summarizer._generate_chunked_summary()
        assert result == "Combined summary"


@pytest.mark.parametrize(
    "mock_config", list(Summarizer.PROVIDERS.keys()), indirect=True
)
@pytest.mark.asyncio
async def test_youtube_integration(mock_config, mock_youtube_api):
    summarizer = Summarizer(**mock_config)
    summarizer.youtube_api_key = "test-key"

    with patch(
        "youtube_transcript_api.YouTubeTranscriptApi.get_transcript"
    ) as mock_transcript:
        mock_transcript.return_value = [{"text": "This is a test transcript."}]
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 200
            mock_get.return_value.__aenter__.return_value.json.return_value = (
                mock_youtube_api
            )
            with patch.object(summarizer.provider, "generate_summary") as mock_generate:
                mock_generate.return_value = "Test summary"
                await summarizer.get_summary(video_id="test_id")
                assert summarizer.transcript == "This is a test transcript."
                assert summarizer.metadata["title"] == "Test Video"


@pytest.mark.parametrize(
    "mock_config", list(Summarizer.PROVIDERS.keys()), indirect=True
)
@pytest.mark.asyncio
async def test_error_handling(mock_config):
    summarizer = Summarizer(**mock_config)
    with pytest.raises(ValueError):
        await summarizer.get_summary(video_id="")


@pytest.mark.parametrize(
    "mock_config", list(Summarizer.PROVIDERS.keys()), indirect=True
)
@pytest.mark.asyncio
async def test_full_context_handling(mock_config, sample_transcript):
    summarizer = Summarizer(**mock_config)
    summarizer.use_full_context = True

    # Test small input (should use full context)
    with patch.object(
        summarizer.provider, "generate_summary", return_value="Full summary"
    ):
        with patch.object(
            summarizer.provider, "count_tokens", return_value=100
        ) as mock_count:
            result = await summarizer.get_summary(transcript_text="Short text")
            assert result == "Full summary"
            mock_count.assert_called_once()

    # Test large input (should fall back to chunked)
    with patch.object(
        summarizer.provider, "generate_summary", return_value="Chunked summary"
    ):
        with patch.object(
            summarizer.provider, "count_tokens", return_value=200000
        ) as mock_count:
            result = await summarizer.get_summary(transcript_text=sample_transcript)
            assert result == "Chunked summary"
            mock_count.assert_called_once()

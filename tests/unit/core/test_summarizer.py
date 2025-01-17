from unittest.mock import patch

import pytest

from tldwatch.core.summarizer import Summarizer, SummarizerError


@pytest.fixture
def summarizer(mock_env_vars):
    return Summarizer()


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


@pytest.mark.asyncio
async def test_chunk_processing(summarizer, mock_successful_completion):
    """Test processing of individual chunks"""
    chunk = "This is a test chunk."
    with patch.object(summarizer.provider, "generate_summary") as mock_generate:
        mock_generate.return_value = mock_successful_completion[
            summarizer.provider_name
        ]
        result = await summarizer._process_chunk(chunk, 0)
        assert result == "Summary"


@pytest.mark.asyncio
async def test_summary_combination(summarizer, mock_successful_completion):
    """Test combining multiple chunk summaries"""
    summaries = ["Summary 1", "Summary 2", "Summary 3"]
    with patch.object(summarizer.provider, "generate_summary") as mock_generate:
        mock_generate.return_value = "Combined summary"
        result = await summarizer._generate_chunked_summary()
        assert result == "Combined summary"


@pytest.mark.asyncio
async def test_youtube_integration(summarizer, mock_youtube_api):
    """Test YouTube transcript and metadata fetching"""
    with patch(
        "youtube_transcript_api.YouTubeTranscriptApi.get_transcript"
    ) as mock_transcript:
        mock_transcript.return_value = [{"text": "This is a test transcript."}]

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 200
            mock_get.return_value.__aenter__.return_value.json.return_value = (
                mock_youtube_api
            )

            await summarizer.get_summary(video_id="test_id")

            assert summarizer.transcript == "This is a test transcript."
            assert summarizer.metadata["title"] == "Test Video"
            assert summarizer.metadata["channelTitle"] == "Test Channel"


@pytest.mark.asyncio
async def test_error_handling(summarizer):
    """Test error handling in summarizer"""
    # Test invalid video ID
    with pytest.raises(SummarizerError):
        await summarizer.get_summary(video_id="")

    # Test failed transcript fetch
    with patch(
        "youtube_transcript_api.YouTubeTranscriptApi.get_transcript"
    ) as mock_transcript:
        mock_transcript.side_effect = Exception("Transcript fetch failed")
        with pytest.raises(SummarizerError):
            await summarizer.get_summary(video_id="test_id")


@pytest.mark.asyncio
async def test_full_context_handling(summarizer, sample_transcript):
    """Test full context vs chunked processing decision"""
    summarizer.use_full_context = True

    # Test small input (should use full context)
    with patch.object(summarizer, "_generate_full_summary") as mock_full:
        mock_full.return_value = "Full summary"
        result = await summarizer.get_summary(transcript_text="Short text")
        assert result == "Full summary"

    # Test large input (should fall back to chunked)
    long_transcript = sample_transcript * 100
    with patch.object(summarizer, "_generate_chunked_summary") as mock_chunked:
        mock_chunked.return_value = "Chunked summary"
        result = await summarizer.get_summary(transcript_text=long_transcript)
        assert result == "Chunked summary"

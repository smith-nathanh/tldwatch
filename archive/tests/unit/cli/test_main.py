from unittest.mock import patch

import pytest

from tldwatch.cli.main import create_parser
from tldwatch.core.summarizer import Summarizer, SummarizerError


def test_cli_argument_parsing():
    """Test CLI argument parsing"""
    parser = create_parser()
    args = parser.parse_args(["--video-id", "abc123", "--provider", "openai"])
    assert args.video_id == "abc123"
    assert args.provider == "openai"


@pytest.mark.asyncio
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
async def test_cli_input_handling():
    """Test different input methods (URL, video ID, stdin)"""
    parser = create_parser()

    summarizer = Summarizer(provider="openai", model="gpt-3.5-turbo")

    # Test video ID input
    args = parser.parse_args(["--video-id", "dQw4w9WgXcQ"])
    video_id = summarizer.validate_input(video_id=args.video_id)
    assert video_id == "dQw4w9WgXcQ"

    # Test URL input
    args = parser.parse_args(["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
    video_id = summarizer.validate_input(url=args.url)
    assert video_id == "dQw4w9WgXcQ"

    # Test stdin input
    args = parser.parse_args(["--stdin"])
    with pytest.raises(SummarizerError):
        summarizer.validate_input(stdin_content=None)


@pytest.mark.asyncio
async def test_cli_output_formats():
    """Test different output formats and file handling"""
    parser = create_parser()

    # Test valid JSON output file
    args = parser.parse_args(["--video-id", "abc123", "--out", "output.json"])
    assert args.out == "output.json"

    # Test invalid output file format
    from tldwatch.cli.main import main

    args = parser.parse_args(["--video-id", "abc123", "--out", "output.txt"])
    with pytest.raises(SystemExit):
        await main()

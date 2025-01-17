import pytest

from tldwatch.cli.main import create_parser, get_input_source


def test_cli_argument_parsing():
    """Test CLI argument parsing"""
    parser = create_parser()
    args = parser.parse_args(["--video-id", "abc123", "--provider", "openai"])
    assert args.video_id == "abc123"
    assert args.provider == "openai"


def test_cli_input_handling():
    """Test different input methods (URL, video ID, stdin)"""
    parser = create_parser()

    # Test video ID input
    args = parser.parse_args(["--video-id", "abc123"])
    assert get_input_source(args) == "abc123"

    # Test URL input
    args = parser.parse_args(["https://www.youtube.com/watch?v=abc123"])
    assert get_input_source(args) == "abc123"

    # Test stdin input
    args = parser.parse_args(["--stdin"])
    with pytest.raises(SystemExit):
        get_input_source(args)


def test_cli_output_formats():
    """Test different output formats and file handling"""
    parser = create_parser()

    # Test valid JSON output file
    args = parser.parse_args(["--video-id", "abc123", "--out", "output.json"])
    assert args.out == "output.json"

    # Test invalid output file format
    with pytest.raises(SystemExit):
        parser.parse_args(["--video-id", "abc123", "--out", "output.txt"])

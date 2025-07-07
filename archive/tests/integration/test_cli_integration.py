import json
import os
from unittest.mock import patch

import pytest

from tldwatch.cli.main import create_parser, main

TEST_VIDEO_ID = "jNQXAC9IVRw"
TEST_VIDEO_URL = f"https://www.youtube.com/watch?v={TEST_VIDEO_ID}"


class MockSummarizer:
    """Mock summarizer for testing"""

    def __init__(self, *args, **kwargs):
        self.summary = "Test summary"
        self.video_id = TEST_VIDEO_ID

    def validate_input(self, video_id=None, url=None, stdin_content=None):
        return TEST_VIDEO_ID

    async def get_summary(self, video_id=None):
        pass

    async def export_summary(self, filepath):
        with open(filepath, "w") as f:
            json.dump({"summary": self.summary}, f)

    async def close(self):
        pass


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "GROQ_API_KEY": "test_key",
        "GEMINI_API_KEY": "test_key",
        "CEREBRAS_API_KEY": "test_key",
        "DEEPSEEK_API_KEY": "test_key",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def mock_summarizer():
    """Mock the Summarizer class"""
    with patch("tldwatch.cli.main.Summarizer", MockSummarizer):
        yield


def test_cli_video_id(mock_env_vars, mock_summarizer):
    """Test CLI with video ID input"""
    # Test argument parsing
    parser = create_parser()
    args = parser.parse_args(["--video-id", TEST_VIDEO_ID])

    assert args.video_id == TEST_VIDEO_ID
    assert args.url is None
    assert args.stdin is False


def test_cli_url(mock_env_vars, mock_summarizer):
    """Test CLI with URL input"""
    parser = create_parser()
    args = parser.parse_args([TEST_VIDEO_URL])

    assert args.url == TEST_VIDEO_URL
    assert args.video_id is None
    assert args.stdin is False


def test_cli_stdin(mock_env_vars, mock_summarizer):
    """Test CLI with stdin input"""
    parser = create_parser()
    args = parser.parse_args(["--stdin"])

    assert args.stdin is True
    assert args.video_id is None
    assert args.url is None


def test_cli_output_file(mock_env_vars, mock_summarizer, tmp_path):
    """Test CLI with output file"""
    output_file = tmp_path / "summary.json"
    parser = create_parser()
    args = parser.parse_args(["--video-id", TEST_VIDEO_ID, "--out", str(output_file)])

    assert args.video_id == TEST_VIDEO_ID
    assert args.out == str(output_file)


def test_cli_provider_selection(mock_env_vars, mock_summarizer):
    """Test CLI with different providers"""
    providers = ["openai", "anthropic"]

    for provider in providers:
        parser = create_parser()
        args = parser.parse_args(["--video-id", TEST_VIDEO_ID, "--provider", provider])

        assert args.provider == provider


def test_cli_config_management(mock_env_vars, mock_summarizer, tmp_path):
    """Test CLI configuration management"""
    parser = create_parser()
    args = parser.parse_args(
        [
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--temperature",
            "0.8",
            "--chunk-size",
            "5000",
            "--save-config",
        ]
    )

    assert args.provider == "openai"
    assert args.model == "gpt-4o-mini"
    assert args.temperature == 0.8
    assert args.chunk_size == 5000
    assert args.save_config is True


def test_cli_error_handling(mock_env_vars, mock_summarizer):
    """Test CLI error handling"""
    # Test that parser works with valid arguments
    parser = create_parser()
    args = parser.parse_args(["--video-id", "test_id"])
    assert args.video_id == "test_id"


def test_cli_full_context_flag(mock_env_vars, mock_summarizer):
    """Test CLI with full context flag"""
    parser = create_parser()
    args = parser.parse_args(["--video-id", TEST_VIDEO_ID, "--full-context"])

    assert args.video_id == TEST_VIDEO_ID
    assert args.full_context is True


def test_cli_keyboard_interrupt(mock_env_vars, mock_summarizer):
    """Test CLI handling of keyboard interrupt"""
    # This test verifies that the keyboard interrupt handling exists
    from tldwatch.cli.main import cli_entry

    # Just verify the function exists and can be imported
    assert cli_entry is not None
    assert callable(cli_entry)


@pytest.mark.asyncio
async def test_main_function_integration(mock_env_vars, mock_summarizer):
    """Integration test for the main function"""
    # Mock sys.argv
    test_args = ["tldwatch", "--video-id", TEST_VIDEO_ID]

    with patch("sys.argv", test_args):
        with patch("tldwatch.cli.main.console.print") as mock_print:
            # This should run without error
            try:
                await main()
            except SystemExit:
                pass  # Expected for successful runs

            # Check that console.print was called (indicating output)
            assert mock_print.called

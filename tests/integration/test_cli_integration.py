import json
import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tldwatch.cli.main import main

TEST_VIDEO_ID = "jNQXAC9IVRw"
TEST_VIDEO_URL = f"https://www.youtube.com/watch?v={TEST_VIDEO_ID}"


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing"""
    return CliRunner()


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing"""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "YOUTUBE_API_KEY": "test_key",
        },
    ):
        yield


def test_cli_video_id(cli_runner, mock_env_vars):
    """Test CLI with video ID input"""
    result = cli_runner.invoke(main, ["--video-id", TEST_VIDEO_ID])
    assert result.exit_code == 0
    assert len(result.output) > 0


def test_cli_url(cli_runner, mock_env_vars):
    """Test CLI with URL input"""
    result = cli_runner.invoke(main, [TEST_VIDEO_URL])
    assert result.exit_code == 0
    assert len(result.output) > 0


def test_cli_stdin(cli_runner, mock_env_vars):
    """Test CLI with stdin input"""
    result = cli_runner.invoke(main, ["--stdin"], input=TEST_VIDEO_URL)
    assert result.exit_code == 0
    assert len(result.output) > 0


def test_cli_output_file(cli_runner, mock_env_vars, tmp_path):
    """Test CLI with output file"""
    output_file = tmp_path / "summary.json"
    result = cli_runner.invoke(
        main, ["--video-id", TEST_VIDEO_ID, "--out", str(output_file)]
    )

    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)
        assert "summary" in data
        assert "transcript" in data


def test_cli_provider_selection(cli_runner, mock_env_vars):
    """Test CLI with different providers"""
    providers = ["openai", "anthropic"]

    for provider in providers:
        result = cli_runner.invoke(
            main, ["--video-id", TEST_VIDEO_ID, "--provider", provider]
        )
        assert result.exit_code == 0
        assert len(result.output) > 0


def test_cli_config_management(cli_runner, mock_env_vars, tmp_path):
    """Test CLI configuration management"""
    # Test saving config
    result = cli_runner.invoke(
        main,
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
        ],
    )
    assert result.exit_code == 0

    # Test printing config
    result = cli_runner.invoke(main, ["--print-config"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "gpt-4o-mini" in result.output


def test_cli_error_handling(cli_runner, mock_env_vars):
    """Test CLI error handling"""
    # Test invalid video ID
    result = cli_runner.invoke(main, ["--video-id", "invalid_id"])
    assert result.exit_code != 0

    # Test invalid URL
    result = cli_runner.invoke(main, ["https://youtube.com/invalid"])
    assert result.exit_code != 0

    # Test missing required input
    result = cli_runner.invoke(main, [])
    assert result.exit_code != 0


def test_cli_full_context_flag(cli_runner, mock_env_vars):
    """Test CLI with full context flag"""
    result = cli_runner.invoke(main, ["--video-id", TEST_VIDEO_ID, "--full-context"])
    assert result.exit_code == 0
    assert len(result.output) > 0


def test_cli_keyboard_interrupt(cli_runner, mock_env_vars):
    """Test CLI handling of keyboard interrupt"""
    with patch("asyncio.run", side_effect=KeyboardInterrupt):
        result = cli_runner.invoke(main, ["--video-id", TEST_VIDEO_ID])
        assert result.exit_code != 0
        assert "Operation cancelled" in result.output

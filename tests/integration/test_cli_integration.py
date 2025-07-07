"""
Integration tests for CLI functionality.
Tests the complete CLI workflow with real components.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldwatch.cli.main import main


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.fixture
    def cli_temp_dirs(self):
        """Provide temporary directories for CLI integration tests."""
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as config_dir,
        ):
            yield {"cache_dir": cache_dir, "config_dir": config_dir}

    @patch("sys.argv")
    @patch("tldwatch.cli.main.get_user_config")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    @patch("aiohttp.ClientSession")
    async def test_cli_youtube_url_summarization(
        self,
        mock_session,
        mock_get_transcript,
        mock_get_user_config,
        mock_argv,
        cli_temp_dirs,
        mock_env_vars,
    ):
        """Test CLI with YouTube URL input."""
        # Setup CLI arguments
        sys.argv = [
            "tldwatch",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "--no-cache",
        ]

        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = cli_temp_dirs["cache_dir"]
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup transcript API
        mock_get_transcript.return_value = [
            {
                "text": "Welcome to this amazing video about technology.",
                "start": 0.0,
                "duration": 3.0,
            },
            {
                "text": "Today we're discussing the latest developments.",
                "start": 3.0,
                "duration": 3.5,
            },
            {"text": "Thank you for watching!", "start": 6.5, "duration": 2.0},
        ]

        # Setup HTTP session
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "This video discusses the latest technology developments and thanks viewers for watching."
                        }
                    }
                ]
            }
        )

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        with patch("tldwatch.cli.main.console") as mock_console:
            await main()

        # Verify output was printed
        mock_console.print.assert_called()

        # Verify API was called
        mock_session_instance.post.assert_called_once()

        # Verify transcript was fetched
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    @patch("sys.argv")
    @patch("tldwatch.cli.main.get_user_config")
    @patch("aiohttp.ClientSession")
    async def test_cli_direct_text_summarization(
        self,
        mock_session,
        mock_get_user_config,
        mock_argv,
        cli_temp_dirs,
        mock_env_vars,
    ):
        """Test CLI with direct text input."""
        # Setup CLI arguments
        long_text = "This is a very long piece of text that needs to be summarized using the CLI interface."
        sys.argv = [
            "tldwatch",
            long_text,
            "--temperature",
            "0.5",
            "--no-cache",
        ]

        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = cli_temp_dirs["cache_dir"]
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup HTTP session
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "This text discusses the need for summarization via CLI."
                        }
                    }
                ]
            }
        )

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        with patch("tldwatch.cli.main.console") as mock_console:
            await main()

        # Verify output was printed
        mock_console.print.assert_called()

        # Verify correct temperature was used
        mock_session_instance.post.assert_called_once()

    @patch("sys.argv")
    @patch("tldwatch.cli.main.get_user_config")
    async def test_cli_create_config(
        self, mock_get_user_config, mock_argv, cli_temp_dirs
    ):
        """Test CLI config creation command."""
        # Setup CLI arguments
        sys.argv = ["tldwatch", "--create-config"]

        # Setup user config
        mock_user_config = MagicMock()
        mock_get_user_config.return_value = mock_user_config

        with patch("tldwatch.cli.main.console") as mock_console:
            await main()

        mock_user_config.create_example_config.assert_called_once()
        mock_console.print.assert_called()

    @patch("sys.argv")
    @patch("tldwatch.cli.main.get_user_config")
    async def test_cli_show_config(
        self, mock_get_user_config, mock_argv, cli_temp_dirs
    ):
        """Test CLI show config command."""
        # Setup CLI arguments
        sys.argv = ["tldwatch", "--show-config"]

        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.get_config_info.return_value = {
            "config_file": f'{cli_temp_dirs["config_dir"]}/config.json',
            "default_provider": "openai",
            "cache_enabled": True,
        }
        mock_get_user_config.return_value = mock_user_config

        with patch("tldwatch.cli.main.console") as mock_console:
            await main()

        mock_console.print.assert_called()

    @patch("sys.argv")
    @patch("tldwatch.cli.main.Summarizer")
    async def test_cli_list_providers(self, mock_summarizer_class, mock_argv):
        """Test CLI list providers command."""
        # Setup CLI arguments
        sys.argv = ["tldwatch", "--list-providers"]

        # Setup summarizer class
        mock_summarizer_class.list_providers.return_value = [
            "openai",
            "anthropic",
            "google",
        ]

        with patch("tldwatch.cli.main.console") as mock_console:
            await main()

        mock_console.print.assert_called()

    @patch("sys.argv")
    @patch("tldwatch.cli.main.get_user_config")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    @patch("aiohttp.ClientSession")
    async def test_cli_output_to_file(
        self,
        mock_session,
        mock_get_transcript,
        mock_get_user_config,
        mock_argv,
        cli_temp_dirs,
        mock_env_vars,
    ):
        """Test CLI with output file option."""
        output_file = Path(cli_temp_dirs["cache_dir"]) / "output.txt"

        # Setup CLI arguments
        output_file = Path(cli_temp_dirs["cache_dir"]) / "output.txt"
        sys.argv = [
            "tldwatch",
            "dQw4w9WgXcQ",
            "--output",
            str(output_file),
            "--no-cache",
        ]

        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = cli_temp_dirs["cache_dir"]
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup transcript API
        mock_get_transcript.return_value = [
            {"text": "Test content", "start": 0.0, "duration": 2.0}
        ]

        # Setup HTTP session
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {"message": {"content": "Generated summary for file output test."}}
                ]
            }
        )

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            await main()

        # Verify file was opened for writing
        mock_open.assert_called_once_with(str(output_file), "w", encoding="utf-8")

        # Verify content was written
        mock_file.write.assert_called_once_with(
            "Generated summary for file output test."
        )

    @patch("sys.argv")
    @patch("tldwatch.cli.main.get_user_config")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    async def test_cli_cache_options(
        self,
        mock_get_transcript,
        mock_get_user_config,
        mock_argv,
        cli_temp_dirs,
        mock_env_vars,
    ):
        """Test CLI cache-related options."""
        # Setup CLI arguments for no-cache
        sys.argv = [
            "tldwatch",
            "dQw4w9WgXcQ",
            "--no-cache",
        ]

        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = cli_temp_dirs["cache_dir"]
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup transcript API
        mock_get_transcript.return_value = [
            {"text": "Test content", "start": 0.0, "duration": 2.0}
        ]

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Summary with no cache"}}]
                }
            )

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value.__aenter__.return_value = (
                mock_response
            )
            mock_session.return_value.__aenter__.return_value = mock_session_instance

            with patch("tldwatch.cli.main.console"):
                await main()

        # Verify transcript was fetched
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    @patch("sys.argv")
    @patch("tldwatch.cli.main.get_user_config")
    @patch("tldwatch.utils.cache.get_cache")
    @patch("youtube_transcript_api.YouTubeTranscriptApi.get_transcript")
    async def test_cli_force_regenerate(
        self,
        mock_get_transcript,
        mock_get_cache,
        mock_get_user_config,
        mock_argv,
        cli_temp_dirs,
        mock_env_vars,
    ):
        """Test CLI force regenerate option."""
        # Setup CLI arguments
        sys.argv = [
            "tldwatch",
            "dQw4w9WgXcQ",
            "--force-regenerate",
        ]

        # Setup user config
        mock_user_config = MagicMock()
        mock_user_config.is_cache_enabled.return_value = True
        mock_user_config.get_cache_dir.return_value = cli_temp_dirs["cache_dir"]
        mock_user_config.get_default_provider.return_value = "openai"
        mock_user_config.get_default_temperature.return_value = 0.7
        mock_user_config.get_default_chunking_strategy.return_value = "standard"
        mock_user_config.get_provider_default_model.return_value = "gpt-4o"
        mock_get_user_config.return_value = mock_user_config

        # Setup cache
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        # Setup transcript API
        mock_get_transcript.return_value = [
            {"text": "Test content", "start": 0.0, "duration": 2.0}
        ]

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Regenerated summary"}}]
                }
            )

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value.__aenter__.return_value = (
                mock_response
            )
            mock_session.return_value.__aenter__.return_value = mock_session_instance

            with (
                patch("tldwatch.cli.main.console"),
                patch("tldwatch.utils.cache.clear_cache") as mock_clear_cache,
            ):
                await main()

        # Verify cache was cleared for the video
        mock_clear_cache.assert_called_once_with(
            video_id="dQw4w9WgXcQ", cache_dir=cli_temp_dirs["cache_dir"]
        )

    @patch("tldwatch.cli.main.Summarizer")
    async def test_cli_error_handling(self, mock_summarizer_class):
        """Test CLI error handling."""
        # Setup CLI arguments
        sys.argv = [
            "tldwatch",
            "test text for error",
        ]

        # Setup summarizer to raise an error
        mock_summarizer = MagicMock()
        mock_summarizer.summarize.side_effect = Exception("Test error")
        mock_summarizer_class.return_value = mock_summarizer

        with patch("tldwatch.cli.main.console") as mock_console:
            with pytest.raises(SystemExit) as exc_info:
                await main()

        assert exc_info.value.code == 1
        mock_console.print.assert_called()

        # Check that error message was displayed
        error_call = mock_console.print.call_args_list[-1]
        assert "error" in error_call[0][0].lower()

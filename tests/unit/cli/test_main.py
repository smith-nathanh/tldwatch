"""
Unit tests for CLI functionality.
Tests argument parsing, command execution, and error handling.
"""

import argparse
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldwatch.cli.main import create_proxy_config, main, setup_logging


class TestSetupLogging:
    """Test logging setup functionality."""

    @patch("logging.basicConfig")
    def test_setup_logging_normal(self, mock_basic_config):
        """Test normal logging setup."""
        setup_logging(verbose=False)
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args[1]
        assert call_args["level"] == 20  # logging.INFO

    @patch("logging.basicConfig")
    def test_setup_logging_verbose(self, mock_basic_config):
        """Test verbose logging setup."""
        setup_logging(verbose=True)
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args[1]
        assert call_args["level"] == 10  # logging.DEBUG


class TestCreateProxyConfig:
    """Test proxy configuration creation."""

    def test_no_proxy_config(self):
        """Test when no proxy configuration is provided."""
        args = argparse.Namespace(
            webshare_username=None,
            webshare_password=None,
            http_proxy=None,
            https_proxy=None,
        )

        with patch.dict("os.environ", {}, clear=True):
            result = create_proxy_config(args)
            assert result is None

    @patch("tldwatch.cli.main.create_webshare_proxy")
    def test_webshare_proxy_from_args(self, mock_create_webshare):
        """Test creating Webshare proxy from command line arguments."""
        mock_proxy = MagicMock()
        mock_create_webshare.return_value = mock_proxy

        args = argparse.Namespace(
            webshare_username="test_user",
            webshare_password="test_pass",
            http_proxy=None,
            https_proxy=None,
        )

        result = create_proxy_config(args)

        assert result == mock_proxy
        mock_create_webshare.assert_called_once_with(
            proxy_username="test_user", proxy_password="test_pass"
        )

    @patch("tldwatch.cli.main.create_webshare_proxy")
    def test_webshare_proxy_from_env(self, mock_create_webshare):
        """Test creating Webshare proxy from environment variables."""
        mock_proxy = MagicMock()
        mock_create_webshare.return_value = mock_proxy

        args = argparse.Namespace(
            webshare_username=None,
            webshare_password=None,
            http_proxy=None,
            https_proxy=None,
        )

        with patch.dict(
            "os.environ",
            {
                "WEBSHARE_PROXY_USERNAME": "env_user",
                "WEBSHARE_PROXY_PASSWORD": "env_pass",
            },
        ):
            result = create_proxy_config(args)

        assert result == mock_proxy
        mock_create_webshare.assert_called_once_with(
            proxy_username="env_user", proxy_password="env_pass"
        )

    @patch("tldwatch.cli.main.create_generic_proxy")
    def test_generic_proxy_from_args(self, mock_create_generic):
        """Test creating generic proxy from command line arguments."""
        mock_proxy = MagicMock()
        mock_create_generic.return_value = mock_proxy

        args = argparse.Namespace(
            webshare_username=None,
            webshare_password=None,
            http_proxy="http://proxy.example.com:8080",
            https_proxy="https://proxy.example.com:8080",
        )

        result = create_proxy_config(args)

        assert result == mock_proxy
        mock_create_generic.assert_called_once_with(
            http_url="http://proxy.example.com:8080",
            https_url="https://proxy.example.com:8080",
        )

    @patch("tldwatch.cli.main.create_webshare_proxy")
    @patch("tldwatch.cli.main.console")
    def test_webshare_proxy_error(self, mock_console, mock_create_webshare):
        """Test handling Webshare proxy configuration error."""
        from tldwatch.core.proxy_config import ProxyConfigError

        mock_create_webshare.side_effect = ProxyConfigError("Test error")

        args = argparse.Namespace(
            webshare_username="test_user",
            webshare_password="test_pass",
            http_proxy=None,
            https_proxy=None,
        )

        result = create_proxy_config(args)

        assert result is None
        mock_console.print.assert_called_once()
        assert (
            "Webshare proxy configuration error" in mock_console.print.call_args[0][0]
        )


class TestMainFunction:
    """Test the main CLI function."""

    def setup_method(self):
        """Setup for each test method."""
        self.original_argv = sys.argv.copy()

    def teardown_method(self):
        """Cleanup after each test method."""
        sys.argv = self.original_argv

    @patch("tldwatch.cli.main.console")
    async def test_main_list_providers(self, mock_console):
        """Test --list-providers flag."""
        sys.argv = ["tldwatch", "--list-providers"]

        with patch("tldwatch.cli.main.Summarizer") as mock_summarizer_class:
            mock_summarizer_class.list_providers.return_value = ["openai", "anthropic"]

            # Function should return without raising SystemExit
            result = await main()
            assert result is None
            mock_console.print.assert_called()

    @patch("tldwatch.cli.main.console")
    async def test_main_show_defaults(self, mock_console):
        """Test --show-defaults flag."""
        sys.argv = ["tldwatch", "--show-defaults"]

        with patch("tldwatch.cli.main.Summarizer") as mock_summarizer_class:
            mock_summarizer_class.list_providers.return_value = ["openai", "anthropic"]

            result = await main()
            assert result is None
            mock_console.print.assert_called()

    @patch("tldwatch.cli.main.console")
    @patch("tldwatch.cli.main.get_user_config")
    async def test_main_create_config(self, mock_get_user_config, mock_console):
        """Test --create-config flag."""
        sys.argv = ["tldwatch", "--create-config"]

        mock_config = MagicMock()
        mock_get_user_config.return_value = mock_config

        result = await main()
        assert result is None
        mock_config.create_example_config.assert_called_once()

    @patch("tldwatch.cli.main.console")
    @patch("tldwatch.cli.main.get_user_config")
    async def test_main_show_config(self, mock_get_user_config, mock_console):
        """Test --show-config flag."""
        sys.argv = ["tldwatch", "--show-config"]

        mock_config = MagicMock()
        mock_config.get_config_info.return_value = {
            "config_file": "/path/to/config.json",
            "default_provider": "openai",
        }
        mock_get_user_config.return_value = mock_config

        result = await main()
        assert result is None
        mock_console.print.assert_called()

    @patch("tldwatch.cli.main.Summarizer")
    async def test_main_summarize_text(self, mock_summarizer_class):
        """Test text summarization."""
        sys.argv = [
            "tldwatch",
            "This is a long text that needs to be summarized for testing purposes.",
        ]

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(return_value="Generated summary")
        mock_summarizer_class.return_value = mock_summarizer

        with patch("tldwatch.cli.main.console"):
            await main()

        mock_summarizer.summarize.assert_called_once()
        call_args = mock_summarizer.summarize.call_args
        # Check that video_input keyword argument contains the expected text
        assert (
            call_args.kwargs["video_input"]
            == "This is a long text that needs to be summarized for testing purposes."
        )

    @patch("tldwatch.cli.main.Summarizer")
    async def test_main_summarize_with_options(self, mock_summarizer_class):
        """Test summarization with custom options."""
        # Set up the mock before main() is called
        mock_summarizer_class.list_providers.return_value = [
            "openai",
            "anthropic",
            "claude",
        ]
        mock_summarizer_class.list_chunking_strategies.return_value = [
            "none",
            "standard",
            "large",
        ]

        sys.argv = [
            "tldwatch",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "--provider",
            "anthropic",
            "--model",
            "claude-3-5-sonnet-20241022",
            "--temperature",
            "0.5",
            "--chunking",
            "large",
        ]

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(return_value="Generated summary")
        mock_summarizer_class.return_value = mock_summarizer

        with patch("tldwatch.cli.main.console"):
            await main()

        mock_summarizer.summarize.assert_called_once()
        call_args = mock_summarizer.summarize.call_args
        call_kwargs = call_args[1]

        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["chunking_strategy"] == "large"

    @patch("tldwatch.cli.main.Summarizer")
    async def test_main_with_output_file(self, mock_summarizer_class):
        """Test saving output to file."""
        sys.argv = [
            "tldwatch",
            "Test text for summarization",
            "--output",
            "/tmp/test_output.txt",
        ]

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(return_value="Generated summary")
        mock_summarizer_class.return_value = mock_summarizer

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            await main()

        mock_open.assert_called_once_with("/tmp/test_output.txt", "w", encoding="utf-8")
        mock_file.write.assert_called_once_with("Generated summary")

    @patch("tldwatch.cli.main.Summarizer")
    @patch("tldwatch.cli.main.console")
    async def test_main_error_handling(self, mock_console, mock_summarizer_class):
        """Test error handling in main function."""
        sys.argv = ["tldwatch", "test text"]

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(side_effect=Exception("Test error"))
        mock_summarizer_class.return_value = mock_summarizer

        with pytest.raises(SystemExit) as exc_info:
            await main()

        assert exc_info.value.code == 1
        mock_console.print.assert_called()
        assert "error" in mock_console.print.call_args[0][0].lower()

    async def test_main_no_input(self):
        """Test main function with no input provided."""
        sys.argv = ["tldwatch"]

        with patch("sys.stderr", new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                await main()

        assert exc_info.value.code == 1  # no input error

    @patch("tldwatch.cli.main.Summarizer")
    async def test_main_cache_options(self, mock_summarizer_class):
        """Test cache-related options."""
        sys.argv = ["tldwatch", "dQw4w9WgXcQ", "--no-cache"]

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(return_value="Generated summary")
        mock_summarizer_class.return_value = mock_summarizer

        with patch("tldwatch.cli.main.console"):
            await main()

        mock_summarizer.summarize.assert_called_once()
        call_kwargs = mock_summarizer.summarize.call_args[1]
        assert call_kwargs["use_cache"] is False

    @patch("tldwatch.cli.main.Summarizer")
    async def test_main_force_regenerate(self, mock_summarizer_class):
        """Test force regenerate option."""
        sys.argv = ["tldwatch", "dQw4w9WgXcQ", "--force-regenerate"]

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(return_value="Generated summary")
        mock_summarizer_class.return_value = mock_summarizer

        with (
            patch("tldwatch.cli.main.console"),
            patch("tldwatch.utils.cache.clear_cache") as mock_clear_cache,
            patch("tldwatch.cli.main.get_user_config") as mock_get_user_config,
        ):
            mock_user_config = MagicMock()
            mock_user_config.get_cache_dir.return_value = "/tmp/cache"
            mock_get_user_config.return_value = mock_user_config

            await main()

        # Should clear cache for the video before summarizing
        mock_clear_cache.assert_called_once_with(
            video_id="dQw4w9WgXcQ", cache_dir="/tmp/cache"
        )

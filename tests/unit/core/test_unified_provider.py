"""
Unit tests for the unified provider system.
Tests provider configuration, initialization, and basic functionality.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from tldwatch.core.providers.unified_provider import (
    ChunkingStrategy,
    ProviderConfig,
    ProviderError,
    UnifiedProvider,
)


class TestChunkingStrategy:
    """Test ChunkingStrategy enum."""

    def test_chunking_strategy_values(self):
        """Test that chunking strategy enum has expected values."""
        assert ChunkingStrategy.NONE.value == "none"
        assert ChunkingStrategy.STANDARD.value == "standard"
        assert ChunkingStrategy.SMALL.value == "small"
        assert ChunkingStrategy.LARGE.value == "large"


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_provider_config_creation(self):
        """Test creating a provider configuration."""
        config = ProviderConfig(
            name="openai",
            api_key_env="OPENAI_API_KEY",
            api_base="https://api.openai.com/v1",
            default_model="gpt-4o",
        )

        assert config.name == "openai"
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.api_base == "https://api.openai.com/v1"
        assert config.default_model == "gpt-4o"


class TestUnifiedProvider:
    """Test UnifiedProvider functionality."""

    @pytest.fixture
    def mock_provider_config(self):
        """Mock provider configuration."""
        return {
            "openai": ProviderConfig(
                name="openai",
                api_key_env="OPENAI_API_KEY",
                api_base="https://api.openai.com/v1",
                default_model="gpt-4o",
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                api_base="https://api.anthropic.com",
                default_model="claude-3-5-sonnet-20241022",
            ),
        }

    @pytest.fixture
    def mock_user_config(self):
        """Mock user configuration."""
        mock_config = MagicMock()
        mock_config.get_default_provider.return_value = "openai"
        mock_config.get_default_temperature.return_value = 0.7
        mock_config.get_default_chunking_strategy.return_value = "standard"
        mock_config.get_default_model.return_value = "gpt-4o"
        return mock_config

    def test_provider_initialization_defaults(
        self, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test provider initialization with default values."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            provider = UnifiedProvider()

            assert provider.config.name == "openai"
            assert provider.model == "gpt-4o"
            assert provider.temperature == 0.7
            assert provider.chunking_strategy == ChunkingStrategy.STANDARD

    def test_provider_initialization_with_params(
        self, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test provider initialization with explicit parameters."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            provider = UnifiedProvider(
                provider="anthropic",
                model="claude-3-opus-20240229",
                temperature=0.5,
                chunking_strategy="large",
            )

            assert provider.config.name == "anthropic"
            assert provider.model == "claude-3-opus-20240229"
            assert provider.temperature == 0.5
            assert provider.chunking_strategy == ChunkingStrategy.LARGE

    def test_provider_initialization_invalid_provider(
        self, mock_provider_config, mock_user_config
    ):
        """Test provider initialization with invalid provider name."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            with pytest.raises(ValueError, match="Unsupported provider"):
                UnifiedProvider(provider="invalid_provider")

    def test_provider_initialization_missing_api_key(
        self, mock_provider_config, mock_user_config
    ):
        """Test provider initialization with missing API key - should not raise during init."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            # Remove API key from environment
            with patch.dict(os.environ, {}, clear=True):
                # Should not raise during initialization
                provider = UnifiedProvider(provider="openai")
                assert provider.api_key is None

    def test_chunking_strategy_conversion(
        self, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test chunking strategy string to enum conversion."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            # Test string conversion
            provider = UnifiedProvider(chunking_strategy="small")
            assert provider.chunking_strategy == ChunkingStrategy.SMALL

            # Test enum passed directly
            provider2 = UnifiedProvider(chunking_strategy=ChunkingStrategy.LARGE)
            assert provider2.chunking_strategy == ChunkingStrategy.LARGE

    def test_invalid_chunking_strategy(
        self, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test invalid chunking strategy raises error."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            with pytest.raises(ValueError, match="Invalid chunking strategy"):
                UnifiedProvider(chunking_strategy="invalid_strategy")

    def test_list_providers(self):
        """Test listing available providers."""
        mock_providers = {
            "openai": ProviderConfig(
                "openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o"
            ),
            "anthropic": ProviderConfig(
                "anthropic",
                "ANTHROPIC_API_KEY",
                "https://api.anthropic.com",
                "claude-3-5-sonnet-20241022",
            ),
        }

        with patch.object(
            UnifiedProvider, "_load_providers", return_value=mock_providers
        ):
            providers = UnifiedProvider.list_providers()
            assert "openai" in providers
            assert "anthropic" in providers

    def test_list_chunking_strategies(self):
        """Test listing available chunking strategies."""
        strategies = [strategy.value for strategy in ChunkingStrategy]
        assert "none" in strategies
        assert "standard" in strategies
        assert "small" in strategies
        assert "large" in strategies

    @patch("aiohttp.ClientSession")
    async def test_generate_summary_success(
        self, mock_session, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test successful summary generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a test summary."}}]
        }

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            provider = UnifiedProvider(provider="openai")
            result = await provider.generate_summary("Test text to summarize")

            assert result == "This is a test summary."
            mock_session_instance.post.assert_called_once()

    @patch("aiohttp.ClientSession")
    async def test_generate_summary_api_error(
        self, mock_session, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test summary generation with API error."""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text.return_value = "Bad Request"

        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance

        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            provider = UnifiedProvider(provider="openai")

            with pytest.raises(ProviderError, match="API request failed"):
                await provider.generate_summary("Test text")

    def test_text_chunking_none(
        self, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test text chunking with 'none' strategy."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            provider = UnifiedProvider(chunking_strategy="none")
            text = "This is a test text that should not be chunked."

            chunks = provider.chunk_text(text)
            assert len(chunks) == 1
            assert chunks[0] == text

    def test_text_chunking_standard(
        self, mock_provider_config, mock_user_config, mock_env_vars
    ):
        """Test text chunking with 'standard' strategy."""
        with (
            patch.object(
                UnifiedProvider, "_load_providers", return_value=mock_provider_config
            ),
            patch(
                "tldwatch.core.providers.unified_provider.get_user_config",
                return_value=mock_user_config,
            ),
        ):
            provider = UnifiedProvider(chunking_strategy="standard")

            # Create a long text that should be chunked
            long_text = "This is a test sentence. " * 1000  # Should trigger chunking

            chunks = provider.chunk_text(long_text)
            assert len(chunks) > 1  # Should be chunked

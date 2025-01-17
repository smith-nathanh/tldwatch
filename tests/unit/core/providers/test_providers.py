from unittest.mock import patch

import pytest

from tests.conftest import PROVIDER_TEST_CONFIG, PROVIDERS, mock_successful_completion


@pytest.mark.parametrize(
    "provider_name,provider_instance",
    [(name, name) for name in PROVIDERS.keys()],
    indirect=["provider_instance"],
)
class TestProviderInterface:
    """Test suite that runs against all providers"""

    async def test_initialization(self, provider_name, provider_instance):
        """Test provider initialization and config"""
        assert (
            provider_instance.model
            == PROVIDER_TEST_CONFIG[provider_name]["default_model"]
        )
        assert provider_instance.temperature == 0.7
        assert provider_instance.api_key.startswith("sk-") or provider_name == "ollama"

    async def test_rate_limiting(
        self, provider_name, provider_instance, mock_rate_limit_error
    ):
        """Test rate limit implementation"""
        # Verify rate limit config
        expected_rate_limit = PROVIDER_TEST_CONFIG[provider_name]["rate_limit"]
        assert (
            provider_instance.rate_limit_config.requests_per_minute
            == expected_rate_limit
        )

        # Test rate limit detection
        for _ in range(expected_rate_limit + 1):
            provider_instance._record_request()

        with pytest.raises(Exception) as exc:
            provider_instance._check_rate_limit()
            assert "rate limit" in str(exc.value).lower()

    async def test_token_counting(self, provider_name, provider_instance):
        """Test token counting"""
        test_text = "This is a test input" * 100
        token_count = provider_instance.count_tokens(test_text)
        assert isinstance(token_count, int)
        assert token_count > 0

        # Test empty input
        assert provider_instance.count_tokens("") == 0

    async def test_context_window(self, provider_name, provider_instance):
        """Test context window handling"""
        expected_window = PROVIDER_TEST_CONFIG[provider_name]["context_window"]
        assert provider_instance.context_window == expected_window

        # Test context window check
        long_text = "test " * 10000
        assert not provider_instance.can_use_full_context(long_text)

        short_text = "test"
        assert provider_instance.can_use_full_context(short_text)

    async def test_successful_completion(
        self, provider_name, provider_instance, mock_successful_completion
    ):
        """Test successful API response handling"""
        with patch.object(provider_instance, "_make_request") as mock_request:
            mock_request.return_value = mock_successful_completion[provider_name]

            result = await provider_instance.generate_summary("Test input")
            assert result == "Summary"
            mock_request.assert_called_once()

    async def test_error_handling(
        self,
        provider_name,
        provider_instance,
        mock_rate_limit_error,
        mock_auth_error,
        mock_network_error,
    ):
        """Test error handling"""
        # Test rate limit error
        with patch.object(provider_instance, "_make_request") as mock_request:
            mock_request.side_effect = Exception(mock_rate_limit_error[provider_name])

            with pytest.raises(Exception) as exc:
                await provider_instance.generate_summary("Test input")
                assert "rate limit" in str(exc.value).lower()

        # Test auth error
        with patch.object(provider_instance, "_make_request") as mock_request:
            mock_request.side_effect = Exception(mock_auth_error[provider_name])

            with pytest.raises(Exception) as exc:
                await provider_instance.generate_summary("Test input")
                assert "auth" in str(exc.value).lower()

    async def test_retry_mechanism(
        self, provider_name, provider_instance, mock_network_error
    ):
        """Test retry behavior"""
        with patch.object(provider_instance, "_make_request") as mock_request:
            mock_request.side_effect = [
                Exception(mock_network_error[provider_name]),
                mock_successful_completion[provider_name],
            ]

            result = await provider_instance.generate_summary("Test input")
            assert result == "Summary"
            assert mock_request.call_count == 2

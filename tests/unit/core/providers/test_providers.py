from unittest.mock import patch

import pytest

from tests.conftest import PROVIDERS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_name,provider_instance",
    [(name, name) for name in PROVIDERS.keys()],
    indirect=["provider_instance"],
)
class TestProviderInterface:
    """Test suite that runs against all providers"""

    async def test_initialization(self, provider_name, provider_instance):
        """Test provider initialization and config"""
        # Check temperature
        assert provider_instance.temperature == 0.7

        # Check API key format
        if provider_name == "ollama":
            # Ollama doesn't use an API key
            pass
        else:
            expected_prefixes = {
                "openai": "sk-",
                "anthropic": "sk-ant-",
                "groq": "gsk-",
                "cerebras": "csk-",
                "deepseek": "sk-",
            }
            assert provider_instance.api_key.startswith(
                expected_prefixes[provider_name]
            ), (
                f"API key for {provider_name} should start with {expected_prefixes[provider_name]}"
            )

    async def test_rate_limiting(
        self, provider_name, provider_instance, mock_rate_limit_error
    ):
        """Test rate limit implementation"""
        # Get rate limit directly from provider instance
        expected_rate_limit = provider_instance.rate_limit_config.requests_per_minute

        # Test rate limit detection
        for _ in range(expected_rate_limit + 1):
            provider_instance._record_request()

        with pytest.raises(Exception) as exc:
            provider_instance._check_rate_limit()
            assert "rate limit" in str(exc.value).lower()

    async def test_context_window(self, provider_name, provider_instance):
        """Test context window handling"""
        # Calculate text length based on provider's context window
        window_size = provider_instance.context_window
        # Create text that's definitely larger than the context window
        long_text = "test " * (window_size // 2 * 3)  # 150% of window size
        assert not provider_instance.can_use_full_context(long_text)

        # Test with very short text that should fit in any provider's window
        short_text = "test"
        assert provider_instance.can_use_full_context(short_text)

    async def test_token_counting(self, provider_name, provider_instance):
        """Test token counting"""
        test_text = "This is a test input" * 100
        token_count = provider_instance.count_tokens(test_text)
        assert isinstance(token_count, int)
        assert token_count > 0

        # Test empty input
        assert provider_instance.count_tokens("") == 0

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

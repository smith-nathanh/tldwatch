"""
Anthropic provider implementation using the unified request handler.
"""

import re
from typing import Any, Dict, Optional

from .base_provider import AuthenticationError, BaseProvider, RateLimitConfig
from .config_loader import get_context_windows, get_default_model


class AnthropicProvider(BaseProvider):
    """Anthropic API provider implementation with unified request handling"""

    API_BASE = "https://api.anthropic.com/v1"
    RESPONSE_KEY_PATH = ("content", 0, "text")

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Use config file for default model if not specified
        if model is None:
            model = get_default_model("anthropic")

        # Get context windows from config
        context_windows = get_context_windows("anthropic")

        if model not in context_windows:
            raise ValueError(
                f"Invalid model. Choose from: {', '.join(context_windows.keys())}"
            )

        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )

        # Verify API key exists after BaseProvider initialization
        if not self.api_key:
            raise AuthenticationError("Anthropic API key is required")

    def _get_provider_name(self) -> str:
        return "anthropic"

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Anthropic based on model"""
        if self.model == "claude-3-opus":
            tokens_per_minute = 20000  # Input tokens per minute for Opus
        elif self.model in ["claude-3-haiku", "claude-3-5-haiku"]:
            tokens_per_minute = 50000  # Input tokens per minute for Haiku models
        else:  # Sonnet models
            tokens_per_minute = 40000  # Input tokens per minute for Sonnet models

        return RateLimitConfig(
            requests_per_minute=50,  # All models have 50 RPM limit
            tokens_per_minute=tokens_per_minute,
            max_retries=3,
            retry_delay=2.0,
            timeout=120.0,
        )

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 10

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("anthropic")
        return context_windows[self.model]

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Claude models.
        This is an approximation - actual token count may vary slightly.
        """
        if not text:
            return 0

        # Clean the text
        text = text.strip()

        # Count words (including contractions and hyphenated words)
        words = len(re.findall(r"\b\w+(?:[-\']\w+)*\b", text))

        # Count special characters and punctuation
        special_chars = len(re.findall(r"[^\w\s]", text))

        # Count whitespace
        whitespace = len(re.findall(r"\s+", text))

        # Approximation based on typical Claude tokenization patterns
        # On average, words are ~1.3 tokens, special chars are 1 token each,
        # and whitespace is usually bundled with words
        estimated_tokens = (
            words * 1.3  # Words with common subword tokenization
            + special_chars  # Punctuation and special characters
            + whitespace * 0.1  # Small addition for whitespace
        )

        # Add 10% margin for safety
        return int(estimated_tokens * 1.1)

    def _prepare_request_headers(self) -> Dict[str, str]:
        """Prepare headers for Anthropic API request"""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Prepare request data for Anthropic API"""
        max_tokens = kwargs.pop("max_tokens", 4096)

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for Anthropic"""
        return "messages"
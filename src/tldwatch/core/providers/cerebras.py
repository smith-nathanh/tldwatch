from typing import Any, Dict, Optional

import requests

from .base import AuthenticationError, BaseProvider, ProviderError, RateLimitConfig


class CerebrasProvider(BaseProvider):
    """Cerebras API provider implementation"""

    API_BASE = "https://api.cerebras.xyz/v1"

    # Available models and their context windows
    CONTEXT_WINDOWS = {"llama3.1-8b": 8192, "llama3.1-70b": 8192}

    def __init__(
        self,
        model: str = "llama3.1-70b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        if model not in self.CONTEXT_WINDOWS:
            raise ValueError(
                f"Invalid model. Choose from: {', '.join(self.CONTEXT_WINDOWS.keys())}"
            )

        super().__init__(
            model, api_key, temperature, rate_limit_config, use_full_context
        )

        if not api_key:
            raise AuthenticationError("Cerebras API key is required")

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Cerebras"""
        return RateLimitConfig(
            requests_per_minute=60,  # Conservative default
            tokens_per_minute=None,  # Cerebras doesn't specify token rate limits
            max_retries=3,
            retry_delay=1.0,
        )

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        return self.CONTEXT_WINDOWS[self.model]

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Cerebras models.
        Using a conservative estimate of 4 characters per token.
        """
        return len(text) // 4 + 1

    def generate_summary(self, text: str) -> str:
        """Generate a summary using Cerebras's API"""
        try:
            self._check_rate_limit()
            response = self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make a request to Cerebras's API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise summaries.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            **kwargs,
        }

        response = requests.post(
            f"{self.API_BASE}/chat/completions", headers=headers, json=data
        )

        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            retry_after = float(
                response.headers.get("Retry-After", self.rate_limit_config.retry_delay)
            )
            self._handle_rate_limit(retry_after)
            # Retry the request
            return self._make_request(prompt, **kwargs)
        elif response.status_code != 200:
            raise ProviderError(f"Cerebras API error: {response.text}")

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected API response format: {str(e)}")

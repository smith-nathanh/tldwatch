from typing import Any, Dict, Optional

import requests

from .base import AuthenticationError, BaseProvider, ProviderError, RateLimitConfig


class GroqProvider(BaseProvider):
    """Groq API provider implementation"""

    API_BASE = "https://api.groq.com/v1"

    # Available models and their context windows from Groq documentation
    CONTEXT_WINDOWS = {
        "mixtral-8x7b-32768": 32768,
        "llama-3.3-70b-versatile": 128000,  # 128k
        "llama-3.1-8b-instant": 128000,  # 128k
        "llama-guard-3-8b": 8192,
        "llama3-70b-8192": 8192,
        "llama3-8b-8192": 8192,
        "gemma-2b-it": 8192,
        "whisper-large-v3": None,  # Audio model
        "whisper-large-v3-turbo": None,  # Audio model
        "distil-whisper-large-v3-en": None,  # Audio model
    }

    def __init__(
        self,
        model: str = "mixtral-8x7b-32768",
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
            raise AuthenticationError("Groq API key is required")

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Groq based on their documentation"""
        return RateLimitConfig(
            requests_per_minute=600,  # 14400 RPD ≈ 600 RPM
            tokens_per_minute=18000,  # From documentation
            max_retries=5,
            retry_delay=2.0,  # Default retry time from documentation
        )

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        return self.CONTEXT_WINDOWS[self.model]

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Groq models.
        Using a conservative estimate of 4 characters per token.
        """
        return len(text) // 4 + 1

    def generate_summary(self, text: str) -> str:
        """Generate a summary using Groq's API"""
        try:
            self._check_rate_limit()
            response = self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make a request to Groq's API"""
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
            f"{self.API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=30,  # Longer timeout for potentially large responses
        )

        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            # Groq often includes precise retry-after times
            retry_after = float(
                response.headers.get("Retry-After", self.rate_limit_config.retry_delay)
            )
            self._handle_rate_limit(retry_after)
            # Retry the request
            return self._make_request(prompt, **kwargs)
        elif response.status_code != 200:
            raise ProviderError(f"Groq API error: {response.text}")

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected API response format: {str(e)}")

    def _handle_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Handle rate limit by waiting, with specific adjustments for Groq"""
        # Groq sometimes needs a bit longer than the suggested retry time
        if retry_after is not None:
            retry_after *= 1.1  # Add 10% to the suggested wait time
        super()._handle_rate_limit(retry_after)

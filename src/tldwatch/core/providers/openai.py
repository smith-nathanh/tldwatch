from typing import Any, Dict, Optional

import requests
import tiktoken

from .base import AuthenticationError, BaseProvider, ProviderError, RateLimitConfig


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation"""

    API_BASE = "https://api.openai.com/v1"

    # Model context windows
    CONTEXT_WINDOWS = {
        "gpt-3.5-turbo": 16385,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo-preview": 128000,
    }

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        super().__init__(model, api_key, temperature, rate_limit_config)
        if not api_key:
            raise AuthenticationError("OpenAI API key is required")
        self._encoding = tiktoken.encoding_for_model(model)

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for OpenAI"""
        # These values should be adjusted based on the API tier
        return RateLimitConfig(
            requests_per_minute=3500,  # Standard tier limit
            tokens_per_minute=180000,  # Approximate for GPT-4
            max_retries=3,
            retry_delay=1.0,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model-specific tokenizer"""
        return len(self._encoding.encode(text))

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        return self.CONTEXT_WINDOWS.get(self.model, 8192)  # Default to 8k if unknown

    def generate_summary(self, text: str) -> str:
        """Generate a summary using OpenAI's chat completion API"""
        try:
            self._check_rate_limit()
            response = self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make a request to OpenAI's API"""
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
            raise ProviderError(f"OpenAI API error: {response.text}")

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected API response format: {str(e)}")

import asyncio
import re
from typing import Any, Dict, Optional

import aiohttp

from .base import (
    AuthenticationError,
    BaseProvider,
    ProviderError,
    RateLimitConfig,
    RateLimitError,
)
from .config_loader import get_context_windows, get_default_model


class CerebrasProvider(BaseProvider):
    """Provider implementation for Cerebras API"""

    API_BASE = "https://api.cerebras.ai/v1"

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Use config file for default model if not specified
        if model is None:
            model = get_default_model("cerebras")

        # Get context windows from config
        context_windows = get_context_windows("cerebras")

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
            raise AuthenticationError("Cerebras API key is required")

        self._retry_count = 0

    def _get_provider_name(self) -> str:
        return "cerebras"

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Cerebras"""
        return RateLimitConfig(
            requests_per_minute=30,
            tokens_per_minute=60000,
            max_retries=3,
            retry_delay=3.0,
            timeout=120.0,
        )

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 3  # Based on Cerebras's 30 RPM limit

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("cerebras")
        return context_windows[self.model]

    def count_tokens(self, text: str) -> int:
        """Approximate token counting for LLaMA-based models"""
        if not text:
            return 0

        # Clean the text
        text = text.strip()

        # Count words (including contractions and hyphenated words)
        words = len(re.findall(r"\b\w+(?:[-\']\w+)*\b", text))

        # Count punctuation and special characters
        punctuation = len(re.findall(r"[^\w\s]", text))

        # Count numerals
        numbers = len(re.findall(r"\d+", text))

        # Approximate LLaMA tokenization behavior
        estimated_tokens = (
            words * 1.3  # Average word-to-token ratio
            + punctuation  # Most punctuation is its own token
            + numbers * 0.5  # Numbers often combine into single tokens
        )

        return int(estimated_tokens * 1.1)  # Add safety margin

    async def generate_summary(self, text: str) -> str:
        """Generate a summary using Cerebras API"""
        try:
            self._check_rate_limit()
            self._retry_count = 0  # Reset retry counter before new request
            response = await self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make an async request to Cerebras API with error handling and retries"""
        last_exception = None

        while self._retry_count < self.rate_limit_config.max_retries:
            try:
                async with aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=30),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as session:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
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
                        "stream": False,
                        **kwargs,
                    }

                    async with session.post(
                        f"{self.API_BASE}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status == 401:
                            raise AuthenticationError("Invalid API key")
                        elif response.status == 429:
                            retry_after = float(
                                response.headers.get(
                                    "Retry-After", self.rate_limit_config.retry_delay
                                )
                            )
                            # Instead of raising immediately, we'll handle the retry
                            self._retry_count += 1
                            if self._retry_count >= self.rate_limit_config.max_retries:
                                raise RateLimitError(retry_after=retry_after)
                            await asyncio.sleep(retry_after)
                            continue

                        elif response.status != 200:
                            error_text = await response.text()
                            raise ProviderError(
                                f"API error (status {response.status}): {error_text}"
                            )

                        response_data = await response.json()
                        return response_data["choices"][0]["message"]["content"]

            except RateLimitError as e:
                last_exception = e
                raise  # If we've exceeded max retries, propagate the error
            except aiohttp.ClientError as e:
                last_exception = e
                self._retry_count += 1
                if self._retry_count >= self.rate_limit_config.max_retries:
                    raise ProviderError(
                        f"Network error after {self._retry_count} retries: {str(e)}"
                    )
                await asyncio.sleep(
                    self.rate_limit_config.retry_delay * self._retry_count
                )
            except (KeyError, IndexError, ValueError) as e:
                raise ProviderError(f"Invalid API response: {str(e)}")

        raise last_exception or ProviderError("Maximum retry attempts exceeded")

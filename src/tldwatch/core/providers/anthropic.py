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


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic API"""

    API_BASE = "https://api.anthropic.com/v1"

    # Available models and their context windows
    CONTEXT_WINDOWS = {
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
    }

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        if model not in self.CONTEXT_WINDOWS:
            raise ValueError(
                f"Invalid model. Choose from: {', '.join(self.CONTEXT_WINDOWS.keys())}"
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

        # Initialize session for reuse
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=30),
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._retry_count = 0

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
        return self.CONTEXT_WINDOWS[self.model]

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Claude 3 models.
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

        # Approximation based on typical Claude 3 tokenization patterns
        # On average, words are ~1.3 tokens, special chars are 1 token each,
        # and whitespace is usually bundled with words
        estimated_tokens = (
            words * 1.3  # Words with common subword tokenization
            + special_chars  # Punctuation and special characters
            + whitespace * 0.1  # Small addition for whitespace
        )

        # Add 10% margin for safety
        return int(estimated_tokens * 1.1)

    async def generate_summary(self, text: str) -> str:
        """Generate a summary using Anthropic's API"""
        try:
            self._check_rate_limit()
            self._retry_count = 0  # Reset retry counter before new request
            response = await self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make an async request to Anthropic's API with error handling and retries"""
        last_exception = None

        while self._retry_count < self.rate_limit_config.max_retries:
            try:
                # Create a new session if we don't have one
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(limit=30),
                        timeout=aiohttp.ClientTimeout(total=30),
                    )

                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }

                max_tokens = kwargs.get("max_tokens", 4096)

                data = {
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

                async with self._session.post(
                    f"{self.API_BASE}/messages",
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
                    return response_data["content"][0]["text"]

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

    async def close(self):
        """Close the aiohttp session"""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def __del__(self):
        """Ensure the session is closed when the provider is deleted"""
        if self._session is not None and not self._session.closed:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass  # We're in cleanup, so we can't raise exceptions

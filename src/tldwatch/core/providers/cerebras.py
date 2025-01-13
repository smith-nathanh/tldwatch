import asyncio
import re
import time
from typing import Any, Dict, List, Optional

import aiohttp

from .base import (
    AuthenticationError,
    BaseProvider,
    ProviderError,
    RateLimitConfig,
    RateLimitError,
)


class CerebrasProvider(BaseProvider):
    """Provider implementation for Cerebras API"""

    API_BASE = "https://api.cerebras.ai/v1"

    # Available models and their context windows
    CONTEXT_WINDOWS = {"llama3.1-8b": 8192, "llama3.1-70b": 8192, "llama-3.3-70b": 8192}

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
            model=model,
            api_key=api_key,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )

        if not api_key:
            raise AuthenticationError("Cerebras API key is required")

        # Initialize connection pool
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=30),
            timeout=aiohttp.ClientTimeout(total=30),
        )

        # Rate limiting state
        self._request_timestamps: List[float] = []
        self._request_lock = asyncio.Lock()

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
        return self.CONTEXT_WINDOWS[self.model]

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

    async def _check_rate_limit(self) -> None:
        """Check rate limits using rolling window"""
        async with self._request_lock:
            current_time = time.time()

            # Remove timestamps older than 1 minute
            self._request_timestamps = [
                ts for ts in self._request_timestamps if current_time - ts < 60
            ]

            if (
                len(self._request_timestamps)
                >= self.rate_limit_config.requests_per_minute
            ):
                oldest = self._request_timestamps[0]
                wait_time = 60 - (current_time - oldest)
                if wait_time > 0:
                    jitter = (
                        hash(str(current_time)) % 100
                    ) / 1000  # Deterministic jitter
                    await asyncio.sleep(wait_time + jitter)

            self._request_timestamps.append(current_time)

    async def generate_summary(self, text: str) -> str:
        """Generate a summary with error handling and retries"""
        try:
            await self._check_rate_limit()
            return await self._make_request_with_retries(text)
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    async def _make_request_with_retries(
        self, prompt: str, **kwargs: Dict[str, Any]
    ) -> str:
        """Make request with retry logic"""
        last_exception = None
        backoff = self.rate_limit_config.retry_delay

        for attempt in range(self.rate_limit_config.max_retries):
            try:
                return await self._make_request(prompt, **kwargs)
            except RateLimitError as e:
                last_exception = e
                if attempt < self.rate_limit_config.max_retries - 1:
                    await asyncio.sleep(backoff)
                backoff *= 2
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.rate_limit_config.max_retries - 1:
                    await asyncio.sleep(backoff)
                backoff *= 1.5

        raise last_exception or ProviderError("Maximum retry attempts exceeded")

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make API request with error handling"""
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

        try:
            async with self._session.post(
                f"{self.API_BASE}/chat/completions",
                headers=headers,
                json=data,
                raise_for_status=True,
            ) as response:
                if response.status == 429:
                    retry_after = float(
                        response.headers.get(
                            "Retry-After", self.rate_limit_config.retry_delay
                        )
                    )
                    raise RateLimitError(retry_after=retry_after)

                response_data = await response.json()
                return response_data["choices"][0]["message"]["content"]

        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                raise AuthenticationError("Invalid API key")
            raise ProviderError(f"API error (status {e.status}): {str(e)}")
        except aiohttp.ClientError as e:
            raise ProviderError(f"Network error: {str(e)}")
        except (KeyError, IndexError, ValueError) as e:
            raise ProviderError(f"Invalid API response: {str(e)}")

    async def close(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()

    def __del__(self):
        """Ensure cleanup on deletion"""
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass  # We're in cleanup, so we can't raise exceptions

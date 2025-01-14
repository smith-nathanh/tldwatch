import asyncio
import logging
import re
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import TCPConnector

from .base import (
    AuthenticationError,
    BaseProvider,
    ProviderError,
    RateLimitConfig,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    """Groq API provider implementation optimized for performance"""

    API_BASE = "https://api.groq.com/openai/v1"

    # Optimization constants
    MAX_CONNECTIONS = 100
    KEEPALIVE_TIMEOUT = 30
    REQUEST_TIMEOUT = aiohttp.ClientTimeout(
        total=30, connect=5, sock_connect=5, sock_read=25
    )

    CONTEXT_WINDOWS = {
        "mixtral-8x7b-32768": 32768,
        "llama-3.3-70b-versatile": 128000,
        "llama-3.1-8b-instant": 128000,
        "llama-guard-3-8b": 8192,
        "llama3-70b-8192": 8192,
        "llama3-8b-8192": 8192,
        "gemma2-9b-it": 8192,
    }

    def __init__(
        self,
        model: str = "mixtral-8x7b-32768",
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
            raise AuthenticationError("Groq API key is required")

        # Initialize session-related attributes
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._retry_count = 0

    def _get_provider_name(self) -> str:
        return "groq"

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Groq based on their documentation"""
        return RateLimitConfig(
            requests_per_minute=600,  # 14400 RPD ≈ 600 RPM
            tokens_per_minute=18000,  # From documentation
            max_retries=5,
            retry_delay=2.0,
        )

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 15  # Conservative limit based on Groq's guidelines

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        return self.CONTEXT_WINDOWS[self.model]

    def count_tokens(self, text: str) -> int:
        """Approximate token counting for LLaMA-based models"""
        if not text:
            return 0

        text = text.strip()
        words = len(re.findall(r"\b\w+(?:[-\']\w+)*\b", text))
        punctuation = len(re.findall(r"[^\w\s]", text))
        numbers = len(re.findall(r"\d+", text))

        # Approximate LLaMA tokenization behavior
        estimated_tokens = (
            words * 1.3  # Average word-to-token ratio
            + punctuation  # Most punctuation is its own token
            + numbers * 0.5  # Numbers often combine into single tokens
        )

        return int(estimated_tokens * 1.1)  # Add safety margin

    async def generate_summary(self, text: str) -> str:
        """Generate a summary with optimized async handling"""
        try:
            self._check_rate_limit()
            self._retry_count = 0  # Reset retry counter before new request
            response = await self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an optimized aiohttp session"""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        connector=TCPConnector(
                            limit=self.MAX_CONNECTIONS,
                            keepalive_timeout=self.KEEPALIVE_TIMEOUT,
                            force_close=False,
                        ),
                        timeout=self.REQUEST_TIMEOUT,
                    )
        return self._session

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make an optimized async request to Groq's API with error handling and retries"""
        last_exception = None
        session = await self._get_session()

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

        while self._retry_count < self.rate_limit_config.max_retries:
            try:
                async with session.post(
                    f"{self.API_BASE}/chat/completions",
                    headers=headers,
                    json=data,
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key")
                    elif response.status == 429:
                        retry_after = float(
                            response.headers.get(
                                "Retry-After", self.rate_limit_config.retry_delay
                            )
                        )
                        logger.warning(
                            f"Groq API rate limit hit. Need to wait {retry_after} seconds."
                        )
                        # Instead of raising immediately, we'll handle the retry
                        self._retry_count += 1
                        if self._retry_count >= self.rate_limit_config.max_retries:
                            raise RateLimitError(retry_after=retry_after)
                        await asyncio.sleep(retry_after)
                        continue

                    elif response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Groq API error: {error_text}")
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

    async def close(self) -> None:
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

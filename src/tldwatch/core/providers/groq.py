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
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Initialize session-related attributes
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

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
            raise AuthenticationError("Groq API key is required")

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
        self._check_rate_limit()  # Use base class rate limiting
        try:
            response = await self._make_request(text)
            self._record_request()  # Record the request in base class
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
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
        return self._session

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make an optimized async request to Groq's API with rate limit logging"""
        logger.debug(
            f"Rate limit state: {len(self._request_timestamps)} requests in last minute, limit is {self.rate_limit_config.requests_per_minute}"
        )
        session = await self._get_session()

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

        tries = 0
        while tries < self.rate_limit_config.max_retries:
            try:
                async with session.post(
                    f"{self.API_BASE}/chat/completions",
                    json=data,
                ) as response:
                    response_json = await response.json()
                    headers = dict(response.headers)

                    # Log API response details
                    # logger.debug(
                    #    f"Groq API Response - Status: {response.status}, Headers: {headers}"
                    # )

                    if response.status == 429:
                        retry_after = float(
                            headers.get(
                                "Retry-After", self.rate_limit_config.retry_delay
                            )
                        )
                        logger.warning(
                            f"Groq API rate limit hit. Need to wait {retry_after} seconds. Headers: {headers}"
                        )
                        raise RateLimitError(retry_after=retry_after)

                    if response.status != 200:
                        error_msg = response_json.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        logger.error(
                            f"Groq API error: {error_msg}. Status: {response.status}, Headers: {headers}"
                        )
                        response.raise_for_status()

                    # Log successful response details
                    logger.debug("Groq API request successful")

                    return response_json["choices"][0]["message"]["content"]
            except RateLimitError as e:
                tries += 1
                if tries >= self.rate_limit_config.max_retries:
                    raise
                await asyncio.sleep(e.retry_after or self.rate_limit_config.retry_delay)
            except aiohttp.ClientError as e:
                tries += 1
                if tries >= self.rate_limit_config.max_retries:
                    raise ProviderError(
                        f"Network error after {tries} retries: {str(e)}"
                    )
                await asyncio.sleep(self.rate_limit_config.retry_delay * tries)

    async def close(self) -> None:
        """Cleanup resources"""
        if hasattr(self, "_session") and self._session and not self._session.closed:
            await self._session.close()

    def __del__(self):
        """Ensure the session is closed when the provider is deleted"""
        if hasattr(self, "_session") and self._session and not self._session.closed:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass  # We're in cleanup, so we can't raise exceptions

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


class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider implementation"""

    API_BASE = "https://api.deepseek.com/v1"

    CONTEXT_WINDOWS = {
        "deepseek-chat": 65536,  # 64K context window
    }

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )

        if not self.api_key:
            raise AuthenticationError("DeepSeek API key is required")

        self._session = aiohttp.ClientSession()
        self._retry_count = 0

    def _get_provider_name(self) -> str:
        return "deepseek"

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for DeepSeek"""
        return RateLimitConfig(
            requests_per_minute=600,  # Conservative default
            tokens_per_minute=90000,  # Conservative default
            max_retries=3,
            retry_delay=1.0,
            timeout=60.0,
        )

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

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 10  # Conservative default

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        return self.CONTEXT_WINDOWS.get(self.model, 65536)

    async def generate_summary(self, text: str) -> str:
        """Generate a summary using DeepSeek's API"""
        try:
            self._check_rate_limit()
            self._retry_count = 0  # Reset retry counter before new request
            response = await self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make an async request to DeepSeek's API with error handling and retries"""
        last_exception = None

        while self._retry_count < self.rate_limit_config.max_retries:
            try:
                if self._session is None:
                    self._session = aiohttp.ClientSession()

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

                async with self._session.post(
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

    async def close(self):
        """Close the aiohttp session"""
        if self._session is not None:
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
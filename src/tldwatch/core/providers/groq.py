import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

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

    # Model context windows and TPM limits
    MODEL_CONFIGS = {
        "mixtral-8x7b-32768": {"context_window": 32768, "tpm": 5000},
        "llama-3.3-70b-versatile": {"context_window": 128000, "tpm": 6000},
        "llama-3.1-8b-instant": {"context_window": 128000, "tpm": 20000},
        "llama-guard-3-8b": {"context_window": 8192, "tpm": 15000},
        "llama3-70b-8192": {"context_window": 8192, "tpm": 6000},
        "llama3-8b-8192": {"context_window": 8192, "tpm": 30000},
        "gemma2-9b-it": {"context_window": 8192, "tpm": 14400},
    }

    def __init__(
        self,
        model: str = "mixtral-8x7b-32768",
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        if model not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model. Choose from: {', '.join(self.MODEL_CONFIGS.keys())}"
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

        self._token_usage: List[
            Tuple[float, int]
        ] = []  # [(timestamp, token_count), ...]
        self._retry_count = 0

    def _get_provider_name(self) -> str:
        return "groq"

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Groq based on their documentation"""
        return RateLimitConfig(
            requests_per_minute=30,  # From documentation
            tokens_per_minute=self.MODEL_CONFIGS[self.model][
                "tpm"
            ],  # Model-specific TPM
            max_retries=5,
            retry_delay=2.0,
            timeout=120.0,
        )

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 15  # Conservative limit based on Groq's guidelines

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        return self.MODEL_CONFIGS[self.model]["context_window"]

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

            # Check token count before making request
            estimated_tokens = self.count_tokens(text)
            if estimated_tokens > self.rate_limit_config.tokens_per_minute:
                raise ProviderError(
                    f"Request exceeds token limit. Estimated tokens: {estimated_tokens}, "
                    f"Limit: {self.rate_limit_config.tokens_per_minute}"
                )

            response = await self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    def _clean_token_usage(self) -> None:
        """Remove token usage older than 60 seconds"""
        current_time = time.time()
        minute_ago = current_time - 60
        self._token_usage = [
            (ts, count) for ts, count in self._token_usage if ts > minute_ago
        ]

    def _get_current_token_usage(self) -> int:
        """Get total token usage in current window"""
        self._clean_token_usage()
        return sum(count for _, count in self._token_usage)

    def _calculate_wait_time(self, requested_tokens: int) -> float:
        """Calculate wait time based on token usage history"""
        current_time = time.time()
        self._clean_token_usage()

        if not self._token_usage:
            return 1.0  # No history, short wait

        # Sort by timestamp
        usage = sorted(self._token_usage)
        total_tokens = self._get_current_token_usage()
        tpm_limit = self.rate_limit_config.tokens_per_minute

        # If adding requested tokens would exceed limit
        if total_tokens + requested_tokens > tpm_limit:
            # Calculate how many old tokens need to expire
            tokens_to_expire = total_tokens + requested_tokens - tpm_limit

            # Find how far back we need to go to expire enough tokens
            tokens_so_far = 0
            for timestamp, count in usage:
                tokens_so_far += count
                if tokens_so_far >= tokens_to_expire:
                    # Wait until this timestamp's tokens expire
                    wait_time = (timestamp + 60) - current_time
                    # Add small buffer and bounds
                    return min(max(wait_time * 1.1, 1.0), 60.0)

        return 1.0  # Default short wait

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        last_exception = None
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

        estimated_tokens = self.count_tokens(prompt)
        connector = aiohttp.TCPConnector(
            limit=self.MAX_CONNECTIONS,
            keepalive_timeout=self.KEEPALIVE_TIMEOUT,
        )

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=self.REQUEST_TIMEOUT,
        ) as session:
            while self._retry_count < self.rate_limit_config.max_retries:
                try:
                    async with session.post(
                        f"{self.API_BASE}/chat/completions",
                        headers=headers,
                        json=data,
                    ) as response:
                        if response.status == 401:
                            raise AuthenticationError("Invalid API key")
                        elif response.status == 429 or response.status == 413:
                            error_data = await response.json()
                            error_message = error_data.get("error", {}).get(
                                "message", ""
                            )

                            if "tokens per minute" in error_message.lower():
                                # Extract requested tokens from error if possible
                                try:
                                    requested = int(
                                        re.search(
                                            r"Requested (\d+)", error_message
                                        ).group(1)
                                    )
                                except (AttributeError, ValueError):
                                    requested = estimated_tokens

                                retry_after = self._calculate_wait_time(requested)
                                logger.warning(
                                    f"Groq API TPM limit hit. Waiting {retry_after:.1f} seconds before retry."
                                )
                            else:
                                retry_after = float(
                                    response.headers.get(
                                        "Retry-After",
                                        self.rate_limit_config.retry_delay,
                                    )
                                )
                                logger.warning(
                                    f"Groq API rate limit hit. Need to wait {retry_after} seconds."
                                )

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
                        # Record successful request token usage
                        current_time = time.time()
                        self._token_usage.append((current_time, estimated_tokens))
                        return response_data["choices"][0]["message"]["content"]

                except RateLimitError as e:
                    last_exception = e
                    raise
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

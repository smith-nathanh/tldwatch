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


class GoogleProvider(BaseProvider):
    """Provider implementation for Google's Gemini API"""

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    CONTEXT_WINDOWS = {
        "gemini-2.0-flash": {"input": 1048576, "output": 8192},
        "gemini-1.5-flash": {"input": 1048576, "output": 8192},
        "gemini-1.5-flash-8b": {"input": 1048576, "output": 8192},
        "gemini-1.5-pro": {"input": 2097152, "output": 8192},
    }

    RATE_LIMITS = {
        "gemini-2.0-flash": {"free_rpm": 10, "paid_rpm": None, "tpm": 4000000},
        "gemini-1.5-flash": {"free_rpm": 15, "paid_rpm": 2000, "tpm": 4000000},
        "gemini-1.5-flash-8b": {"free_rpm": 15, "paid_rpm": 4000, "tpm": 4000000},
        "gemini-1.5-pro": {"free_rpm": 2, "paid_rpm": 1000, "tpm": 4000000},
    }

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
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

        if not self.api_key:
            raise AuthenticationError("Google API key is required")

        self._retry_count = 0

    def _get_provider_name(self) -> str:
        return "google"

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits based on the selected model"""
        model_limits = self.RATE_LIMITS[self.model]
        return RateLimitConfig(
            requests_per_minute=model_limits["free_rpm"]
            // 2,  # Half of free tier limit
            tokens_per_minute=model_limits["tpm"],
            max_retries=3,
            retry_delay=5.0,  # 5 seconds between retries
            timeout=60.0,
        )

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        # Conservative concurrent request limit based on RPM
        return min(5, self.rate_limit_config.requests_per_minute // 4)

    @property
    def context_window(self) -> int:
        """Return input context window size for the current model"""
        return self.CONTEXT_WINDOWS[self.model]["input"]

    def count_tokens(self, text: str) -> int:
        """More accurate token count approximation for Gemini models"""
        if not text:
            return 0

        # Count bytes (most tokens are 2-4 bytes)
        byte_count = len(text.encode("utf-8"))
        token_estimate = byte_count // 3

        # Add margin for special characters and tokenization overhead
        special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
        token_estimate += special_chars

        return int(token_estimate * 1.1)  # 10% safety margin

    async def generate_summary(self, text: str) -> str:
        """Generate a summary using Gemini API"""
        try:
            self._check_rate_limit()
            self._retry_count = 0
            response = await self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make an async request to Gemini API with error handling and retries"""
        last_exception = None
        base_delay = 60  # Base delay of 1 minute between retries

        while self._retry_count < self.rate_limit_config.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {"Content-Type": "application/json"}

                    data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "safetySettings": [],
                        "generationConfig": {
                            "temperature": self.temperature,
                            "candidateCount": 1,
                            "stopSequences": [],
                            "maxOutputTokens": self.CONTEXT_WINDOWS[self.model][
                                "output"
                            ],
                        },
                    }

                    url = f"{self.API_BASE}/{self.model}:generateContent?key={self.api_key}"

                    async with session.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        if response.status == 429:
                            self._retry_count += 1
                            retry_delay = base_delay + (
                                self._retry_count * 10
                            )  # Progressive backoff
                            await asyncio.sleep(retry_delay)
                            continue

                        response_data = await response.json()

                        if "error" in response_data:
                            error = response_data["error"]
                            if error.get("code") == 429:
                                self._retry_count += 1
                                retry_delay = base_delay + (self._retry_count * 10)
                                await asyncio.sleep(retry_delay)
                                continue
                            raise ProviderError(
                                f"API error: {error.get('message', str(error))}"
                            )

                        try:
                            text = response_data["candidates"][0]["content"]["parts"][
                                0
                            ]["text"]
                            await asyncio.sleep(
                                4
                            )  # Add spacing between successful requests
                            return text
                        except (KeyError, IndexError):
                            raise ProviderError("Invalid API response structure")
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

                        # Extract text from Gemini response structure
                        try:
                            text = response_data["candidates"][0]["content"]["parts"][
                                0
                            ]["text"]
                            return text
                        except (KeyError, IndexError):
                            raise ProviderError("Invalid API response structure")

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

        raise last_exception or ProviderError("Maximum retry attempts exceeded")

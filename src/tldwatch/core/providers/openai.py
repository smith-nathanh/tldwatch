from typing import Any, Dict, Optional

import aiohttp
import tiktoken

from .base import (
    AuthenticationError,
    BaseProvider,
    ProviderError,
    RateLimitConfig,
    RateLimitError,
)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation with proper async handling"""

    API_BASE = "https://api.openai.com/v1"

    # Model context windows
    CONTEXT_WINDOWS = {
        "gpt-3.5-turbo": 16385,
        "gpt-4o": 8192,
        "gpt-4o-32k": 32768,
        "gpt-4o-turbo-preview": 128000,
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )
        if not api_key:
            raise AuthenticationError("OpenAI API key is required")
        self._encoding = tiktoken.encoding_for_model(model)
        self._retry_count = 0
        self._session = None

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for OpenAI"""
        return RateLimitConfig(
            requests_per_minute=3500,
            tokens_per_minute=180000,
            max_retries=3,
            retry_delay=1.0,
            timeout=120.0,  # 2 minutes base timeout
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model-specific tokenizer"""
        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            raise ProviderError(f"Token counting error: {str(e)}")

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 15

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        return self.CONTEXT_WINDOWS.get(self.model, 8192)

    async def generate_summary(self, text: str) -> str:
        """Generate a summary using OpenAI's chat completion API"""
        try:
            self._check_rate_limit()
            self._retry_count = 0  # Reset retry counter before new request
            response = await self._make_request(text)
            self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    async def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make an async request to OpenAI's API with error handling"""
        if self._retry_count >= self.rate_limit_config.max_retries:
            raise ProviderError("Maximum retry attempts exceeded")

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

        try:
            # Create a new session if we don't have one
            if self._session is None:
                self._session = aiohttp.ClientSession()

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
                    raise RateLimitError(retry_after=retry_after)
                elif response.status != 200:
                    error_text = await response.text()
                    raise ProviderError(
                        f"API error (status {response.status}): {error_text}"
                    )

                response_data = await response.json()
                return response_data["choices"][0]["message"]["content"]

        except aiohttp.ClientError as e:
            raise ProviderError(f"Network error: {str(e)}")
        except (KeyError, IndexError, ValueError) as e:
            raise ProviderError(f"Invalid API response: {str(e)}")

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

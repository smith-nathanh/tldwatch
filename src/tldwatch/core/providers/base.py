import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting settings"""

    requests_per_minute: int
    tokens_per_minute: Optional[int] = None
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay in seconds
    timeout: float = 60.0  # Base timeout in seconds for requests


class ProviderError(Exception):
    """Base exception class for provider errors"""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded"""

    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        message = (
            f"Rate limit exceeded. Retry after {retry_after} seconds"
            if retry_after
            else "Rate limit exceeded"
        )
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised when authentication fails"""

    pass


class BaseProvider(ABC):
    """Base class for all LLM providers"""

    # Class-level mapping of providers to env vars
    _ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None,  # Local provider doesn't need API key
    }

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.rate_limit_config = rate_limit_config or self._default_rate_limit_config()
        self.use_full_context = use_full_context

        # API key assignment
        self.api_key = self._get_api_key()

        # Rate limiting state
        self._request_timestamps: list[float] = []
        self._last_request_time: Optional[datetime] = None

    async def close(self):
        """Close any resources held by the provider"""
        pass

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables"""
        provider = self._get_provider_name().lower()
        env_var = self._ENV_VARS.get(provider)
        return os.environ.get(env_var) if env_var else None

    @classmethod
    def register_provider(cls, provider_name: str, env_var: Optional[str]) -> None:
        """Register a new provider and its environment variable"""
        cls._ENV_VARS[provider_name.lower()] = env_var

    @abstractmethod
    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Return default rate limit configuration for the provider"""
        pass

    @abstractmethod
    def generate_summary(self, text: str) -> str:
        """Generate a summary for the given text"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text"""
        pass

    def _check_rate_limit(self) -> None:
        """Check if we're within rate limits"""
        if not self._request_timestamps:
            return

        # Clean old timestamps
        current_time = time.time()
        minute_ago = current_time - 60
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > minute_ago
        ]

        # Check requests per minute
        if len(self._request_timestamps) >= self.rate_limit_config.requests_per_minute:
            oldest_timestamp = self._request_timestamps[0]
            retry_after = 60 - (current_time - oldest_timestamp)
            raise RateLimitError(retry_after=retry_after)

    def _record_request(self) -> None:
        """Record a request for rate limiting purposes"""
        self._request_timestamps.append(time.time())
        self._last_request_time = datetime.now()

    @abstractmethod
    def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make a request to the provider's API"""
        pass

    def _handle_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Handle rate limit by waiting"""
        if retry_after is not None:
            time.sleep(retry_after)
        else:
            time.sleep(self.rate_limit_config.retry_delay)

    @property
    @abstractmethod
    def max_concurrent_requests(self) -> int:
        """Return the recommended max concurrent requests for this provider"""
        pass

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Return the context window size (in tokens) for the current model"""
        pass

    def can_use_full_context(self, text: str) -> bool:
        """Check if the text can fit in the model's context window"""
        return (
            self.count_tokens(text) <= self.context_window * 0.9
        )  # 90% to leave room for prompt

    @property
    def requests_remaining(self) -> int:
        """Return number of requests remaining in the current minute"""
        self._check_rate_limit()  # This cleans old timestamps
        return self.rate_limit_config.requests_per_minute - len(
            self._request_timestamps
        )

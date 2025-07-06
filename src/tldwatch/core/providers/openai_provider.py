"""
OpenAI provider implementation using the unified request handler.
"""

from typing import Any, Dict, Optional

import tiktoken

from .base_provider import AuthenticationError, BaseProvider, RateLimitConfig
from .config_loader import get_context_windows, get_default_model


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation with unified request handling"""

    API_BASE = "https://api.openai.com/v1"
    RESPONSE_KEY_PATH = ("choices", 0, "message", "content")

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Use config file for default model if not specified
        if model is None:
            model = get_default_model("openai")

        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )

        # Verify API key exists after BaseProvider initialization
        if not self.api_key:
            raise AuthenticationError("OpenAI API key is required")

        self._encoding = tiktoken.encoding_for_model(model)

    def _get_provider_name(self) -> str:
        return "openai"

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
            raise ValueError(f"Token counting error: {str(e)}")

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 20  # Increase concurrency if API allows

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("openai")
        return context_windows.get(self.model, 8192)

    def _prepare_request_headers(self) -> Dict[str, str]:
        """Prepare headers for OpenAI API request"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Prepare request data for OpenAI API"""
        return {
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

    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for OpenAI"""
        return "chat/completions"
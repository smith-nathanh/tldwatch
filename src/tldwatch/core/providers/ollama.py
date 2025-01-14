import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from .base import BaseProvider, ProviderError, RateLimitConfig


class OllamaConnectionError(ProviderError):
    """Raised when unable to connect to Ollama service"""

    pass


class OllamaProvider(BaseProvider):
    """Provider implementation optimized for local Ollama models

    This provider is designed specifically for local model inference, with:
    - Efficient connection and resource management
    - Hardware-aware concurrency limits
    - Optimized for GPU/CPU inference
    """

    API_BASE = "http://localhost:11434/api"

    # Default context windows for common models
    CONTEXT_WINDOWS = {
        "llama3.1:8b": 4096,
        "llama3.3:70b": 8192,  # need to verify
        "phi4:14b": 8192,
    }

    def __init__(
        self,
        model: str = "mistral",
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Initialize session with optimized pooling
        self._session = requests.Session()

        # Configure connection pool based on concurrency limit
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self._get_optimal_concurrency(),
            pool_maxsize=self._get_optimal_concurrency(),
            pool_block=True,  # Block when pool is full to prevent overload
        )
        self._session.mount("http://", adapter)

        # Thread safety for request tracking
        self._request_lock = threading.Lock()

        # Check if Ollama is running before proceeding
        if not self._is_ollama_running():
            self._session.close()
            raise OllamaConnectionError(
                "Could not connect to Ollama service. "
                "Please ensure Ollama is installed and running on localhost:11434"
            )

        # Get available models and validate requested model
        available_models = self._get_available_models()
        if not available_models:
            self._session.close()
            raise OllamaConnectionError("Could not fetch available models from Ollama")

        if model not in available_models:
            self._session.close()
            raise ValueError(
                f"Model '{model}' not available locally. "
                f"Available models: {', '.join(available_models)}"
            )

        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )

        self._retry_count = 0

    def _get_provider_name(self) -> str:
        return "ollama"

    def _get_optimal_concurrency(self) -> int:
        """Determine optimal concurrency based on hardware

        This checks for GPU availability and system resources to suggest
        an appropriate concurrency limit for local inference.
        """
        try:
            # Check NVIDIA GPU availability via nvidia-smi
            has_gpu = os.system("nvidia-smi >/dev/null 2>&1") == 0
        except:
            has_gpu = False

        # Get CPU count, but be conservative
        cpu_count = os.cpu_count() or 1

        if has_gpu:
            # With GPU, we can handle more concurrent requests
            # Still be conservative to avoid VRAM issues
            return min(4, cpu_count)
        else:
            # CPU-only mode should be more conservative
            return min(2, cpu_count)

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum concurrent requests based on hardware capability"""
        return self._get_optimal_concurrency()

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits based on local hardware capabilities"""
        return RateLimitConfig(
            requests_per_minute=60
            * self.max_concurrent_requests,  # Scale with concurrency
            tokens_per_minute=None,  # Limited by hardware, not rate
            max_retries=2,  # Local service needs fewer retries
            retry_delay=0.5,  # Faster retry for local service
            timeout=60.0,  # Local inference can take time
        )

    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = self._session.get(f"{self.API_BASE}/tags", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _get_available_models(self) -> List[str]:
        """Get list of available models from local Ollama installation"""
        try:
            response = self._session.get(f"{self.API_BASE}/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.exceptions.RequestException:
            return []

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        base_model = self.model.split(":")[0].lower()

        for model_prefix, window in self.CONTEXT_WINDOWS.items():
            if base_model.startswith(model_prefix):
                return window
        return 2048  # Conservative default for unknown models

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Ollama models.
        Using a conservative estimate of 4 characters per token.
        """
        return len(text) // 4 + 1

    async def generate_summary(self, text: str) -> str:
        """Generate a summary using local Ollama model.
        Method is async to match interface but implementation is synchronous
        since we're dealing with local compute."""
        try:
            with self._request_lock:
                self._check_rate_limit()
                self._retry_count = 0

            # Direct synchronous call - no need for asyncio.to_thread since
            # we're CPU/GPU bound locally
            response = self._make_request(text)

            with self._request_lock:
                self._record_request()
            return response
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make a request to local Ollama service with retries"""
        last_exception = None
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise summaries.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, **kwargs},
        }

        while self._retry_count < self.rate_limit_config.max_retries:
            try:
                response = self._session.post(
                    f"{self.API_BASE}/chat",
                    json=data,
                    timeout=self.rate_limit_config.timeout,
                )

                if response.status_code != 200:
                    self._retry_count += 1
                    if self._retry_count >= self.rate_limit_config.max_retries:
                        raise ProviderError(
                            f"Ollama API error (status {response.status_code}): {response.text}"
                        )
                    time.sleep(self.rate_limit_config.retry_delay * self._retry_count)
                    continue

                try:
                    return response.json()["message"]["content"]
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    raise ProviderError(f"Unexpected API response format: {str(e)}")

            except requests.exceptions.RequestException as e:
                last_exception = e
                self._retry_count += 1
                if self._retry_count >= self.rate_limit_config.max_retries:
                    raise OllamaConnectionError(
                        f"Error connecting to Ollama after {self._retry_count} retries: {str(e)}"
                    )
                time.sleep(self.rate_limit_config.retry_delay * self._retry_count)

        raise last_exception or ProviderError("Maximum retry attempts exceeded")

    def __del__(self):
        """Ensure the session is closed when the provider is deleted"""
        if hasattr(self, "_session"):
            self._session.close()

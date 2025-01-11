import json
from typing import Any, Dict, List, Optional

import requests

from .base import BaseProvider, ProviderError, RateLimitConfig


class OllamaConnectionError(ProviderError):
    """Raised when unable to connect to Ollama service"""

    pass


class OllamaProvider(BaseProvider):
    """Ollama local API provider implementation"""

    API_BASE = "http://localhost:11434/api"

    # Default context windows for common models
    CONTEXT_WINDOWS = {
        "llama2": 4096,
        "mistral": 8192,
        "mixtral": 32768,
        "neural-chat": 8192,
        "codellama": 16384,
        "phi": 2048,
        "gemma": 8192,
    }

    def __init__(
        self,
        model: str = "mistral",
        api_key: Optional[
            str
        ] = None,  # Not used for Ollama but kept for interface consistency
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Check if Ollama is running before proceeding
        if not self._is_ollama_running():
            raise OllamaConnectionError(
                "Could not connect to Ollama service. "
                "Please ensure Ollama is installed and running on localhost:11434"
            )

        # Get available models and validate requested model
        available_models = self._get_available_models()
        if not available_models:
            raise OllamaConnectionError("Could not fetch available models from Ollama")

        if model not in available_models:
            raise ValueError(
                f"Model '{model}' not available locally. "
                f"Available models: {', '.join(available_models)}"
            )

        super().__init__(
            model, api_key, temperature, rate_limit_config, use_full_context
        )

    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Ollama - primarily limited by local hardware"""
        return RateLimitConfig(
            requests_per_minute=60,  # Conservative default
            tokens_per_minute=None,  # Limited by hardware, not by rate
            max_retries=2,
            retry_delay=0.5,
        )

    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.API_BASE}/tags", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _get_available_models(self) -> List[str]:
        """Get list of available models from local Ollama installation"""
        try:
            response = requests.get(f"{self.API_BASE}/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.exceptions.RequestException:
            return []

    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        # Extract base model name (e.g., 'llama2' from 'llama2:latest')
        base_model = self.model.split(":")[0].lower()

        # Find matching context window or use conservative default
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

    def generate_summary(self, text: str) -> str:
        """Generate a summary using local Ollama model"""
        try:
            self._check_rate_limit()
            response = self._make_request(text)
            self._record_request()
            return response
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Error connecting to Ollama: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Error generating summary: {str(e)}")

    def _make_request(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """Make a request to local Ollama service"""
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

        response = requests.post(
            f"{self.API_BASE}/chat",
            json=data,
            timeout=30,  # Longer timeout for local inference
        )

        if response.status_code != 200:
            raise ProviderError(f"Ollama API error: {response.text}")

        try:
            return response.json()["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ProviderError(f"Unexpected API response format: {str(e)}")

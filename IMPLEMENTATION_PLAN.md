# Implementation Plan: Remaining Providers

This document outlines the plan for implementing the remaining providers using the new consolidated architecture.

## Providers to Implement

1. ✅ OpenAI Provider (completed)
2. ✅ Anthropic Provider (completed)
3. Google (Gemini) Provider
4. Groq Provider
5. Cerebras Provider
6. DeepSeek Provider
7. Ollama Provider

## Implementation Steps for Each Provider

For each provider, follow these steps:

1. Create a new file `{provider_name}_provider.py` in the `src/tldwatch/core/providers` directory
2. Implement the provider class extending `BaseProvider`
3. Define the API_BASE and RESPONSE_KEY_PATH constants
4. Implement all required abstract methods
5. Add the provider to the registry in `provider_registry.py`
6. Add tests for the provider

## Google (Gemini) Provider

```python
# src/tldwatch/core/providers/google_provider.py

from typing import Any, Dict, Optional

from .base_provider import AuthenticationError, BaseProvider, RateLimitConfig
from .config_loader import get_context_windows, get_default_model

class GoogleProvider(BaseProvider):
    """Google Gemini API provider implementation"""
    
    API_BASE = "https://generativelanguage.googleapis.com/v1"
    RESPONSE_KEY_PATH = ("candidates", 0, "content", "parts", 0, "text")
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Use config file for default model if not specified
        if model is None:
            model = get_default_model("google")
            
        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )
        
        # Verify API key exists after BaseProvider initialization
        if not self.api_key:
            raise AuthenticationError("Google API key is required")
    
    def _get_provider_name(self) -> str:
        return "google"
        
    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Google Gemini"""
        # Get model-specific rate limits from config
        return RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=250000,
            max_retries=3,
            retry_delay=1.0,
            timeout=60.0,
        )
        
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Gemini models.
        This is an approximation - actual token count may vary.
        """
        # Simple approximation: 4 characters ≈ 1 token
        return len(text) // 4
        
    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 10
        
    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("google")
        model_config = context_windows.get(self.model, {})
        return model_config.get("input", 8192)
        
    def _prepare_request_headers(self) -> Dict[str, str]:
        """Prepare headers for Google API request"""
        return {
            "Content-Type": "application/json",
        }
        
    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Prepare request data for Google API"""
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                **kwargs,
            }
        }
        
    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for Google Gemini"""
        return f"models/{self.model}:generateContent?key={self.api_key}"
```

## Groq Provider

```python
# src/tldwatch/core/providers/groq_provider.py

from typing import Any, Dict, Optional

import tiktoken

from .base_provider import AuthenticationError, BaseProvider, RateLimitConfig
from .config_loader import get_context_windows, get_default_model

class GroqProvider(BaseProvider):
    """Groq API provider implementation"""
    
    API_BASE = "https://api.groq.com/v1"
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
            model = get_default_model("groq")
            
        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )
        
        # Verify API key exists after BaseProvider initialization
        if not self.api_key:
            raise AuthenticationError("Groq API key is required")
            
        # Initialize tokenizer (Groq uses OpenAI-compatible tokenization)
        try:
            self._encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def _get_provider_name(self) -> str:
        return "groq"
        
    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Groq"""
        return RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=None,  # Groq doesn't specify TPM limits
            max_retries=3,
            retry_delay=1.0,
            timeout=60.0,
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
        return 20
        
    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("groq")
        return context_windows.get(self.model, 8192)
        
    def _prepare_request_headers(self) -> Dict[str, str]:
        """Prepare headers for Groq API request"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Prepare request data for Groq API"""
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
        """Get the API endpoint for Groq"""
        return "chat/completions"
```

## Cerebras Provider

```python
# src/tldwatch/core/providers/cerebras_provider.py

from typing import Any, Dict, Optional

from .base_provider import AuthenticationError, BaseProvider, RateLimitConfig
from .config_loader import get_context_windows, get_default_model

class CerebrasProvider(BaseProvider):
    """Cerebras API provider implementation"""
    
    API_BASE = "https://api.cerebras.ai/v1"
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
            model = get_default_model("cerebras")
            
        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )
        
        # Verify API key exists after BaseProvider initialization
        if not self.api_key:
            raise AuthenticationError("Cerebras API key is required")
    
    def _get_provider_name(self) -> str:
        return "cerebras"
        
    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Cerebras"""
        return RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=None,
            max_retries=3,
            retry_delay=1.0,
            timeout=60.0,
        )
        
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Cerebras models.
        This is an approximation - actual token count may vary.
        """
        # Simple approximation: 4 characters ≈ 1 token
        return len(text) // 4
        
    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 10
        
    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("cerebras")
        return context_windows.get(self.model, 8192)
        
    def _prepare_request_headers(self) -> Dict[str, str]:
        """Prepare headers for Cerebras API request"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Prepare request data for Cerebras API"""
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
        """Get the API endpoint for Cerebras"""
        return "chat/completions"
```

## DeepSeek Provider

```python
# src/tldwatch/core/providers/deepseek_provider.py

from typing import Any, Dict, Optional

from .base_provider import AuthenticationError, BaseProvider, RateLimitConfig
from .config_loader import get_context_windows, get_default_model

class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider implementation"""
    
    API_BASE = "https://api.deepseek.com/v1"
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
            model = get_default_model("deepseek")
            
        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )
        
        # Verify API key exists after BaseProvider initialization
        if not self.api_key:
            raise AuthenticationError("DeepSeek API key is required")
    
    def _get_provider_name(self) -> str:
        return "deepseek"
        
    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for DeepSeek"""
        return RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=None,
            max_retries=3,
            retry_delay=1.0,
            timeout=60.0,
        )
        
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for DeepSeek models.
        This is an approximation - actual token count may vary.
        """
        # Simple approximation: 4 characters ≈ 1 token
        return len(text) // 4
        
    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 10
        
    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("deepseek")
        return context_windows.get(self.model, 8192)
        
    def _prepare_request_headers(self) -> Dict[str, str]:
        """Prepare headers for DeepSeek API request"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Prepare request data for DeepSeek API"""
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
        """Get the API endpoint for DeepSeek"""
        return "chat/completions"
```

## Ollama Provider

```python
# src/tldwatch/core/providers/ollama_provider.py

import json
from typing import Any, Dict, Optional

from .base_provider import BaseProvider, ProviderError, RateLimitConfig
from .config_loader import get_context_windows, get_default_model

class OllamaProvider(BaseProvider):
    """Ollama API provider implementation for local LLM inference"""
    
    API_BASE = "http://localhost:11434/api"
    RESPONSE_KEY_PATH = ("response",)
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ):
        # Use config file for default model if not specified
        if model is None:
            model = get_default_model("ollama")
            
        super().__init__(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )
    
    def _get_provider_name(self) -> str:
        return "ollama"
        
    def _default_rate_limit_config(self) -> RateLimitConfig:
        """Default rate limits for Ollama (local inference)"""
        return RateLimitConfig(
            requests_per_minute=30,  # Conservative limit for local inference
            tokens_per_minute=None,
            max_retries=3,
            retry_delay=1.0,
            timeout=120.0,  # Longer timeout for local inference
        )
        
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Ollama models.
        This is an approximation - actual token count may vary.
        """
        # Simple approximation: 4 characters ≈ 1 token
        return len(text) // 4
        
    @property
    def max_concurrent_requests(self) -> int:
        """Maximum recommended concurrent requests"""
        return 1  # Local inference is typically resource-constrained
        
    @property
    def context_window(self) -> int:
        """Return the context window size for the current model"""
        context_windows = get_context_windows("ollama")
        return context_windows.get(self.model, 8192)
        
    def _prepare_request_headers(self) -> Dict[str, str]:
        """Prepare headers for Ollama API request"""
        return {
            "Content-Type": "application/json",
        }
        
    def _prepare_request_data(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Prepare request data for Ollama API"""
        return {
            "model": self.model,
            "prompt": prompt,
            "system": "You are a helpful assistant that generates concise summaries.",
            "temperature": self.temperature,
            "stream": False,
            **kwargs,
        }
        
    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for Ollama"""
        return "generate"
```

## Update Provider Registry

After implementing all providers, update the provider registry:

```python
# src/tldwatch/core/providers/provider_registry.py

from .anthropic_provider import AnthropicProvider
from .base_provider import BaseProvider
from .cerebras_provider import CerebrasProvider
from .deepseek_provider import DeepSeekProvider
from .google_provider import GoogleProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .provider_factory import ProviderFactory

def register_all_providers() -> None:
    """Register all available providers with the factory"""
    # Define all available providers
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "groq": GroqProvider,
        "cerebras": CerebrasProvider,
        "deepseek": DeepSeekProvider,
        "ollama": OllamaProvider,
    }
    
    # Register each provider with the factory
    for name, provider_class in providers.items():
        ProviderFactory.register_provider(name, provider_class)
```

## Testing Plan

For each provider, create a test file in the `tests/providers` directory:

1. Test provider initialization
2. Test token counting
3. Test request preparation
4. Test response parsing
5. Test error handling

## Migration Timeline

1. Implement OpenAI and Anthropic providers (completed)
2. Implement Google and Groq providers (high priority)
3. Implement Cerebras and DeepSeek providers (medium priority)
4. Implement Ollama provider (low priority)
5. Update tests and documentation
6. Release new version
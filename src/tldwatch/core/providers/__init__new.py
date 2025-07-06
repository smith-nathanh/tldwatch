"""
Provider package for LLM API integrations.

This package provides a unified interface for interacting with various LLM providers,
with standardized request handling, error management, and configuration.
"""

# Import base classes and utilities
from .base_provider import (
    AuthenticationError,
    BaseProvider,
    ProviderError,
    RateLimitConfig,
    RateLimitError,
)
from .provider_factory import ProviderFactory
from .request_handler import RequestHandler

# Import provider implementations
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

# Import registry to ensure all providers are registered
from .provider_registry import register_all_providers

# Ensure all providers are registered
register_all_providers()

__all__ = [
    # Base classes
    "BaseProvider",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "RateLimitConfig",
    
    # Factory and utilities
    "ProviderFactory",
    "RequestHandler",
    
    # Provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
]
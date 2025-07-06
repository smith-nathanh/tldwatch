"""
Provider factory for creating and managing LLM providers.

This module provides a factory pattern for instantiating the appropriate provider
based on configuration, with unified error handling and configuration management.
"""

import logging
from typing import Dict, Optional, Type

from .base import BaseProvider, RateLimitConfig
from .config_loader import get_default_model

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating provider instances"""

    # Registry of available providers
    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider class with the factory.

        Args:
            name: Provider name (lowercase)
            provider_class: Provider class to register
        """
        cls._providers[name.lower()] = provider_class
        logger.debug(f"Registered provider: {name}")

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        rate_limit_config: Optional[RateLimitConfig] = None,
        use_full_context: bool = False,
    ) -> BaseProvider:
        """
        Create a provider instance based on the provider name.

        Args:
            provider_name: Name of the provider to create
            model: Model name (if None, uses provider default)
            temperature: Temperature for generation
            rate_limit_config: Custom rate limit configuration
            use_full_context: Whether to use the model's full context window

        Returns:
            Instantiated provider

        Raises:
            ValueError: If the provider is not registered
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Provider '{provider_name}' not registered. Available providers: {available}"
            )
        
        provider_class = cls._providers[provider_name]
        
        # Use default model from config if not specified
        if model is None:
            model = get_default_model(provider_name)
            logger.debug(f"Using default model for {provider_name}: {model}")
        
        # Create and return the provider instance
        return provider_class(
            model=model,
            temperature=temperature,
            rate_limit_config=rate_limit_config,
            use_full_context=use_full_context,
        )

    @classmethod
    def get_available_providers(cls) -> Dict[str, Type[BaseProvider]]:
        """Get a dictionary of all registered providers"""
        return cls._providers.copy()
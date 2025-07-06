"""
Provider registry for initializing and registering all available providers.

This module ensures all providers are properly registered with the factory.
"""

import logging
from typing import Dict, Type

from .anthropic_provider import AnthropicProvider
from .base_provider import BaseProvider
from .openai_provider import OpenAIProvider
from .provider_factory import ProviderFactory

logger = logging.getLogger(__name__)


def register_all_providers() -> None:
    """Register all available providers with the factory"""
    # Define all available providers
    providers: Dict[str, Type[BaseProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        # Add other providers here as they're implemented
    }
    
    # Register each provider with the factory
    for name, provider_class in providers.items():
        ProviderFactory.register_provider(name, provider_class)
        logger.debug(f"Registered provider: {name}")
    
    logger.info(f"Registered {len(providers)} providers")


# Initialize the registry when the module is imported
register_all_providers()
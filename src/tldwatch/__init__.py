from .core.summarizer import Summarizer
from .core.simple_summarizer import SimpleSummarizer, summarize_video
from .core.providers.unified_provider import UnifiedProvider, ChunkingStrategy
from .core.proxy_config import (
    TldwatchProxyConfig,
    create_webshare_proxy,
    create_generic_proxy,
    ProxyConfigError,
)

__all__ = [
    # Legacy interface (for backward compatibility)
    "Summarizer",
    
    # New simplified interface
    "SimpleSummarizer",
    "summarize_video",
    "UnifiedProvider", 
    "ChunkingStrategy",
    
    # Proxy configuration
    "TldwatchProxyConfig",
    "create_webshare_proxy",
    "create_generic_proxy",
    "ProxyConfigError",
]

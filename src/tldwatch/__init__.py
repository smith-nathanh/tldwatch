from .core.providers.unified_provider import ChunkingStrategy, UnifiedProvider
from .core.proxy_config import (
    ProxyConfigError,
    TldwatchProxyConfig,
    create_generic_proxy,
    create_webshare_proxy,
)
from .core.summarizer import Summarizer, summarize_video
from .utils.cache import SummaryCache, clear_cache, get_cache, get_cache_stats

__all__ = [
    # Main interface
    "Summarizer",
    "summarize_video",
    "UnifiedProvider",
    "ChunkingStrategy",
    # Cache management
    "SummaryCache",
    "get_cache",
    "clear_cache",
    "get_cache_stats",
    # Proxy configuration
    "TldwatchProxyConfig",
    "create_webshare_proxy",
    "create_generic_proxy",
    "ProxyConfigError",
]

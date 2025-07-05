"""
Proxy configuration module for tldwatch.

This module provides proxy configuration classes to work around IP blocking
when fetching YouTube transcripts. It supports Webshare rotating residential
proxies and generic HTTP/HTTPS/SOCKS proxies.
"""

import logging
from typing import Optional, Union

try:
    from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
except ImportError:
    # Fallback for older versions of youtube-transcript-api
    WebshareProxyConfig = None
    GenericProxyConfig = None

logger = logging.getLogger(__name__)


class ProxyConfigError(Exception):
    """Base exception for proxy configuration errors"""
    pass


class TldwatchProxyConfig:
    """
    Base proxy configuration class for tldwatch.
    
    This class wraps the youtube-transcript-api proxy configurations
    and provides a unified interface for proxy management.
    """
    
    def __init__(self, proxy_config: Optional[Union[object, dict]] = None):
        """
        Initialize proxy configuration.
        
        Args:
            proxy_config: Either a youtube-transcript-api proxy config object
                         or a dictionary with proxy configuration
        """
        self._proxy_config = proxy_config
        
    @property
    def proxy_config(self):
        """Get the underlying proxy configuration object"""
        return self._proxy_config
    
    @classmethod
    def create_webshare_config(
        cls,
        proxy_username: str,
        proxy_password: str,
        **kwargs
    ) -> "TldwatchProxyConfig":
        """
        Create a Webshare proxy configuration.
        
        Args:
            proxy_username: Webshare proxy username
            proxy_password: Webshare proxy password
            **kwargs: Additional arguments passed to WebshareProxyConfig
            
        Returns:
            TldwatchProxyConfig instance with Webshare configuration
            
        Raises:
            ProxyConfigError: If WebshareProxyConfig is not available
        """
        if WebshareProxyConfig is None:
            raise ProxyConfigError(
                "WebshareProxyConfig is not available. "
                "Please update youtube-transcript-api to a version that supports proxies."
            )
        
        if not proxy_username or not proxy_password:
            raise ProxyConfigError(
                "Both proxy_username and proxy_password are required for Webshare configuration"
            )
        
        try:
            webshare_config = WebshareProxyConfig(
                proxy_username=proxy_username,
                proxy_password=proxy_password,
                **kwargs
            )
            logger.info("Created Webshare proxy configuration")
            return cls(webshare_config)
        except Exception as e:
            raise ProxyConfigError(f"Failed to create Webshare proxy configuration: {str(e)}")
    
    @classmethod
    def create_generic_config(
        cls,
        http_url: Optional[str] = None,
        https_url: Optional[str] = None,
        **kwargs
    ) -> "TldwatchProxyConfig":
        """
        Create a generic proxy configuration.
        
        Args:
            http_url: HTTP proxy URL (e.g., "http://user:pass@proxy.example.com:8080")
            https_url: HTTPS proxy URL (e.g., "https://user:pass@proxy.example.com:8080")
            **kwargs: Additional arguments passed to GenericProxyConfig
            
        Returns:
            TldwatchProxyConfig instance with generic configuration
            
        Raises:
            ProxyConfigError: If GenericProxyConfig is not available
        """
        if GenericProxyConfig is None:
            raise ProxyConfigError(
                "GenericProxyConfig is not available. "
                "Please update youtube-transcript-api to a version that supports proxies."
            )
        
        if not http_url and not https_url:
            raise ProxyConfigError(
                "At least one of http_url or https_url must be provided for generic proxy configuration"
            )
        
        try:
            generic_config = GenericProxyConfig(
                http_url=http_url,
                https_url=https_url,
                **kwargs
            )
            logger.info("Created generic proxy configuration")
            return cls(generic_config)
        except Exception as e:
            raise ProxyConfigError(f"Failed to create generic proxy configuration: {str(e)}")
    
    @classmethod
    def from_config_dict(cls, config: dict) -> Optional["TldwatchProxyConfig"]:
        """
        Create proxy configuration from a configuration dictionary.
        
        Args:
            config: Dictionary containing proxy configuration
            
        Returns:
            TldwatchProxyConfig instance or None if no proxy configuration
            
        Example config formats:
            # Webshare configuration
            {
                "type": "webshare",
                "proxy_username": "your_username",
                "proxy_password": "your_password"
            }
            
            # Generic configuration
            {
                "type": "generic",
                "http_url": "http://user:pass@proxy.example.com:8080",
                "https_url": "https://user:pass@proxy.example.com:8080"
            }
        """
        if not config or not isinstance(config, dict):
            return None
        
        proxy_type = config.get("type", "").lower()
        
        if proxy_type == "webshare":
            return cls.create_webshare_config(
                proxy_username=config.get("proxy_username", ""),
                proxy_password=config.get("proxy_password", "")
            )
        elif proxy_type == "generic":
            return cls.create_generic_config(
                http_url=config.get("http_url"),
                https_url=config.get("https_url")
            )
        else:
            logger.warning(f"Unknown proxy type: {proxy_type}")
            return None
    
    def __str__(self) -> str:
        """String representation of proxy configuration"""
        if self._proxy_config is None:
            return "No proxy configuration"
        
        config_type = type(self._proxy_config).__name__
        return f"Proxy configuration: {config_type}"
    
    def __repr__(self) -> str:
        """Detailed representation of proxy configuration"""
        return f"TldwatchProxyConfig({self._proxy_config})"


def create_webshare_proxy(proxy_username: str, proxy_password: str) -> TldwatchProxyConfig:
    """
    Convenience function to create a Webshare proxy configuration.
    
    Args:
        proxy_username: Webshare proxy username
        proxy_password: Webshare proxy password
        
    Returns:
        TldwatchProxyConfig instance with Webshare configuration
    """
    return TldwatchProxyConfig.create_webshare_config(
        proxy_username=proxy_username,
        proxy_password=proxy_password
    )


def create_generic_proxy(
    http_url: Optional[str] = None,
    https_url: Optional[str] = None
) -> TldwatchProxyConfig:
    """
    Convenience function to create a generic proxy configuration.
    
    Args:
        http_url: HTTP proxy URL
        https_url: HTTPS proxy URL
        
    Returns:
        TldwatchProxyConfig instance with generic configuration
    """
    return TldwatchProxyConfig.create_generic_config(
        http_url=http_url,
        https_url=https_url
    )
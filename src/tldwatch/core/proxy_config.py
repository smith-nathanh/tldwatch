"""
Proxy configuration module for tldwatch.

This module provides proxy configuration classes to work around IP blocking
when fetching YouTube transcripts. It supports Webshare rotating residential
proxies and generic HTTP/HTTPS/SOCKS proxies.
"""

import logging
from typing import Optional, Union

try:
    from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig
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

    def __init__(
        self,
        proxy_config: Optional[Union[object, dict]] = None,
        http_url: Optional[str] = None,
        https_url: Optional[str] = None,
    ):
        """
        Initialize proxy configuration.

        Args:
            proxy_config: Either a youtube-transcript-api proxy config object
                         or a dictionary with proxy configuration
            http_url: HTTP proxy URL (for backwards compatibility with tests)
            https_url: HTTPS proxy URL (for backwards compatibility with tests)
        """
        # If direct URLs are provided (backward compatibility mode)
        if http_url is not None or https_url is not None:
            self.http_url = http_url
            self.https_url = https_url
            self._proxy_config = None
        else:
            self._proxy_config = proxy_config
            self.http_url = None
            self.https_url = None

    def get_proxy_dict(self) -> dict:
        """
        Get proxy dictionary for aiohttp.

        Returns:
            Dictionary with proxy configuration for aiohttp
        """
        proxy_dict = {}
        if self.http_url:
            proxy_dict["http"] = self.http_url
        if self.https_url:
            proxy_dict["https"] = self.https_url
        return proxy_dict

    @property
    def proxy_config(self):
        """Get the underlying proxy configuration object"""
        return self._proxy_config

    @classmethod
    def create_webshare_config(
        cls,
        proxy_username: str,
        proxy_password: str,
        endpoint: Optional[str] = None,
        **kwargs,
    ) -> "TldwatchProxyConfig":
        """
        Create a Webshare proxy configuration.

        Args:
            proxy_username: Webshare proxy username
            proxy_password: Webshare proxy password
            endpoint: Custom endpoint (defaults to rotating-residential.webshare.io:9000)
            **kwargs: Additional arguments passed to WebshareProxyConfig

        Returns:
            TldwatchProxyConfig instance with Webshare configuration

        Raises:
            ProxyConfigError: If WebshareProxyConfig is not available
        """
        if not proxy_username or not proxy_password:
            raise ProxyConfigError(
                "Username and password are required for Webshare configuration"
            )

        if WebshareProxyConfig is None:
            # Create a compatible proxy configuration for testing/backwards compatibility
            webshare_endpoint = endpoint or "rotating-residential.webshare.io:9000"
            http_url = f"http://{proxy_username}:{proxy_password}@{webshare_endpoint}"
            https_url = f"https://{proxy_username}:{proxy_password}@{webshare_endpoint}"
            logger.info("Created Webshare proxy configuration (compatibility mode)")
            return cls(http_url=http_url, https_url=https_url)

        # For compatibility with tests, always expose URLs directly
        webshare_endpoint = endpoint or "rotating-residential.webshare.io:9000"
        http_url = f"http://{proxy_username}:{proxy_password}@{webshare_endpoint}"
        https_url = f"https://{proxy_username}:{proxy_password}@{webshare_endpoint}"

        try:
            webshare_config = WebshareProxyConfig(
                proxy_username=proxy_username, proxy_password=proxy_password, **kwargs
            )
            logger.info("Created Webshare proxy configuration")
            instance = cls(webshare_config)
            # Set attributes for test compatibility
            instance.http_url = http_url
            instance.https_url = https_url
            return instance
        except Exception as e:
            raise ProxyConfigError(
                f"Failed to create Webshare proxy configuration: {str(e)}"
            )

    @classmethod
    def create_generic_config(
        cls, http_url: Optional[str] = None, https_url: Optional[str] = None, **kwargs
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
        if not http_url and not https_url:
            raise ProxyConfigError("At least one proxy URL must be provided")

        # Basic URL validation
        from urllib.parse import urlparse

        for url, scheme in [(http_url, "http"), (https_url, "https")]:
            if url:
                try:
                    parsed = urlparse(url)
                    if not parsed.scheme:
                        raise ProxyConfigError(f"Invalid proxy URL: {url}")
                    if parsed.scheme not in ["http", "https"]:
                        raise ProxyConfigError(
                            f"Unsupported proxy scheme: {parsed.scheme}"
                        )
                    if not parsed.netloc:
                        raise ProxyConfigError(f"Invalid proxy URL: {url}")
                except ValueError:
                    raise ProxyConfigError(f"Invalid proxy URL: {url}")

        if GenericProxyConfig is None:
            # Create a compatible proxy configuration for testing/backwards compatibility
            logger.info("Created generic proxy configuration (compatibility mode)")
            return cls(http_url=http_url, https_url=https_url)

        try:
            generic_config = GenericProxyConfig(
                http_url=http_url, https_url=https_url, **kwargs
            )
            logger.info("Created generic proxy configuration")
            instance = cls(generic_config)
            # Set attributes for test compatibility
            instance.http_url = http_url
            instance.https_url = https_url
            return instance
        except Exception as e:
            raise ProxyConfigError(
                f"Failed to create generic proxy configuration: {str(e)}"
            )

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
                proxy_password=config.get("proxy_password", ""),
            )
        elif proxy_type == "generic":
            return cls.create_generic_config(
                http_url=config.get("http_url"), https_url=config.get("https_url")
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


def create_webshare_proxy(
    proxy_username: str, proxy_password: str, endpoint: Optional[str] = None
) -> TldwatchProxyConfig:
    """
    Convenience function to create a Webshare proxy configuration.

    Args:
        proxy_username: Webshare proxy username
        proxy_password: Webshare proxy password
        endpoint: Custom endpoint (defaults to rotating-residential.webshare.io:9000)

    Returns:
        TldwatchProxyConfig instance with Webshare configuration
    """
    return TldwatchProxyConfig.create_webshare_config(
        proxy_username=proxy_username, proxy_password=proxy_password, endpoint=endpoint
    )


def create_generic_proxy(
    http_url: Optional[str] = None, https_url: Optional[str] = None
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
        http_url=http_url, https_url=https_url
    )

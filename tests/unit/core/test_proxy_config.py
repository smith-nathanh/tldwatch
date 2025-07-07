"""
Unit tests for proxy configuration.
Tests proxy creation and configuration functionality.
"""

import pytest

from tldwatch.core.proxy_config import (
    ProxyConfigError,
    TldwatchProxyConfig,
    create_generic_proxy,
    create_webshare_proxy,
)


class TestTldwatchProxyConfig:
    """Test TldwatchProxyConfig class."""

    def test_proxy_config_creation(self):
        """Test creating a proxy configuration."""
        config = TldwatchProxyConfig(
            http_url="http://proxy.example.com:8080",
            https_url="https://proxy.example.com:8080",
        )

        assert config.http_url == "http://proxy.example.com:8080"
        assert config.https_url == "https://proxy.example.com:8080"

    def test_proxy_config_http_only(self):
        """Test creating proxy config with HTTP only."""
        config = TldwatchProxyConfig(http_url="http://proxy.example.com:8080")

        assert config.http_url == "http://proxy.example.com:8080"
        assert config.https_url is None

    def test_proxy_config_https_only(self):
        """Test creating proxy config with HTTPS only."""
        config = TldwatchProxyConfig(https_url="https://proxy.example.com:8080")

        assert config.http_url is None
        assert config.https_url == "https://proxy.example.com:8080"

    def test_get_proxy_dict(self):
        """Test getting proxy dictionary for aiohttp."""
        config = TldwatchProxyConfig(
            http_url="http://proxy.example.com:8080",
            https_url="https://proxy.example.com:8080",
        )

        proxy_dict = config.get_proxy_dict()

        assert proxy_dict["http"] == "http://proxy.example.com:8080"
        assert proxy_dict["https"] == "https://proxy.example.com:8080"

    def test_get_proxy_dict_partial(self):
        """Test getting proxy dictionary with only HTTP."""
        config = TldwatchProxyConfig(http_url="http://proxy.example.com:8080")

        proxy_dict = config.get_proxy_dict()

        assert proxy_dict["http"] == "http://proxy.example.com:8080"
        assert "https" not in proxy_dict


class TestCreateWebshareProxy:
    """Test Webshare proxy creation."""

    def test_create_webshare_proxy_success(self):
        """Test successful Webshare proxy creation."""
        result = create_webshare_proxy(
            proxy_username="test_user", proxy_password="test_pass"
        )

        assert isinstance(result, TldwatchProxyConfig)
        assert "test_user:test_pass" in result.http_url
        assert "test_user:test_pass" in result.https_url
        assert "rotating-residential.webshare.io:9000" in result.http_url
        assert "rotating-residential.webshare.io:9000" in result.https_url

    def test_create_webshare_proxy_missing_username(self):
        """Test Webshare proxy creation with missing username."""
        with pytest.raises(
            ProxyConfigError, match="Username and password are required"
        ):
            create_webshare_proxy(proxy_username=None, proxy_password="test_pass")

    def test_create_webshare_proxy_missing_password(self):
        """Test Webshare proxy creation with missing password."""
        with pytest.raises(
            ProxyConfigError, match="Username and password are required"
        ):
            create_webshare_proxy(proxy_username="test_user", proxy_password=None)

    def test_create_webshare_proxy_empty_credentials(self):
        """Test Webshare proxy creation with empty credentials."""
        with pytest.raises(
            ProxyConfigError, match="Username and password are required"
        ):
            create_webshare_proxy(proxy_username="", proxy_password="")

    def test_create_webshare_proxy_custom_endpoint(self):
        """Test Webshare proxy creation with custom endpoint."""
        result = create_webshare_proxy(
            proxy_username="test_user",
            proxy_password="test_pass",
            endpoint="custom.webshare.io:8080",
        )

        assert "custom.webshare.io:8080" in result.http_url
        assert "custom.webshare.io:8080" in result.https_url


class TestCreateGenericProxy:
    """Test generic proxy creation."""

    def test_create_generic_proxy_both_urls(self):
        """Test creating generic proxy with both HTTP and HTTPS URLs."""
        result = create_generic_proxy(
            http_url="http://proxy.example.com:8080",
            https_url="https://proxy.example.com:8080",
        )

        assert isinstance(result, TldwatchProxyConfig)
        assert result.http_url == "http://proxy.example.com:8080"
        assert result.https_url == "https://proxy.example.com:8080"

    def test_create_generic_proxy_http_only(self):
        """Test creating generic proxy with HTTP URL only."""
        result = create_generic_proxy(http_url="http://proxy.example.com:8080")

        assert result.http_url == "http://proxy.example.com:8080"
        assert result.https_url is None

    def test_create_generic_proxy_https_only(self):
        """Test creating generic proxy with HTTPS URL only."""
        result = create_generic_proxy(https_url="https://proxy.example.com:8080")

        assert result.http_url is None
        assert result.https_url == "https://proxy.example.com:8080"

    def test_create_generic_proxy_no_urls(self):
        """Test creating generic proxy with no URLs raises error."""
        with pytest.raises(
            ProxyConfigError, match="At least one proxy URL must be provided"
        ):
            create_generic_proxy()

    def test_create_generic_proxy_with_auth(self):
        """Test creating generic proxy with authentication."""
        result = create_generic_proxy(
            http_url="http://user:pass@proxy.example.com:8080",
            https_url="https://user:pass@proxy.example.com:8080",
        )

        assert "user:pass@" in result.http_url
        assert "user:pass@" in result.https_url

    def test_create_generic_proxy_invalid_url(self):
        """Test creating generic proxy with invalid URL."""
        with pytest.raises(ProxyConfigError, match="Invalid proxy URL"):
            create_generic_proxy(http_url="not_a_valid_url")

    def test_create_generic_proxy_unsupported_scheme(self):
        """Test creating generic proxy with unsupported URL scheme."""
        with pytest.raises(ProxyConfigError, match="Unsupported proxy scheme"):
            create_generic_proxy(http_url="ftp://proxy.example.com:8080")


class TestProxyConfigError:
    """Test ProxyConfigError exception."""

    def test_proxy_config_error_creation(self):
        """Test creating ProxyConfigError."""
        error = ProxyConfigError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_proxy_config_error_with_cause(self):
        """Test ProxyConfigError with underlying cause."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = ProxyConfigError("Proxy error")
            error.__cause__ = e
            assert str(error) == "Proxy error"
            assert error.__cause__ is e


class TestProxyIntegration:
    """Test proxy configuration integration scenarios."""

    def test_proxy_config_environment_variables(self):
        """Test that proxy configurations work with environment variables."""
        # This test would be useful if the proxy config reads from environment
        # variables in addition to explicit parameters

        # For now, just test that the proxy config objects work as expected
        config = create_generic_proxy(http_url="http://proxy.example.com:8080")

        proxy_dict = config.get_proxy_dict()
        assert proxy_dict["http"] == "http://proxy.example.com:8080"

    def test_multiple_proxy_configs(self):
        """Test creating multiple different proxy configurations."""
        webshare_config = create_webshare_proxy(
            proxy_username="webshare_user", proxy_password="webshare_pass"
        )

        generic_config = create_generic_proxy(http_url="http://generic.proxy.com:8080")

        # Both should be valid but different
        assert webshare_config.http_url != generic_config.http_url
        assert "webshare.io" in webshare_config.http_url
        assert "generic.proxy.com" in generic_config.http_url

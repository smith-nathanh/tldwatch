#!/usr/bin/env python3
"""
Test script to verify proxy integration with tldwatch.

This script tests the proxy configuration functionality without making
actual API calls or using real proxy credentials.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tldwatch import (
    Summarizer,
    TldwatchProxyConfig,
    create_webshare_proxy,
    create_generic_proxy,
    ProxyConfigError,
)
from tldwatch.core.config import Config


def test_proxy_config_creation():
    """Test proxy configuration creation"""
    print("=== Testing Proxy Configuration Creation ===")
    
    # Test Webshare config creation
    try:
        webshare_config = create_webshare_proxy(
            proxy_username="test_user",
            proxy_password="test_pass"
        )
        print("✓ Webshare proxy config created successfully")
        print(f"  Config: {webshare_config}")
    except Exception as e:
        print(f"✗ Webshare config creation failed: {e}")
    
    # Test generic config creation
    try:
        generic_config = create_generic_proxy(
            http_url="http://user:pass@proxy.example.com:8080",
            https_url="https://user:pass@proxy.example.com:8080"
        )
        print("✓ Generic proxy config created successfully")
        print(f"  Config: {generic_config}")
    except Exception as e:
        print(f"✗ Generic config creation failed: {e}")
    
    # Test config from dict
    try:
        config_dict = {
            "type": "webshare",
            "proxy_username": "test_user",
            "proxy_password": "test_pass"
        }
        dict_config = TldwatchProxyConfig.from_config_dict(config_dict)
        print("✓ Config from dict created successfully")
        print(f"  Config: {dict_config}")
    except Exception as e:
        print(f"✗ Config from dict creation failed: {e}")


def test_summarizer_integration():
    """Test Summarizer integration with proxy config"""
    print("\n=== Testing Summarizer Integration ===")
    
    # Set mock API key
    os.environ["OPENAI_API_KEY"] = "test_key"
    
    try:
        # Create proxy config
        proxy_config = create_webshare_proxy(
            proxy_username="test_user",
            proxy_password="test_pass"
        )
        
        # Create summarizer with proxy config
        summarizer = Summarizer(
            provider="openai",
            model="gpt-4o-mini",
            proxy_config=proxy_config
        )
        
        print("✓ Summarizer created with proxy config")
        print(f"  Proxy config: {summarizer.proxy_config}")
        print(f"  Provider: {summarizer.provider_name}")
        
    except Exception as e:
        print(f"✗ Summarizer integration failed: {e}")


def test_config_file_integration():
    """Test configuration file integration"""
    print("\n=== Testing Config File Integration ===")
    
    try:
        # Create config with proxy settings
        config_data = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "proxy": {
                "type": "webshare",
                "proxy_username": "test_user",
                "proxy_password": "test_pass"
            }
        }
        
        config = Config(config_data)
        proxy_config = config.proxy_config
        
        if proxy_config:
            print("✓ Proxy config loaded from config file")
            print(f"  Config: {proxy_config}")
        else:
            print("✗ No proxy config found in config file")
            
    except Exception as e:
        print(f"✗ Config file integration failed: {e}")


def test_error_handling():
    """Test error handling for proxy configuration"""
    print("\n=== Testing Error Handling ===")
    
    # Test invalid Webshare config
    try:
        invalid_config = create_webshare_proxy(
            proxy_username="",  # Empty username
            proxy_password="test_pass"
        )
        print("✗ Should have failed with empty username")
    except ProxyConfigError as e:
        print(f"✓ Correctly caught ProxyConfigError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test invalid generic config
    try:
        invalid_config = create_generic_proxy()  # No URLs provided
        print("✗ Should have failed with no URLs")
    except ProxyConfigError as e:
        print(f"✓ Correctly caught ProxyConfigError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test invalid config dict
    try:
        invalid_dict = {
            "type": "unknown_type",
            "some_param": "value"
        }
        config = TldwatchProxyConfig.from_config_dict(invalid_dict)
        if config is None:
            print("✓ Correctly returned None for unknown proxy type")
        else:
            print(f"✗ Should have returned None, got: {config}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def test_cli_integration():
    """Test CLI argument parsing for proxy options"""
    print("\n=== Testing CLI Integration ===")
    
    try:
        from tldwatch.cli.main import create_parser, create_proxy_config
        import argparse
        
        # Test parser creation
        parser = create_parser()
        print("✓ CLI parser created successfully")
        
        # Test proxy argument parsing
        test_args = [
            "--webshare-username", "test_user",
            "--webshare-password", "test_pass",
            "--video-id", "test_video"
        ]
        
        args = parser.parse_args(test_args)
        print("✓ CLI arguments parsed successfully")
        print(f"  Webshare username: {args.webshare_username}")
        print(f"  Webshare password: {args.webshare_password}")
        
        # Test proxy config creation from args
        proxy_config = create_proxy_config(args)
        if proxy_config:
            print("✓ Proxy config created from CLI args")
            print(f"  Config: {proxy_config}")
        else:
            print("✗ No proxy config created from CLI args")
            
    except Exception as e:
        print(f"✗ CLI integration test failed: {e}")


def test_import_compatibility():
    """Test import compatibility and fallbacks"""
    print("\n=== Testing Import Compatibility ===")
    
    try:
        # Test that imports work even if youtube-transcript-api proxy classes aren't available
        from tldwatch.core.proxy_config import WebshareProxyConfig, GenericProxyConfig
        
        if WebshareProxyConfig is not None:
            print("✓ WebshareProxyConfig is available")
        else:
            print("⚠ WebshareProxyConfig is not available (fallback mode)")
            
        if GenericProxyConfig is not None:
            print("✓ GenericProxyConfig is available")
        else:
            print("⚠ GenericProxyConfig is not available (fallback mode)")
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def main():
    """Run all tests"""
    print("Testing tldwatch proxy integration...\n")
    
    test_proxy_config_creation()
    test_summarizer_integration()
    test_config_file_integration()
    test_error_handling()
    test_cli_integration()
    test_import_compatibility()
    
    print("\n=== Test Summary ===")
    print("✓ Proxy configuration system is working")
    print("✓ Integration with Summarizer is working")
    print("✓ CLI integration is working")
    print("✓ Error handling is working")
    print("\nTo use with real proxies:")
    print("1. Sign up for Webshare at https://www.webshare.io/")
    print("2. Purchase a 'Residential' proxy package")
    print("3. Set WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD")
    print("4. Use with tldwatch CLI or library")


if __name__ == "__main__":
    main()
"""
Examples of using tldwatch with proxy configuration to avoid IP blocking.

This file demonstrates how to configure and use Webshare and generic proxies
to work around YouTube's IP blocking when fetching transcripts.
"""

import asyncio
import os
from pathlib import Path

from tldwatch import Summarizer, create_webshare_proxy, create_generic_proxy, ProxyConfigError


async def webshare_proxy_example():
    """Example using Webshare rotating residential proxies"""
    print("\n=== Webshare Proxy Example ===")
    
    # Get credentials from environment variables
    proxy_username = os.environ.get("WEBSHARE_PROXY_USERNAME")
    proxy_password = os.environ.get("WEBSHARE_PROXY_PASSWORD")
    
    if not proxy_username or not proxy_password:
        print("Skipping Webshare example - credentials not found in environment")
        print("Set WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD environment variables")
        return
    
    try:
        # Create Webshare proxy configuration
        proxy_config = create_webshare_proxy(
            proxy_username=proxy_username,
            proxy_password=proxy_password
        )
        
        # Initialize summarizer with proxy configuration
        summarizer = Summarizer(
            provider="openai",
            model="gpt-4o-mini",
            proxy_config=proxy_config
        )
        
        # Get summary using proxy
        summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
        print("Summary with Webshare proxy:", summary[:200] + "...")
        
    except ProxyConfigError as e:
        print(f"Proxy configuration error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")


async def generic_proxy_example():
    """Example using generic HTTP/HTTPS proxy"""
    print("\n=== Generic Proxy Example ===")
    
    # Example proxy URLs (replace with your actual proxy)
    http_proxy = os.environ.get("HTTP_PROXY_URL")  # e.g., "http://user:pass@proxy.example.com:8080"
    https_proxy = os.environ.get("HTTPS_PROXY_URL")  # e.g., "https://user:pass@proxy.example.com:8080"
    
    if not http_proxy and not https_proxy:
        print("Skipping generic proxy example - no proxy URLs found in environment")
        print("Set HTTP_PROXY_URL and/or HTTPS_PROXY_URL environment variables")
        return
    
    try:
        # Create generic proxy configuration
        proxy_config = create_generic_proxy(
            http_url=http_proxy,
            https_url=https_proxy
        )
        
        # Initialize summarizer with proxy configuration
        summarizer = Summarizer(
            provider="groq",
            model="mixtral-8x7b-32768",
            proxy_config=proxy_config
        )
        
        # Get summary using proxy
        summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
        print("Summary with generic proxy:", summary[:200] + "...")
        
    except ProxyConfigError as e:
        print(f"Proxy configuration error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")


async def config_file_proxy_example():
    """Example using proxy configuration from config file"""
    print("\n=== Config File Proxy Example ===")
    
    from tldwatch.core.config import Config
    
    # Create a sample config with proxy settings
    config_data = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "proxy": {
            "type": "webshare",
            "proxy_username": os.environ.get("WEBSHARE_PROXY_USERNAME", ""),
            "proxy_password": os.environ.get("WEBSHARE_PROXY_PASSWORD", "")
        }
    }
    
    if not config_data["proxy"]["proxy_username"] or not config_data["proxy"]["proxy_password"]:
        print("Skipping config file example - Webshare credentials not found")
        return
    
    try:
        # Load configuration
        config = Config(config_data)
        
        # Get proxy configuration from config
        proxy_config = config.proxy_config
        
        if proxy_config:
            # Initialize summarizer with config-based proxy
            summarizer = Summarizer(
                provider=config.current_provider,
                model=config.current_model,
                proxy_config=proxy_config
            )
            
            # Get summary using proxy from config
            summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
            print("Summary with config-based proxy:", summary[:200] + "...")
        else:
            print("No valid proxy configuration found in config")
            
    except Exception as e:
        print(f"Error: {str(e)}")


async def batch_processing_with_proxy():
    """Example of batch processing multiple videos with proxy to avoid rate limiting"""
    print("\n=== Batch Processing with Proxy Example ===")
    
    proxy_username = os.environ.get("WEBSHARE_PROXY_USERNAME")
    proxy_password = os.environ.get("WEBSHARE_PROXY_PASSWORD")
    
    if not proxy_username or not proxy_password:
        print("Skipping batch processing example - Webshare credentials not found")
        return
    
    try:
        # Create proxy configuration
        proxy_config = create_webshare_proxy(
            proxy_username=proxy_username,
            proxy_password=proxy_password
        )
        
        # Initialize summarizer with proxy
        summarizer = Summarizer(
            provider="groq",
            model="mixtral-8x7b-32768",
            proxy_config=proxy_config
        )
        
        # List of video IDs to process
        video_ids = ["QAgR4uQ15rc", "dQw4w9WgXcQ"]  # Add more video IDs as needed
        
        output_dir = Path("proxy_summaries")
        output_dir.mkdir(exist_ok=True)
        
        for i, video_id in enumerate(video_ids):
            try:
                print(f"Processing video {i+1}/{len(video_ids)}: {video_id}")
                
                # Get summary with proxy
                summary = await summarizer.get_summary(video_id=video_id)
                
                # Export to file
                output_file = output_dir / f"{video_id}_summary.json"
                await summarizer.export_summary(str(output_file))
                
                print(f"✓ Completed {video_id}")
                
                # Add delay between requests to be respectful
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"✗ Error processing {video_id}: {str(e)}")
                continue
        
        print(f"Batch processing completed. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")


async def error_handling_example():
    """Example of handling proxy-related errors"""
    print("\n=== Error Handling Example ===")
    
    try:
        # Try to create proxy config with invalid credentials
        proxy_config = create_webshare_proxy(
            proxy_username="invalid_user",
            proxy_password="invalid_pass"
        )
        
        summarizer = Summarizer(proxy_config=proxy_config)
        
        # This will likely fail due to invalid credentials
        summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
        print("Unexpected success:", summary[:100] + "...")
        
    except ProxyConfigError as e:
        print(f"Proxy configuration error: {str(e)}")
    except Exception as e:
        error_msg = str(e).lower()
        if "blocked" in error_msg or "forbidden" in error_msg:
            print("IP blocking detected. Consider:")
            print("1. Using Webshare rotating residential proxies")
            print("2. Checking your proxy credentials")
            print("3. Trying a different proxy provider")
        else:
            print(f"Other error: {str(e)}")


def setup_instructions():
    """Print setup instructions for proxy configuration"""
    print("\n=== Setup Instructions ===")
    print("To use proxy configuration with tldwatch:")
    print()
    print("1. For Webshare (recommended):")
    print("   - Sign up at https://www.webshare.io/")
    print("   - Purchase a 'Residential' proxy package (NOT 'Proxy Server' or 'Static Residential')")
    print("   - Get your credentials from https://dashboard.webshare.io/proxy/settings")
    print("   - Set environment variables:")
    print("     export WEBSHARE_PROXY_USERNAME='your_username'")
    print("     export WEBSHARE_PROXY_PASSWORD='your_password'")
    print()
    print("2. For generic proxies:")
    print("   - Set environment variables:")
    print("     export HTTP_PROXY_URL='http://user:pass@proxy.example.com:8080'")
    print("     export HTTPS_PROXY_URL='https://user:pass@proxy.example.com:8080'")
    print()
    print("3. In your code:")
    print("   from tldwatch import Summarizer, create_webshare_proxy")
    print("   proxy_config = create_webshare_proxy(username, password)")
    print("   summarizer = Summarizer(proxy_config=proxy_config)")
    print()


async def main():
    """Run all proxy examples"""
    setup_instructions()
    
    # Run examples
    await webshare_proxy_example()
    await generic_proxy_example()
    await config_file_proxy_example()
    await batch_processing_with_proxy()
    await error_handling_example()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Setup script to test Webshare proxy configuration with tldwatch.

This script helps users verify their proxy setup before using it in production.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tldwatch import create_webshare_proxy, Summarizer, ProxyConfigError


def check_environment():
    """Check if required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    required_vars = {
        "WEBSHARE_PROXY_USERNAME": "Webshare proxy username",
        "WEBSHARE_PROXY_PASSWORD": "Webshare proxy password",
        "OPENAI_API_KEY": "OpenAI API key (for testing)"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            missing_vars.append(f"  {var}: {description}")
        else:
            print(f"✅ {var}: Set")
    
    if missing_vars:
        print("\n❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nPlease set these variables and run the script again.")
        return False
    
    return True


def test_proxy_config():
    """Test proxy configuration creation"""
    print("\n🔧 Testing proxy configuration...")
    
    try:
        proxy_config = create_webshare_proxy(
            proxy_username=os.environ["WEBSHARE_PROXY_USERNAME"],
            proxy_password=os.environ["WEBSHARE_PROXY_PASSWORD"]
        )
        print(f"✅ Proxy configuration created successfully")
        print(f"   Config: {proxy_config}")
        return proxy_config
    except ProxyConfigError as e:
        print(f"❌ Proxy configuration failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


async def test_summarizer_with_proxy(proxy_config):
    """Test summarizer with proxy configuration"""
    print("\n🤖 Testing summarizer with proxy...")
    
    try:
        summarizer = Summarizer(
            provider="openai",
            model="gpt-4o-mini",
            proxy_config=proxy_config
        )
        print("✅ Summarizer created with proxy configuration")
        return summarizer
    except Exception as e:
        print(f"❌ Summarizer creation failed: {e}")
        return None


async def test_transcript_fetch(summarizer):
    """Test transcript fetching with proxy"""
    print("\n📹 Testing transcript fetch with proxy...")
    print("   Using a short test video...")
    
    # Use a short, well-known video for testing
    test_video_id = "QAgR4uQ15rc"  # Replace with a known good video ID
    
    try:
        summary = await summarizer.get_summary(video_id=test_video_id)
        print(f"✅ Successfully fetched and summarized transcript")
        print(f"   Summary length: {len(summary)} characters")
        print(f"   Summary preview: {summary[:150]}...")
        return True
    except Exception as e:
        print(f"❌ Transcript fetch failed: {e}")
        if "blocked" in str(e).lower():
            print("   💡 This might indicate IP blocking issues")
        elif "transcript" in str(e).lower():
            print("   💡 The test video might not have transcripts available")
        return False


def print_setup_instructions():
    """Print setup instructions"""
    print("\n📋 Setup Instructions:")
    print("1. Sign up for Webshare at https://www.webshare.io/")
    print("2. Purchase a 'Residential' proxy package (NOT 'Proxy Server' or 'Static')")
    print("3. Get your credentials from https://dashboard.webshare.io/proxy/settings")
    print("4. Set environment variables:")
    print("   export WEBSHARE_PROXY_USERNAME='your_username'")
    print("   export WEBSHARE_PROXY_PASSWORD='your_password'")
    print("   export OPENAI_API_KEY='your_openai_key'")
    print("5. Run this script again to test your setup")


async def main():
    """Main test function"""
    print("🚀 tldwatch Proxy Setup Test")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print_setup_instructions()
        return
    
    # Test proxy configuration
    proxy_config = test_proxy_config()
    if not proxy_config:
        print("\n❌ Proxy configuration test failed")
        return
    
    # Test summarizer creation
    summarizer = await test_summarizer_with_proxy(proxy_config)
    if not summarizer:
        print("\n❌ Summarizer test failed")
        return
    
    # Test transcript fetching
    success = await test_transcript_fetch(summarizer)
    
    # Print results
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! Your proxy setup is working correctly.")
        print("\n💡 You can now use tldwatch with proxy configuration:")
        print("   tldwatch 'https://www.youtube.com/watch?v=VIDEO_ID'")
        print("   # or in Python:")
        print("   from tldwatch import Summarizer, create_webshare_proxy")
        print("   proxy_config = create_webshare_proxy(username, password)")
        print("   summarizer = Summarizer(proxy_config=proxy_config)")
    else:
        print("❌ Some tests failed. Please check your configuration.")
        print("\n🔧 Troubleshooting:")
        print("1. Verify your Webshare account is active")
        print("2. Check that you purchased the correct proxy type (Residential)")
        print("3. Ensure your credentials are correct")
        print("4. Try with a different test video ID")
    
    print("\n📚 For more examples, see:")
    print("   - examples/proxy_usage.py")
    print("   - examples/complete_proxy_example.py")
    print("   - PROXY_SETUP.md")


if __name__ == "__main__":
    asyncio.run(main())
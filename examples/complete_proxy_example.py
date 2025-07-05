#!/usr/bin/env python3
"""
Complete example showing how to use tldwatch with Webshare proxy configuration
to avoid IP blocking when processing YouTube transcripts.

This example demonstrates:
1. Setting up Webshare proxy configuration
2. Using the proxy with the Summarizer
3. Batch processing multiple videos
4. Error handling and best practices
"""

import asyncio
import os
import logging
from pathlib import Path

from tldwatch import (
    Summarizer,
    create_webshare_proxy,
    ProxyConfigError,
    SummarizerError,
)

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function"""
    print("=== tldwatch with Webshare Proxy Example ===\n")
    
    # Check for required credentials
    proxy_username = os.environ.get("WEBSHARE_PROXY_USERNAME")
    proxy_password = os.environ.get("WEBSHARE_PROXY_PASSWORD")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not proxy_username or not proxy_password:
        print("❌ Missing Webshare credentials!")
        print("Please set the following environment variables:")
        print("  export WEBSHARE_PROXY_USERNAME='your_username'")
        print("  export WEBSHARE_PROXY_PASSWORD='your_password'")
        print("\nTo get these credentials:")
        print("1. Sign up at https://www.webshare.io/")
        print("2. Purchase a 'Residential' proxy package")
        print("3. Get credentials from https://dashboard.webshare.io/proxy/settings")
        return
    
    if not openai_api_key:
        print("❌ Missing OpenAI API key!")
        print("Please set: export OPENAI_API_KEY='your_api_key'")
        return
    
    try:
        # Step 1: Create proxy configuration
        print("🔧 Setting up Webshare proxy configuration...")
        proxy_config = create_webshare_proxy(
            proxy_username=proxy_username,
            proxy_password=proxy_password
        )
        print(f"✅ Proxy configuration created: {proxy_config}")
        
        # Step 2: Initialize summarizer with proxy
        print("\n🤖 Initializing summarizer with proxy...")
        summarizer = Summarizer(
            provider="openai",
            model="gpt-4o-mini",
            proxy_config=proxy_config,
            temperature=0.7
        )
        print("✅ Summarizer initialized with proxy configuration")
        
        # Step 3: Test with a single video
        print("\n📹 Testing with a single video...")
        test_video_id = "QAgR4uQ15rc"  # Replace with a real video ID
        
        try:
            summary = await summarizer.get_summary(video_id=test_video_id)
            print(f"✅ Successfully generated summary for {test_video_id}")
            print(f"📝 Summary preview: {summary[:200]}...")
            
            # Export the summary
            output_file = "proxy_test_summary.json"
            await summarizer.export_summary(output_file)
            print(f"💾 Summary exported to {output_file}")
            
        except SummarizerError as e:
            print(f"❌ Summarization error: {e}")
            if "blocked" in str(e).lower():
                print("💡 This might be an IP blocking issue. The proxy should help with this.")
            return
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return
        
        # Step 4: Batch processing example
        print("\n📚 Batch processing example...")
        video_ids = [
            "QAgR4uQ15rc",  # Replace with real video IDs
            "dQw4w9WgXcQ",  # This is Rick Roll - might not have transcripts
            # Add more video IDs as needed
        ]
        
        output_dir = Path("proxy_batch_summaries")
        output_dir.mkdir(exist_ok=True)
        
        successful = 0
        failed = 0
        
        for i, video_id in enumerate(video_ids):
            print(f"\n🔄 Processing video {i+1}/{len(video_ids)}: {video_id}")
            
            try:
                # Create a new summarizer instance for each video to avoid state issues
                video_summarizer = Summarizer(
                    provider="openai",
                    model="gpt-4o-mini",
                    proxy_config=proxy_config
                )
                
                summary = await video_summarizer.get_summary(video_id=video_id)
                
                # Export to file
                output_file = output_dir / f"{video_id}_summary.json"
                await video_summarizer.export_summary(str(output_file))
                
                print(f"✅ Completed {video_id}")
                print(f"📝 Summary: {summary[:100]}...")
                successful += 1
                
                # Be respectful - add delay between requests
                print("⏳ Waiting 3 seconds before next request...")
                await asyncio.sleep(3)
                
            except SummarizerError as e:
                print(f"❌ Failed to process {video_id}: {e}")
                failed += 1
                
                if "transcript" in str(e).lower():
                    print("💡 This video might not have transcripts available")
                elif "blocked" in str(e).lower():
                    print("💡 IP blocking detected even with proxy - might need to wait or try different proxy")
                
                continue
            except Exception as e:
                print(f"❌ Unexpected error for {video_id}: {e}")
                failed += 1
                continue
        
        # Step 5: Summary of batch processing
        print(f"\n📊 Batch processing complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Results saved to: {output_dir}")
        
        # Step 6: Demonstrate configuration file usage
        print("\n⚙️ Configuration file example...")
        from tldwatch.core.config import Config
        
        config_data = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "proxy": {
                "type": "webshare",
                "proxy_username": proxy_username,
                "proxy_password": proxy_password
            }
        }
        
        config = Config(config_data)
        config_proxy = config.proxy_config
        
        if config_proxy:
            print(f"✅ Proxy configuration loaded from config: {config_proxy}")
            
            # Use config-based summarizer
            config_summarizer = Summarizer(
                provider=config.current_provider,
                model=config.current_model,
                proxy_config=config_proxy
            )
            print("✅ Summarizer created from configuration file")
        
        print("\n🎉 Example completed successfully!")
        print("\n💡 Tips for production use:")
        print("1. Monitor your Webshare usage to avoid exceeding limits")
        print("2. Add appropriate delays between requests")
        print("3. Implement retry logic for failed requests")
        print("4. Store proxy credentials securely")
        print("5. Consider using different proxy endpoints for different use cases")
        
    except ProxyConfigError as e:
        print(f"❌ Proxy configuration error: {e}")
        print("💡 Check your Webshare credentials and account status")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    asyncio.run(main())
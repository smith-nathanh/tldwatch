# Webshare Integration for tldwatch

This document provides a complete guide for integrating Webshare rotating residential proxies with tldwatch to avoid IP blocking when fetching YouTube transcripts.

## Overview

YouTube may block IP addresses that make too many requests, especially from cloud providers. This integration allows you to use [Webshare](https://www.webshare.io/) rotating residential proxies to distribute requests across multiple IP addresses, avoiding blocks.

## Quick Start

### 1. Setup Webshare Account

1. Sign up at [https://www.webshare.io/](https://www.webshare.io/)
2. Purchase a **"Residential"** proxy package (⚠️ NOT "Proxy Server" or "Static Residential")
3. Get your credentials from [Proxy Settings](https://dashboard.webshare.io/proxy/settings)

### 2. Set Environment Variables

```bash
export WEBSHARE_PROXY_USERNAME="your_username"
export WEBSHARE_PROXY_PASSWORD="your_password"
export OPENAI_API_KEY="your_openai_key"  # or other provider key
```

### 3. Test Your Setup

```bash
cd tldwatch
python tests/integration/proxy/test_proxy_setup.py
```

### 4. Use with tldwatch

#### CLI Usage
```bash
# Automatic (uses environment variables)
tldwatch "https://www.youtube.com/watch?v=QAgR4uQ15rc"

# Explicit credentials
tldwatch --webshare-username "user" --webshare-password "pass" "https://www.youtube.com/watch?v=QAgR4uQ15rc"
```

#### Library Usage
```python
import asyncio
from tldwatch import Summarizer, create_webshare_proxy

async def main():
    proxy_config = create_webshare_proxy(
        proxy_username="your_username",
        proxy_password="your_password"
    )
    
    summarizer = Summarizer(
        provider="openai",
        proxy_config=proxy_config
    )
    
    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
    print(summary)

asyncio.run(main())
```

## Implementation Details

### Proxy Configuration Classes

The integration provides several classes for proxy management:

- `TldwatchProxyConfig`: Main proxy configuration wrapper
- `create_webshare_proxy()`: Convenience function for Webshare setup
- `create_generic_proxy()`: For other proxy providers
- `ProxyConfigError`: Exception for proxy configuration issues

### Integration Points

1. **Summarizer Class**: Accepts `proxy_config` parameter
2. **CLI Interface**: Supports `--webshare-username` and `--webshare-password` options
3. **Configuration File**: Supports proxy settings in JSON config
4. **Error Handling**: Enhanced error messages for IP blocking scenarios

### Configuration File Format

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "proxy": {
    "type": "webshare",
    "proxy_username": "your_username",
    "proxy_password": "your_password"
  }
}
```

## Advanced Usage

### Batch Processing

```python
import asyncio
from tldwatch import Summarizer, create_webshare_proxy

async def batch_process():
    proxy_config = create_webshare_proxy(
        proxy_username="your_username",
        proxy_password="your_password"
    )
    
    summarizer = Summarizer(proxy_config=proxy_config)
    video_ids = ["QAgR4uQ15rc", "dQw4w9WgXcQ", "another_id"]
    
    for video_id in video_ids:
        try:
            summary = await summarizer.get_summary(video_id=video_id)
            print(f"Processed {video_id}: {summary[:100]}...")
            
            # Be respectful - add delay between requests
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error processing {video_id}: {e}")

asyncio.run(batch_process())
```

### Error Handling

```python
from tldwatch import Summarizer, create_webshare_proxy, ProxyConfigError, SummarizerError

try:
    proxy_config = create_webshare_proxy(
        proxy_username="your_username",
        proxy_password="your_password"
    )
    
    summarizer = Summarizer(proxy_config=proxy_config)
    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
    
except ProxyConfigError as e:
    print(f"Proxy setup error: {e}")
    # Check credentials, account status
    
except SummarizerError as e:
    if "blocked" in str(e).lower():
        print("IP blocking detected - proxy might need time to rotate")
    elif "transcript" in str(e).lower():
        print("Video has no available transcripts")
    else:
        print(f"Summarization error: {e}")
```

## Files Added/Modified

### New Files
- `src/tldwatch/core/proxy_config.py`: Proxy configuration classes
- `examples/proxy_usage.py`: Comprehensive proxy examples
- `examples/complete_proxy_example.py`: Full integration example
- `tests/integration/proxy/test_proxy_setup.py`: Setup verification script
- `PROXY_SETUP.md`: Detailed proxy setup guide
- `WEBSHARE_INTEGRATION.md`: This file

### Modified Files
- `src/tldwatch/core/summarizer.py`: Added proxy support
- `src/tldwatch/core/config.py`: Added proxy configuration support
- `src/tldwatch/cli/main.py`: Added CLI proxy options
- `src/tldwatch/__init__.py`: Exported proxy classes
- `README.md`: Added proxy configuration section

## Best Practices

1. **Use Residential Proxies**: They're less likely to be blocked than datacenter proxies
2. **Add Delays**: Include 2-3 second delays between requests
3. **Handle Errors**: Implement proper retry logic for failed requests
4. **Monitor Usage**: Track your Webshare usage to avoid exceeding limits
5. **Secure Credentials**: Store proxy credentials in environment variables, not code
6. **Test First**: Use `tests/integration/proxy/test_proxy_setup.py` to verify your configuration

## Troubleshooting

### Common Issues

1. **"Proxy configuration error"**
   - Verify Webshare username/password
   - Check account status and billing
   - Ensure you purchased "Residential" proxies

2. **"Access blocked" errors**
   - Proxy IP might be blocked (rare with residential proxies)
   - Try waiting a few minutes for IP rotation
   - Contact Webshare support if persistent

3. **Slow performance**
   - Proxies add latency (normal)
   - Consider using faster Webshare endpoints
   - Reduce concurrent requests

### Testing Commands

```bash
# Test proxy configuration
python tests/integration/proxy/test_proxy_setup.py

# Test CLI with proxy
tldwatch --webshare-username "user" --webshare-password "pass" --video-id "QAgR4uQ15rc"

# Test library integration
python examples/proxy_usage.py

# Complete example
python examples/complete_proxy_example.py
```

## Cost Considerations

- Webshare residential proxies are paid services
- Pricing varies by package size and usage
- Monitor usage through Webshare dashboard
- Consider your expected request volume when choosing a package

## Support

For issues:
1. Check Webshare account status and billing
2. Verify proxy credentials
3. Test with `tests/integration/proxy/test_proxy_setup.py`
4. Review error messages for specific guidance
5. Check tldwatch logs for detailed error information

## Alternative Proxy Providers

While this integration is optimized for Webshare, you can use other providers with the generic proxy configuration:

```python
from tldwatch import create_generic_proxy

proxy_config = create_generic_proxy(
    http_url="http://user:pass@your-proxy.com:8080",
    https_url="https://user:pass@your-proxy.com:8080"
)
```

However, Webshare is recommended due to:
- Reliable rotating residential IPs
- Good performance with YouTube
- Integrated setup process
- Proven compatibility
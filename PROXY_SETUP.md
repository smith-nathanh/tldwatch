# Proxy Configuration for tldwatch

This guide explains how to set up proxy configuration to avoid IP blocking when fetching YouTube transcripts.

## Why Use Proxies?

YouTube may block IP addresses that make too many requests, especially from cloud providers (AWS, Google Cloud, Azure, etc.). This can result in errors like:
- `RequestBlocked` or `IpBlocked` exceptions
- Failed transcript fetching
- Rate limiting issues

Using rotating residential proxies helps avoid these issues by distributing requests across multiple IP addresses.

## Recommended: Webshare Rotating Residential Proxies

[Webshare](https://www.webshare.io/) provides reliable rotating residential proxies that work well with YouTube.

### Setup Steps

1. **Create Account**: Sign up at [https://www.webshare.io/](https://www.webshare.io/)

2. **Purchase Package**: Buy a "Residential" proxy package
   - ⚠️ **Important**: Choose "Residential" NOT "Proxy Server" or "Static Residential"
   - Residential proxies rotate automatically and are less likely to be blocked

3. **Get Credentials**: Visit [Proxy Settings](https://dashboard.webshare.io/proxy/settings) to get:
   - Proxy Username
   - Proxy Password

4. **Set Environment Variables**:
   ```bash
   export WEBSHARE_PROXY_USERNAME="your_username"
   export WEBSHARE_PROXY_PASSWORD="your_password"
   ```

### Usage Examples

#### Library Usage

```python
import asyncio
from tldwatch import Summarizer, create_webshare_proxy

async def main():
    # Create proxy configuration
    proxy_config = create_webshare_proxy(
        proxy_username="your_username",
        proxy_password="your_password"
    )
    
    # Initialize summarizer with proxy
    summarizer = Summarizer(
        provider="openai",
        proxy_config=proxy_config
    )
    
    # Get summary (will use proxy)
    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
    print(summary)

asyncio.run(main())
```

#### CLI Usage

```bash
# Using environment variables
export WEBSHARE_PROXY_USERNAME="your_username"
export WEBSHARE_PROXY_PASSWORD="your_password"
tldwatch "https://www.youtube.com/watch?v=QAgR4uQ15rc"

# Using command line arguments
tldwatch --webshare-username "your_username" --webshare-password "your_password" "https://www.youtube.com/watch?v=QAgR4uQ15rc"
```

#### Configuration File

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

## Alternative: Generic HTTP/HTTPS Proxies

If you have your own proxy service or prefer a different provider:

### Setup

```bash
export HTTP_PROXY_URL="http://user:pass@proxy.example.com:8080"
export HTTPS_PROXY_URL="https://user:pass@proxy.example.com:8080"
```

### Usage Examples

#### Library Usage

```python
from tldwatch import Summarizer, create_generic_proxy

proxy_config = create_generic_proxy(
    http_url="http://user:pass@proxy.example.com:8080",
    https_url="https://user:pass@proxy.example.com:8080"
)

summarizer = Summarizer(proxy_config=proxy_config)
```

#### CLI Usage

```bash
tldwatch --http-proxy "http://user:pass@proxy.example.com:8080" --https-proxy "https://user:pass@proxy.example.com:8080" "https://www.youtube.com/watch?v=QAgR4uQ15rc"
```

#### Configuration File

```json
{
  "provider": "openai",
  "proxy": {
    "type": "generic",
    "http_url": "http://user:pass@proxy.example.com:8080",
    "https_url": "https://user:pass@proxy.example.com:8080"
  }
}
```

## Batch Processing with Proxies

When processing multiple videos, proxies help avoid rate limiting:

```python
import asyncio
from tldwatch import Summarizer, create_webshare_proxy

async def batch_process():
    proxy_config = create_webshare_proxy(
        proxy_username="your_username",
        proxy_password="your_password"
    )
    
    summarizer = Summarizer(proxy_config=proxy_config)
    
    video_ids = ["QAgR4uQ15rc", "dQw4w9WgXcQ", "another_video_id"]
    
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

## Error Handling

```python
from tldwatch import Summarizer, create_webshare_proxy, ProxyConfigError

try:
    proxy_config = create_webshare_proxy(
        proxy_username="invalid_user",
        proxy_password="invalid_pass"
    )
    summarizer = Summarizer(proxy_config=proxy_config)
    summary = await summarizer.get_summary(video_id="QAgR4uQ15rc")
    
except ProxyConfigError as e:
    print(f"Proxy configuration error: {e}")
    # Handle proxy setup issues
    
except Exception as e:
    if "blocked" in str(e).lower():
        print("IP blocking detected. Check your proxy configuration.")
    else:
        print(f"Other error: {e}")
```

## Best Practices

1. **Use Residential Proxies**: They're less likely to be blocked than datacenter proxies
2. **Add Delays**: Include delays between requests to be respectful to YouTube's servers
3. **Handle Errors**: Implement proper error handling for proxy and network issues
4. **Monitor Usage**: Keep track of your proxy usage to avoid exceeding limits
5. **Secure Credentials**: Store proxy credentials securely (environment variables, not in code)

## Troubleshooting

### Common Issues

1. **"Proxy configuration error"**
   - Check your username/password
   - Verify your Webshare account is active
   - Ensure you purchased the correct proxy type (Residential)

2. **"Access blocked" or "Forbidden"**
   - Your proxy IP might be blocked
   - Try a different proxy provider
   - Contact your proxy provider for support

3. **Slow performance**
   - Proxies add latency
   - Consider using faster proxy endpoints
   - Reduce concurrent requests

### Testing Your Setup

```python
# Test proxy configuration
from tldwatch import create_webshare_proxy, ProxyConfigError

try:
    proxy_config = create_webshare_proxy(
        proxy_username="your_username",
        proxy_password="your_password"
    )
    print("✓ Proxy configuration created successfully")
    print(f"Configuration: {proxy_config}")
except ProxyConfigError as e:
    print(f"✗ Proxy configuration failed: {e}")
```

## Support

If you encounter issues:

1. Check the [youtube-transcript-api documentation](https://github.com/jdepoix/youtube-transcript-api)
2. Verify your proxy provider's status
3. Test with a simple video ID first
4. Check the tldwatch logs for detailed error messages

## Cost Considerations

- Webshare residential proxies are paid services
- Costs vary based on usage and package size
- Consider your expected usage volume when choosing a package
- Monitor your usage to avoid unexpected charges
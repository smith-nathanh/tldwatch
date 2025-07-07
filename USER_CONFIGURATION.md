# TLDWatch User Configuration Guide

TLDWatch supports comprehensive user configuration to customize your defaults for providers, models, caching, proxy settings, and more. This eliminates the need to specify the same parameters repeatedly.

## Quick Start

```bash
# Create an example configuration file
tldwatch --create-config

# View your current configuration
tldwatch --show-config

# Edit the configuration file
nano ~/.config/tldwatch/config.json
```

## Configuration File Location

TLDWatch looks for configuration files in this order:
1. `~/.config/tldwatch/config.json`
2. `~/.config/tldwatch/config.yaml`
3. `~/.config/tldwatch/config.yml`

The first file found will be used. If no configuration file exists, TLDWatch uses built-in defaults.

## Complete Configuration Format

### JSON Format (config.json)

```json
{
  "default_provider": "openai",
  "default_temperature": 0.7,
  "default_chunking_strategy": "standard",
  "cache": {
    "enabled": true,
    "cache_dir": null,
    "max_age_days": 30
  },
  "proxy": {
    "type": "webshare",
    "proxy_username": "your_username",
    "proxy_password": "your_password"
  },
  "providers": {
    "openai": {
      "default_model": "gpt-4o-mini",
      "temperature": 0.7
    },
    "anthropic": {
      "default_model": "claude-3-5-sonnet-20241022",
      "temperature": 0.5
    },
    "google": {
      "default_model": "gemini-2.5-flash"
    },
    "groq": {
      "default_model": "llama-3.1-8b-instant"
    },
    "deepseek": {
      "default_model": "deepseek-chat"
    },
    "cerebras": {
      "default_model": "llama3.1-8b"
    },
    "ollama": {
      "default_model": "llama3.1:8b"
    }
  }
}
```

### YAML Format (config.yaml)

```yaml
default_provider: openai
default_temperature: 0.7
default_chunking_strategy: standard

cache:
  enabled: true
  cache_dir: null  # Uses default if null
  max_age_days: 30

proxy:
  type: webshare
  proxy_username: your_username
  proxy_password: your_password

providers:
  openai:
    default_model: gpt-4o-mini
    temperature: 0.7
  anthropic:
    default_model: claude-3-5-sonnet-20241022
    temperature: 0.5
  google:
    default_model: gemini-2.5-flash
  groq:
    default_model: llama-3.1-8b-instant
```

## Configuration Options

### Global Defaults

- **`default_provider`**: Which provider to use by default
  - Options: `openai`, `anthropic`, `google`, `groq`, `deepseek`, `cerebras`, `ollama`
  - Default: `openai`

- **`default_temperature`**: Default temperature for text generation
  - Range: 0.0 to 1.0
  - Default: 0.7

- **`default_chunking_strategy`**: How to handle long texts
  - Options: `none`, `standard`, `small`, `large`
  - Default: `standard`

### Caching Configuration

- **`cache.enabled`**: Enable/disable caching (default: `true`)
- **`cache.cache_dir`**: Custom cache directory (default: uses system cache)
- **`cache.max_age_days`**: Days to keep cache entries (default: 30)

Cache location defaults:
- Linux/macOS: `~/.cache/tldwatch/summaries/`
- Windows: `%LOCALAPPDATA%\tldwatch\summaries\`

### Proxy Configuration

**Webshare Rotating Residential Proxies (Recommended):**
```json
{
  "proxy": {
    "type": "webshare",
    "proxy_username": "your_username",
    "proxy_password": "your_password"
  }
}
```

**Generic HTTP/HTTPS Proxies:**
```json
{
  "proxy": {
    "type": "generic",
    "http_url": "http://user:pass@proxy.example.com:8080",
    "https_url": "https://user:pass@proxy.example.com:8080"
  }
}
```

### Provider-Specific Settings

For each provider, you can specify:

- **`default_model`**: Which model to use for this provider
- **`temperature`**: Temperature override for this provider (optional)

#### Available Models by Provider

**OpenAI:**
- `gpt-4o` (most capable)
- `gpt-4o-mini` (default, cost-effective)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

**Anthropic:**
- `claude-3-5-sonnet-20241022` (default, most capable)
- `claude-3-5-haiku-20241022` (fast and cost-effective)
- `claude-3-opus-20240229` (most capable, slower)

**Google:**
- `gemini-2.5-pro` (most capable)
- `gemini-2.5-flash` (default, fast)
- `gemini-1.0-pro`

**Groq:**
- `llama-3.1-70b-versatile` (most capable)
- `llama-3.1-8b-instant` (default, fastest)
- `mixtral-8x7b-32768`

**DeepSeek:**
- `deepseek-chat` (default)

**Cerebras:**
- `llama3.1-70b` (most capable)
- `llama3.1-8b` (default, fastest)

**Ollama:**
- `llama3.1:8b` (default)
- `llama3.1:70b`
- `mistral:7b`
- Any model you have installed locally

## Priority and Configuration Resolution

Settings are resolved in this order (highest to lowest priority):

1. **Explicit parameters** - Values you pass directly to functions/CLI
2. **User configuration** - Values from your config file
3. **Package defaults** - Built-in fallback values

### Example Usage

With this configuration:
```json
{
  "default_provider": "anthropic",
  "default_temperature": 0.3,
  "providers": {
    "anthropic": {
      "default_model": "claude-3-5-sonnet-20241022",
      "temperature": 0.2
    },
    "openai": {
      "default_model": "gpt-4o",
      "temperature": 0.8
    }
  }
}
```

**Usage examples:**

```python
from tldwatch import summarize_video

# Uses: anthropic, claude-3-5-sonnet-20241022, temperature=0.2
summary = await summarize_video("video_id")

# Uses: openai, gpt-4o, temperature=0.8
summary = await summarize_video("video_id", provider="openai")

# Uses: anthropic, claude-3-5-sonnet-20241022, temperature=0.9 (override)
summary = await summarize_video("video_id", temperature=0.9)
```

## Caching System

TLDWatch includes automatic caching to save time and API costs:

### Cache Behavior
- Stores summaries with metadata (provider, model, chunking strategy, temperature)
- Only returns cached summaries when parameters match exactly
- Different parameters create separate cache entries

### Cache Management

```bash
# View cache statistics
tldwatch-cache stats

# List cached summaries
tldwatch-cache list

# Show specific video cache
tldwatch-cache show <video_id>

# Clear cache for specific video
tldwatch-cache clear --video-id <video_id>

# Clear all cache
tldwatch-cache clear

# Clean up old entries
tldwatch-cache cleanup --max-age-days 7
```

### Programmatic Cache Control

```python
from tldwatch import Summarizer, get_cache_stats, clear_cache

# Disable cache for one request
summary = await summarizer.summarize(video_id, use_cache=False)

# Force regeneration (clears cache first)
tldwatch "video_url" --force-regenerate

# Get cache statistics
stats = get_cache_stats()
print(f"Cached videos: {stats['total_videos']}")
```

## Proxy Configuration

### Why Use Proxies?

YouTube may block IP addresses making frequent requests, especially from cloud providers. Proxies help avoid:
- `RequestBlocked` or `IpBlocked` exceptions
- Failed transcript fetching
- Rate limiting issues

### Recommended: Webshare Rotating Residential Proxies

1. **Setup**: Sign up at [https://www.webshare.io/](https://www.webshare.io/)
2. **Purchase**: Buy a "Residential" proxy package (⚠️ NOT "Proxy Server")
3. **Configure**: Add credentials to your config file or environment variables

```bash
# Environment variables
export WEBSHARE_PROXY_USERNAME="your_username"
export WEBSHARE_PROXY_PASSWORD="your_password"

# CLI usage
tldwatch --webshare-username "user" --webshare-password "pass" "video_url"
```

### Alternative: Generic HTTP/HTTPS Proxies

```bash
# Environment variables
export HTTP_PROXY_URL="http://user:pass@proxy.example.com:8080"
export HTTPS_PROXY_URL="https://user:pass@proxy.example.com:8080"

# CLI usage
tldwatch --http-proxy "http://user:pass@proxy.example.com:8080" "video_url"
```

### Library Usage with Proxies

```python
from tldwatch import Summarizer, create_webshare_proxy

# Webshare proxy
proxy_config = create_webshare_proxy(
    proxy_username="your_username",
    proxy_password="your_password"
)

summarizer = Summarizer(proxy_config=proxy_config)
summary = await summarizer.summarize("video_id")
```

## CLI Configuration Commands

```bash
# Create example configuration file
tldwatch --create-config

# Show current configuration
tldwatch --show-config

# List available providers
tldwatch --list-providers

# Show default models for each provider
tldwatch --show-defaults
```

## Environment Variables

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export CEREBRAS_API_KEY="your-cerebras-key"
# No API key needed for Ollama (local)
```

## Chunking Strategies

- **`none`**: Submit entire transcript without chunking (best for short videos)
- **`standard`**: ~4000 character chunks (balanced approach)
- **`small`**: ~2000 character chunks (more detailed processing)
- **`large`**: ~8000 character chunks (better context preservation)

## Example Configurations

### Minimal Configuration
```json
{
  "default_provider": "anthropic"
}
```

### Production Configuration
```json
{
  "default_provider": "openai",
  "default_temperature": 0.5,
  "default_chunking_strategy": "large",
  "cache": {
    "enabled": true,
    "max_age_days": 60
  },
  "proxy": {
    "type": "webshare",
    "proxy_username": "your_username",
    "proxy_password": "your_password"
  },
  "providers": {
    "openai": {
      "default_model": "gpt-4o-mini",
      "temperature": 0.5
    },
    "anthropic": {
      "default_model": "claude-3-5-sonnet-20241022",
      "temperature": 0.3
    }
  }
}
```

### Cost-Optimized Configuration
```json
{
  "default_provider": "groq",
  "default_temperature": 0.7,
  "default_chunking_strategy": "standard",
  "providers": {
    "groq": {
      "default_model": "llama-3.1-8b-instant"
    },
    "openai": {
      "default_model": "gpt-4o-mini"
    },
    "anthropic": {
      "default_model": "claude-3-5-haiku-20241022"
    }
  }
}
```

## Troubleshooting

### Configuration Not Loading

1. Check file location: `~/.config/tldwatch/config.json`
2. Validate JSON/YAML syntax
3. Check file permissions
4. Use `tldwatch --show-config` to see what's loaded

### Proxy Issues

1. **"Proxy configuration error"**: Check credentials and account status
2. **"Access blocked"**: Try different proxy provider or contact support
3. **Slow performance**: Proxies add latency, consider faster endpoints

### Cache Issues

1. **Cache not working**: Check `cache.enabled` and directory permissions
2. **Parameters not matching**: Ensure exact same provider/model/temperature
3. **Storage issues**: Use `tldwatch-cache cleanup` to free space

### Provider Issues

```bash
# List available providers
tldwatch --list-providers

# Show default models
tldwatch --show-defaults

# Test specific provider
tldwatch "test text" --provider openai --chunking none
```

### Invalid Configuration

```bash
# Check your configuration
tldwatch --show-config

# Recreate default configuration
rm ~/.config/tldwatch/config.*
tldwatch --create-config
```

## Support

For configuration issues:
1. Check this documentation
2. Use `--show-config` to debug
3. Recreate configuration with `--create-config`
4. Check file permissions and syntax
# TLDWatch User Configuration Guide

TLDWatch supports user configuration files to customize your default settings. This allows you to set your preferred provider, models, temperature, and chunking strategy without having to specify them every time.

## Quick Start

```bash
# Create an example configuration file
tldwatch-simple --create-config

# View your current configuration
tldwatch-simple --show-config

# Edit the configuration file
nano ~/.config/tldwatch/config.json
```

## Configuration File Location

TLDWatch looks for configuration files in this order:
1. `~/.config/tldwatch/config.json`
2. `~/.config/tldwatch/config.yaml`
3. `~/.config/tldwatch/config.yml`

The first file found will be used. If no configuration file exists, TLDWatch uses built-in defaults.

## Configuration Format

### JSON Format (config.json)

```json
{
  "default_provider": "anthropic",
  "default_temperature": 0.3,
  "default_chunking_strategy": "large",
  "providers": {
    "openai": {
      "default_model": "gpt-4o",
      "temperature": 0.8
    },
    "anthropic": {
      "default_model": "claude-3-5-sonnet-20241022",
      "temperature": 0.2
    },
    "google": {
      "default_model": "gemini-1.5-pro"
    },
    "groq": {
      "default_model": "llama-3.1-70b-versatile",
      "temperature": 0.6
    }
  }
}
```

### YAML Format (config.yaml)

```yaml
default_provider: anthropic
default_temperature: 0.3
default_chunking_strategy: large

providers:
  openai:
    default_model: gpt-4o
    temperature: 0.8
  anthropic:
    default_model: claude-3-5-sonnet-20241022
    temperature: 0.2
  google:
    default_model: gemini-1.5-pro
  groq:
    default_model: llama-3.1-70b-versatile
    temperature: 0.6
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
- `gemini-1.5-pro` (most capable)
- `gemini-1.5-flash` (default, fast)
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

## How Configuration Priority Works

Settings are resolved in this order (highest to lowest priority):

1. **Explicit parameters** - Values you pass directly to functions/CLI
2. **User configuration** - Values from your config file
3. **Package defaults** - Built-in fallback values

### Examples

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

# Uses: openai, gpt-4o-mini (override), temperature=0.8
summary = await summarize_video("video_id", provider="openai", model="gpt-4o-mini")
```

## CLI Configuration Commands

```bash
# Create example configuration file
tldwatch-simple --create-config

# Show current configuration
tldwatch-simple --show-config

# List available providers
tldwatch-simple --list-providers

# Show default models for each provider
tldwatch-simple --show-defaults
```

## Environment Variables

You still need to set API keys as environment variables:

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

## Configuration Validation

TLDWatch validates your configuration and will:
- Warn about unknown providers or models
- Fall back to defaults for invalid values
- Show helpful error messages for configuration issues

## Troubleshooting

### Configuration Not Loading

1. Check file location: `~/.config/tldwatch/config.json`
2. Validate JSON/YAML syntax
3. Check file permissions
4. Use `--show-config` to see what's loaded

### Invalid Configuration

```bash
# Check your configuration
tldwatch-simple --show-config

# Recreate default configuration
rm ~/.config/tldwatch/config.*
tldwatch-simple --create-config
```

### Provider Issues

```bash
# List available providers
tldwatch-simple --list-providers

# Show default models
tldwatch-simple --show-defaults

# Test specific provider
tldwatch-simple "test text" --provider openai --chunking none
```

## Example Configurations

### Minimal Configuration
```json
{
  "default_provider": "anthropic"
}
```

### Power User Configuration
```json
{
  "default_provider": "anthropic",
  "default_temperature": 0.1,
  "default_chunking_strategy": "large",
  "providers": {
    "openai": {
      "default_model": "gpt-4o",
      "temperature": 0.3
    },
    "anthropic": {
      "default_model": "claude-3-5-sonnet-20241022",
      "temperature": 0.1
    },
    "google": {
      "default_model": "gemini-1.5-pro",
      "temperature": 0.2
    },
    "groq": {
      "default_model": "llama-3.1-70b-versatile",
      "temperature": 0.4
    },
    "ollama": {
      "default_model": "llama3.1:70b",
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

## Migration from Legacy Interface

The user configuration system is completely separate from the legacy interface. Your existing code will continue to work unchanged, and you can gradually adopt the new simplified interface with user configuration.

## Support

For issues with user configuration:
1. Check this documentation
2. Use `--show-config` to debug
3. Recreate configuration with `--create-config`
4. Check file permissions and syntax
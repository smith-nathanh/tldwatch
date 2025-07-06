"""
Unified provider implementation that consolidates all LLM providers into a single, simplified interface.
This replaces the complex per-provider implementations with a generic approach.
"""

import asyncio
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
import tiktoken
import yaml

from ..user_config import get_user_config

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Simple chunking strategies"""
    NONE = "none"  # Submit entire transcript
    STANDARD = "standard"  # Default chunking approach
    SMALL = "small"  # Smaller chunks for detailed processing
    LARGE = "large"  # Larger chunks for context preservation


@dataclass
class ProviderConfig:
    """Simple provider configuration"""
    name: str
    api_key_env: str
    api_base: str
    default_model: str


class ProviderError(Exception):
    """Base exception for provider errors"""
    pass


class UnifiedProvider:
    """
    Unified provider that handles all LLM providers with a simplified interface.
    No more complex rate limiting, context window management, or custom retry logic.
    """
    
    # Provider configurations
    PROVIDERS = {
        "openai": ProviderConfig(
            name="openai",
            api_key_env="OPENAI_API_KEY",
            api_base="https://api.openai.com/v1",
            default_model="gpt-4o-mini"
        ),
        "anthropic": ProviderConfig(
            name="anthropic", 
            api_key_env="ANTHROPIC_API_KEY",
            api_base="https://api.anthropic.com/v1",
            default_model="claude-3-5-sonnet-20241022"
        ),
        "google": ProviderConfig(
            name="google",
            api_key_env="GEMINI_API_KEY", 
            api_base="https://generativelanguage.googleapis.com/v1beta",
            default_model="gemini-1.5-flash"
        ),
        "groq": ProviderConfig(
            name="groq",
            api_key_env="GROQ_API_KEY",
            api_base="https://api.groq.com/openai/v1",
            default_model="llama-3.1-8b-instant"
        ),
        "deepseek": ProviderConfig(
            name="deepseek",
            api_key_env="DEEPSEEK_API_KEY",
            api_base="https://api.deepseek.com/v1",
            default_model="deepseek-chat"
        ),
        "cerebras": ProviderConfig(
            name="cerebras",
            api_key_env="CEREBRAS_API_KEY", 
            api_base="https://api.cerebras.ai/v1",
            default_model="llama3.1-8b"
        ),
        "ollama": ProviderConfig(
            name="ollama",
            api_key_env=None,  # Local provider
            api_base="http://localhost:11434/v1",
            default_model="llama3.1:8b"
        )
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        chunking_strategy: Optional[Union[str, ChunkingStrategy]] = None
    ):
        """
        Initialize the unified provider.
        
        Args:
            provider: Provider name (uses user default or "openai" if None)
            model: Model name (uses provider default if None)
            temperature: Generation temperature (uses user default or 0.7 if None)
            chunking_strategy: How to handle long texts (uses user default or "standard" if None)
        """
        # Load user configuration
        user_config = get_user_config()
        
        # Apply user defaults
        if provider is None:
            provider = user_config.get_default_provider() or "openai"
        
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: {list(self.PROVIDERS.keys())}")
        
        self.config = self.PROVIDERS[provider]
        
        # Get model (user config > parameter > package default)
        if model is None:
            model = user_config.get_default_model(provider)
        if model is None:
            model = self._get_default_model(provider)
        self.model = model
        
        # Get temperature (user config > parameter > default)
        if temperature is None:
            temperature = user_config.get_default_temperature(provider)
        if temperature is None:
            temperature = 0.7
        self.temperature = temperature
        
        # Get chunking strategy (user config > parameter > default)
        if chunking_strategy is None:
            chunking_strategy = user_config.get_default_chunking_strategy() or "standard"
        
        if isinstance(chunking_strategy, str):
            try:
                chunking_strategy = ChunkingStrategy(chunking_strategy.lower())
            except ValueError:
                raise ValueError(f"Invalid chunking strategy: {chunking_strategy}. "
                               f"Choose from: {[s.value for s in ChunkingStrategy]}")
        
        self.chunking_strategy = chunking_strategy
        
        # Get API key if required
        self.api_key = None
        if self.config.api_key_env:
            self.api_key = os.getenv(self.config.api_key_env)
            # Note: We don't raise an error here to allow testing and initialization
            # The error will be raised when actually making requests
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model from config file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['providers'][provider]['default_model']
        except Exception:
            # Fallback to hardcoded defaults
            return self.PROVIDERS[provider].default_model
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Simple, generic text chunking that works for any model.
        No more complex context window management.
        """
        if self.chunking_strategy == ChunkingStrategy.NONE:
            return [text]
        
        # Simple character-based chunking with sentence boundary detection
        if self.chunking_strategy == ChunkingStrategy.SMALL:
            chunk_size = 2000
        elif self.chunking_strategy == ChunkingStrategy.LARGE:
            chunk_size = 8000
        else:  # STANDARD
            chunk_size = 4000
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look back up to 200 chars for a sentence ending
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i-1] in '.!?':
                        end = i
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    async def generate_summary(self, text: str) -> str:
        """
        Generate summary with unified interface.
        Handles chunking automatically based on strategy.
        """
        chunks = self.chunk_text(text)
        
        if len(chunks) == 1:
            # Single chunk - direct summarization
            return await self._make_request(chunks[0])
        else:
            # Multiple chunks - summarize each then combine
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                summary = await self._make_request(chunk)
                chunk_summaries.append(summary)
                
                # Simple rate limiting - wait between requests
                if i < len(chunks) - 1:
                    await asyncio.sleep(1)
            
            # Combine summaries
            combined_text = "\n\n".join(chunk_summaries)
            if len(combined_text) > 4000:
                # If combined summaries are still long, summarize them
                return await self._make_request(
                    f"Please provide a comprehensive summary of these section summaries:\n\n{combined_text}",
                    is_meta_summary=True
                )
            else:
                return combined_text
    
    async def _make_request(self, text: str, is_meta_summary: bool = False) -> str:
        """
        Unified request handler for all providers.
        Simple retry logic without complex rate limiting.
        """
        # Check API key before making requests
        if self.config.api_key_env and not self.api_key:
            raise ProviderError(f"API key not found. Set {self.config.api_key_env} environment variable.")
        
        if is_meta_summary:
            prompt = f"Please provide a comprehensive summary of these section summaries:\n\n{text}"
        else:
            prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.config.name == "anthropic":
                    return await self._anthropic_request(prompt)
                elif self.config.name == "google":
                    return await self._google_request(prompt)
                else:
                    # OpenAI-compatible providers (openai, groq, deepseek, cerebras, ollama)
                    return await self._openai_compatible_request(prompt)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ProviderError(f"Failed after {max_retries} attempts: {str(e)}")
                
                # Simple exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
    
    async def _openai_compatible_request(self, prompt: str) -> str:
        """Handle OpenAI-compatible APIs (OpenAI, Groq, DeepSeek, Cerebras, Ollama)"""
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that generates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature
            }
            
            async with session.post(
                f"{self.config.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    raise Exception("Rate limited")
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error ({response.status}): {error_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def _anthropic_request(self, prompt: str) -> str:
        """Handle Anthropic API"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 4096
            }
            
            async with session.post(
                f"{self.config.api_base}/messages",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    raise Exception("Rate limited")
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error ({response.status}): {error_text}")
                
                result = await response.json()
                return result["content"][0]["text"]
    
    async def _google_request(self, prompt: str) -> str:
        """Handle Google Gemini API"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.config.api_base}/models/{self.model}:generateContent?key={self.api_key}"
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.temperature}
            }
            
            async with session.post(
                url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 429:
                    await asyncio.sleep(5)
                    raise Exception("Rate limited")
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error ({response.status}): {error_text}")
                
                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers"""
        return list(cls.PROVIDERS.keys())
    
    @classmethod
    def get_default_model(cls, provider: str) -> str:
        """Get default model for a provider"""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        return cls.PROVIDERS[provider].default_model
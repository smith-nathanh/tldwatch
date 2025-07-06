"""
Simplified summarizer that uses the unified provider system.
Provides a clean, simple interface for YouTube video summarization.
"""

import logging
import re
from typing import Optional, Union

from youtube_transcript_api import YouTubeTranscriptApi

from ..utils.url_parser import extract_video_id, is_youtube_url
from .providers.unified_provider import ChunkingStrategy, UnifiedProvider
from .user_config import get_user_config

logger = logging.getLogger(__name__)


class SimpleSummarizer:
    """
    Simplified summarizer with a clean interface.
    
    Usage:
        summarizer = SimpleSummarizer()
        summary = await summarizer.summarize("https://youtube.com/watch?v=...")
        
        # Or with options:
        summary = await summarizer.summarize(
            "video_id", 
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            chunking_strategy="large"
        )
    """
    
    def __init__(self):
        """Initialize the simplified summarizer"""
        pass
    
    async def summarize(
        self,
        video_input: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        chunking_strategy: Optional[Union[str, ChunkingStrategy]] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Summarize a YouTube video or direct text.
        
        Args:
            video_input: YouTube URL, video ID, or direct text to summarize
            provider: LLM provider to use (uses user default or "openai" if None)
            model: Specific model to use (uses user/provider default if None)
            chunking_strategy: How to handle long texts (uses user default or "standard" if None)
            temperature: Generation temperature (uses user default or 0.7 if None)
            
        Returns:
            Generated summary as a string
            
        Raises:
            ValueError: If input is invalid
            Exception: If summarization fails
        """
        # Get transcript text
        text = await self._get_text(video_input)
        
        # Initialize provider (will use user config for defaults)
        unified_provider = UnifiedProvider(
            provider=provider,
            model=model,
            temperature=temperature,
            chunking_strategy=chunking_strategy
        )
        
        # Generate summary
        logger.info(f"Generating summary using {unified_provider.config.name} ({unified_provider.model})")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Chunking strategy: {unified_provider.chunking_strategy.value}")
        
        summary = await unified_provider.generate_summary(text)
        
        logger.info(f"Summary generated successfully ({len(summary)} characters)")
        return summary
    
    async def _get_text(self, video_input: str) -> str:
        """
        Extract text from video input (URL, video ID, or direct text).
        
        Args:
            video_input: YouTube URL, video ID, or direct text
            
        Returns:
            Text content to summarize
        """
        # Check if it's a YouTube URL or video ID
        if is_youtube_url(video_input):
            video_id = extract_video_id(video_input)
            if not video_id:
                raise ValueError("Could not extract video ID from URL")
            return await self._get_transcript(video_id)
        
        # Check if it looks like a video ID (11 characters, alphanumeric + hyphens/underscores)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', video_input):
            return await self._get_transcript(video_input)
        
        # Assume it's direct text
        if len(video_input.strip()) < 50:
            raise ValueError("Input text is too short. Provide a YouTube URL, video ID, or longer text.")
        
        return video_input.strip()
    
    async def _get_transcript(self, video_id: str) -> str:
        """
        Get transcript from YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Transcript text
        """
        try:
            logger.info(f"Fetching transcript for video: {video_id}")
            
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine transcript entries
            transcript_text = " ".join([entry['text'] for entry in transcript_list])
            
            # Clean up the text
            transcript_text = self._clean_transcript(transcript_text)
            
            logger.info(f"Transcript fetched successfully ({len(transcript_text)} characters)")
            return transcript_text
            
        except Exception as e:
            raise Exception(f"Failed to get transcript for video {video_id}: {str(e)}")
    
    def _clean_transcript(self, text: str) -> str:
        """
        Clean up transcript text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned transcript text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible), etc.
        
        # Fix common issues
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        return text.strip()
    
    @staticmethod
    def list_providers() -> list:
        """List available providers"""
        return UnifiedProvider.list_providers()
    
    @staticmethod
    def get_default_model(provider: str) -> str:
        """Get default model for a provider"""
        return UnifiedProvider.get_default_model(provider)
    
    @staticmethod
    def list_chunking_strategies() -> list:
        """List available chunking strategies"""
        return [strategy.value for strategy in ChunkingStrategy]


# Convenience function for quick usage
async def summarize_video(
    video_input: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    chunking_strategy: Optional[str] = None,
    temperature: Optional[float] = None
) -> str:
    """
    Convenience function to quickly summarize a video.
    
    Args:
        video_input: YouTube URL, video ID, or direct text
        provider: LLM provider to use (uses user default or "openai" if None)
        model: Specific model (uses user/provider default if None)
        chunking_strategy: How to handle long texts (uses user default or "standard" if None)
        temperature: Generation temperature (uses user default or 0.7 if None)
        
    Returns:
        Generated summary
    """
    summarizer = SimpleSummarizer()
    return await summarizer.summarize(
        video_input=video_input,
        provider=provider,
        model=model,
        chunking_strategy=chunking_strategy,
        temperature=temperature
    )
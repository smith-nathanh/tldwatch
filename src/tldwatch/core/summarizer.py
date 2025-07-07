"""
Main summarizer that uses the unified provider system.
Provides a clean, simple interface for YouTube video summarization with caching support.
"""

import logging
import re
from typing import Optional, Union

from youtube_transcript_api import YouTubeTranscriptApi

from ..utils.url_parser import extract_video_id, is_youtube_url
from .providers.unified_provider import ChunkingStrategy, UnifiedProvider

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Main summarizer with a clean interface.

    Usage:
        summarizer = Summarizer()
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
        """Initialize the summarizer"""
        pass

    async def summarize(
        self,
        video_input: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        chunking_strategy: Optional[Union[str, ChunkingStrategy]] = None,
        temperature: Optional[float] = None,
        use_cache: Optional[bool] = None,
    ) -> str:
        """
        Summarize a YouTube video or direct text.

        Args:
            video_input: YouTube URL, video ID, or direct text to summarize
            provider: LLM provider to use (uses user default or "openai" if None)
            model: Specific model to use (uses user/provider default if None)
            chunking_strategy: How to handle long texts (uses user default or "standard" if None)
            temperature: Generation temperature (uses user default or 0.7 if None)
            use_cache: Whether to use caching (uses user config default if None)

        Returns:
            Generated summary as a string

        Raises:
            ValueError: If input is invalid
            Exception: If summarization fails
        """
        # Get user config for cache settings
        from .user_config import get_user_config

        user_config = get_user_config()

        # Determine if we should use cache
        if use_cache is None:
            use_cache = user_config.is_cache_enabled()

        # Check if this is a video ID that we can cache
        video_id = None
        if is_youtube_url(video_input):
            video_id = extract_video_id(video_input)
        elif re.match(r"^[a-zA-Z0-9_-]{11}$", video_input):
            video_id = video_input

        # Initialize provider to get final configuration values
        unified_provider = UnifiedProvider(
            provider=provider,
            model=model,
            temperature=temperature,
            chunking_strategy=chunking_strategy,
        )

        # Try to get from cache if enabled and this is a video
        if use_cache and video_id:
            from ..utils.cache import get_cache

            cache = get_cache(user_config.get_cache_dir())

            # Check if we have a cached summary with matching parameters
            if cache.has_cached_summary(
                video_id=video_id,
                provider=unified_provider.config.name,
                model=unified_provider.model,
                chunking_strategy=unified_provider.chunking_strategy.value,
                temperature=unified_provider.temperature,
            ):
                cached_entry = cache.get_cached_summary(
                    video_id=video_id,
                    provider=unified_provider.config.name,
                    model=unified_provider.model,
                    chunking_strategy=unified_provider.chunking_strategy.value,
                    temperature=unified_provider.temperature,
                )
                if cached_entry:
                    logger.info(
                        f"Using cached summary for {video_id} "
                        f"(provider={cached_entry.provider}, model={cached_entry.model})"
                    )
                    return cached_entry.summary

        # Get transcript text (this will now check cache first for videos)
        text = await self._get_text_with_cache(video_input, use_cache)

        # Generate summary
        logger.info(
            f"Generating summary using {unified_provider.config.name} ({unified_provider.model})"
        )
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Chunking strategy: {unified_provider.chunking_strategy.value}")

        summary = await unified_provider.generate_summary(text)

        logger.info(f"Summary generated successfully ({len(summary)} characters)")

        # Cache the summary if enabled and this is a video
        if use_cache and video_id:
            from ..utils.cache import get_cache

            cache = get_cache(user_config.get_cache_dir())

            # For now, cache without metadata (can be enhanced later)
            video_metadata = {
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
            }

            # Cache the summary
            cache.cache_summary(
                video_id=video_id,
                summary=summary,
                provider=unified_provider.config.name,
                model=unified_provider.model,
                chunking_strategy=unified_provider.chunking_strategy.value,
                temperature=unified_provider.temperature,
                video_metadata=video_metadata,
            )

        return summary

    async def _get_text_with_cache(
        self, video_input: str, use_cache: bool = True
    ) -> str:
        """
        Extract text from video input with transcript caching support.

        Args:
            video_input: YouTube URL, video ID, or direct text
            use_cache: Whether to use transcript caching

        Returns:
            Text content to summarize
        """
        # Check if it's a YouTube URL or video ID
        video_id = None
        if is_youtube_url(video_input):
            video_id = extract_video_id(video_input)
            if not video_id:
                raise ValueError("Could not extract video ID from URL")
        elif re.match(r"^[a-zA-Z0-9_-]{11}$", video_input):
            video_id = video_input

        # If we have a video ID, try to get transcript (with caching)
        if video_id:
            return await self._get_transcript_with_cache(video_id, use_cache)

        # Assume it's direct text
        if len(video_input.strip()) < 50:
            raise ValueError(
                "Input text is too short. Provide a YouTube URL, video ID, or longer text."
            )

        return video_input.strip()

    async def _get_transcript_with_cache(
        self, video_id: str, use_cache: bool = True
    ) -> str:
        """
        Get transcript from YouTube video with caching support.

        Args:
            video_id: YouTube video ID
            use_cache: Whether to use transcript caching

        Returns:
            Transcript text
        """
        # Try to get from cache first if caching is enabled
        if use_cache:
            from ..utils.cache import get_cache
            from .user_config import get_user_config

            user_config = get_user_config()
            cache = get_cache(user_config.get_cache_dir())

            cached_transcript = cache.get_cached_transcript(video_id)
            if cached_transcript:
                logger.info(f"Using cached transcript for video: {video_id}")
                return cached_transcript

        # Fetch transcript from YouTube
        try:
            logger.info(f"Fetching transcript for video: {video_id}")

            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Combine transcript entries
            transcript_text = " ".join([entry["text"] for entry in transcript_list])

            # Clean up the text
            transcript_text = self._clean_transcript(transcript_text)

            logger.info(
                f"Transcript fetched successfully ({len(transcript_text)} characters)"
            )

            # Cache the transcript if caching is enabled
            if use_cache:
                from ..utils.cache import get_cache
                from .user_config import get_user_config

                user_config = get_user_config()
                cache = get_cache(user_config.get_cache_dir())

                video_metadata = {
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "video_id": video_id,
                }

                cache.cache_transcript(
                    video_id=video_id,
                    transcript=transcript_text,
                    video_metadata=video_metadata,
                )

            return transcript_text

        except Exception as e:
            raise Exception(f"Failed to get transcript for video {video_id}: {str(e)}")

    async def _get_text(self, video_input: str) -> str:
        """
        Extract text from video input (URL, video ID, or direct text).

        This method is kept for backward compatibility but now calls _get_text_with_cache.

        Args:
            video_input: YouTube URL, video ID, or direct text

        Returns:
            Text content to summarize
        """
        return await self._get_text_with_cache(video_input, use_cache=True)

    def _clean_transcript(self, text: str) -> str:
        """
        Clean up transcript text.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned transcript text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common transcript artifacts
        text = re.sub(r"\[.*?\]", "", text)  # Remove [Music], [Applause], etc.
        text = re.sub(r"\(.*?\)", "", text)  # Remove (inaudible), etc.

        # Fix common issues
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")

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
    temperature: Optional[float] = None,
    use_cache: Optional[bool] = None,
) -> str:
    """
    Convenience function to quickly summarize a video.

    Args:
        video_input: YouTube URL, video ID, or direct text
        provider: LLM provider to use (uses user default or "openai" if None)
        model: Specific model (uses user/provider default if None)
        chunking_strategy: How to handle long texts (uses user default or "standard" if None)
        temperature: Generation temperature (uses user default or 0.7 if None)
        use_cache: Whether to use caching (uses user config default if None)

    Returns:
        Generated summary
    """
    summarizer = Summarizer()
    return await summarizer.summarize(
        video_input=video_input,
        provider=provider,
        model=model,
        chunking_strategy=chunking_strategy,
        temperature=temperature,
        use_cache=use_cache,
    )

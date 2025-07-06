"""
Main summarizer class with improved chunking and provider handling.

This module provides the core functionality for generating summaries from
YouTube video transcripts or direct text input, with flexible chunking strategies.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from youtube_transcript_api import YouTubeTranscriptApi

from ..utils.url_parser import extract_video_id, is_youtube_url
from .chunking import ChunkingConfig, ChunkingStrategy, get_default_chunking_config, prompt_for_chunking_strategy, split_text
from .providers.base_provider import BaseProvider, ProviderError
from .providers.provider_factory import ProviderFactory
from .proxy_config import TldwatchProxyConfig

logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """Base exception for Summarizer errors"""
    pass


class Summarizer:
    """Main class for generating summaries from YouTube video transcripts or direct text input"""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        chunking_config: Optional[ChunkingConfig] = None,
        use_full_context: bool = False,
        youtube_api_key: Optional[str] = None,
        proxy_config: Optional[TldwatchProxyConfig] = None,
        interactive: bool = False,
    ):
        self.provider_name = provider.lower()
        self.chunking_config = chunking_config
        self.use_full_context = use_full_context
        self.youtube_api_key = youtube_api_key
        self.proxy_config = proxy_config
        self.interactive = interactive
        self._lock = asyncio.Lock()
        self._active_tasks: set[asyncio.Task] = set()

        # Initialize provider using factory
        try:
            self.provider = ProviderFactory.create_provider(
                provider_name=provider,
                model=model,
                temperature=temperature,
                use_full_context=use_full_context,
            )
        except ValueError as e:
            available = ", ".join(ProviderFactory.get_available_providers().keys())
            raise ValueError(
                f"Unsupported provider: {provider}. Available providers: {available}"
            ) from e

        # State variables
        self.video_id: Optional[str] = None
        self.transcript: Optional[str] = None
        self.summary: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def validate_input(
        self,
        video_id: Optional[str] = None,
        url: Optional[str] = None,
        stdin_content: Optional[str] = None,
    ) -> str:
        """Validate and process input source to get video ID or transcript"""
        if video_id:
            return video_id
        elif url:
            if not is_youtube_url(url):
                raise SummarizerError("Invalid YouTube URL")
            video_id = extract_video_id(url)
            if not video_id:
                raise SummarizerError("Could not extract video ID from URL")
            return video_id
        elif stdin_content:
            if is_youtube_url(stdin_content):
                video_id = extract_video_id(stdin_content)
                if not video_id:
                    raise SummarizerError("Could not extract video ID from URL")
                return video_id
            return stdin_content
        raise SummarizerError("No valid input source provided")

    async def get_summary(
        self,
        video_id: Optional[str] = None,
        url: Optional[str] = None,
        transcript_text: Optional[str] = None,
    ) -> str:
        """Generate a summary from either a YouTube video or direct transcript input"""
        try:
            if transcript_text is not None:
                logging.info("Using direct transcript input")
                self.transcript = self._clean_transcript(transcript_text)
                self.video_id = None
                return await self._generate_summary()

            if url:
                video_id = extract_video_id(url)
                if not video_id:
                    raise ValueError("Invalid YouTube URL")

            if not video_id:
                raise ValueError(
                    "Must provide either video_id, valid YouTube URL, or transcript_text"
                )

            self.video_id = video_id
            await self._fetch_transcript()

            if self.youtube_api_key:
                await self._fetch_youtube_metadata()

            return await self._generate_summary()
        except Exception as e:
            logger.error(f"Error in get_summary: {str(e)}")
            raise
        finally:
            await self.close()

    async def _generate_summary(self) -> str:
        """Generate summary using either full context or chunked approach"""
        if not self.transcript or not self.transcript.strip():
            raise SummarizerError("No transcript available to summarize")

        # Count tokens in transcript
        transcript_tokens = self.provider.count_tokens(self.transcript)

        # Leave room for prompt and response within context window
        # Use 90% of context window to leave room for prompt and response
        max_input_tokens = int(self.provider.context_window * 0.9)

        if self.use_full_context and transcript_tokens <= max_input_tokens:
            logger.info(
                f"Using full context for summary (transcript: {transcript_tokens} tokens)"
            )
            self.summary = await self._generate_full_summary()
        else:
            if self.use_full_context:
                logger.info(
                    f"Transcript too long for full context ({transcript_tokens} tokens > {max_input_tokens} tokens), "
                    "falling back to chunked approach"
                )
            else:
                logger.info("Using chunked approach for summary")
            self.summary = await self._generate_chunked_summary()

        return self.summary

    async def _generate_chunked_summary(self) -> str:
        """Generate summary using chunked processing with improved error handling"""
        # Determine chunking configuration
        if not self.chunking_config:
            if self.interactive and sys.stdout.isatty():
                # Interactive mode - prompt user for chunking strategy
                self.chunking_config = prompt_for_chunking_strategy(
                    len(self.transcript), self.provider.context_window
                )
            else:
                # Non-interactive mode - use default chunking strategy
                self.chunking_config = get_default_chunking_config(
                    len(self.transcript), self.provider.context_window
                )
            
            logger.info(
                f"Using chunking strategy: {self.chunking_config.strategy.value} "
                f"(size: {self.chunking_config.chunk_size}, "
                f"overlap: {self.chunking_config.chunk_overlap})"
            )

        # Split transcript into chunks using the selected strategy
        chunks = split_text(self.transcript, self.chunking_config)
        logger.info(f"Split transcript into {len(chunks)} chunks")

        # Use provider's rate limits to determine concurrent requests
        max_concurrent = min(
            self.provider.max_concurrent_requests,
            self.provider.rate_limit_config.requests_per_minute // 2,
        )
        semaphore = asyncio.Semaphore(max_concurrent)
        chunk_results: List[Tuple[int, str]] = []

        async def process_chunk(chunk: str, index: int) -> Tuple[int, str]:
            """Process a single chunk with improved error handling"""
            async with semaphore:
                try:
                    # Add jitter to prevent thundering herd
                    jitter = (hash(str(index)) % 1000) / 1000.0
                    await asyncio.sleep(jitter)

                    for attempt in range(self.provider.rate_limit_config.max_retries):
                        try:
                            logger.debug(
                                f"Processing chunk {index + 1}, attempt {attempt + 1}"
                            )
                            prompt = self._create_chunk_prompt(chunk)
                            result = await self.provider.generate_summary(prompt)
                            logger.debug(f"Completed chunk {index + 1}")
                            return (index, result)
                        except Exception as e:
                            if (
                                attempt
                                < self.provider.rate_limit_config.max_retries - 1
                            ):
                                delay = self.provider.rate_limit_config.retry_delay * (
                                    2**attempt
                                )
                                await asyncio.sleep(delay + jitter)
                                logger.warning(
                                    f"Retry {attempt + 1} for chunk {index + 1}: {str(e)}"
                                )
                            else:
                                raise
                except Exception as e:
                    logger.error(
                        f"Failed to process chunk {index + 1} after all retries: {str(e)}"
                    )
                    return (index, f"[Error processing chunk {index + 1}]")

        # Create and track tasks
        tasks = []
        async with self._lock:
            for i, chunk in enumerate(chunks):
                task = asyncio.create_task(process_chunk(chunk, i))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
                tasks.append(task)

        # Gather results with timeout
        try:
            # Set a reasonable timeout based on chunk count
            timeout = 60 + (len(chunks) * 30)  # Base timeout + 30 seconds per chunk
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            chunk_results.extend(results)
        except asyncio.TimeoutError:
            logger.error(f"Summary generation timed out after {timeout} seconds")
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise ProviderError("Summary generation timed out")
        except Exception as e:
            logger.error(f"Error during chunk processing: {str(e)}")
            raise

        # Sort results by original chunk order and filter out errors
        chunk_results.sort(key=lambda x: x[0])
        valid_summaries = [
            result[1] for result in chunk_results if "[Error" not in result[1]
        ]

        if not valid_summaries:
            raise ProviderError("All chunks failed to process")

        if len(valid_summaries) == 1:
            return valid_summaries[0]

        if len(valid_summaries) < len(chunks):
            logger.warning(
                f"Completed with {len(chunks) - len(valid_summaries)} failed chunks"
            )

        # Generate final combined summary
        logger.debug("Generating final combined summary")
        combined_prompt = self._create_combine_prompt("\n\n".join(valid_summaries))
        return await self.provider.generate_summary(combined_prompt)

    async def _generate_full_summary(self) -> str:
        """Generate summary using the full text"""
        prompt = self._create_summary_prompt(self.transcript)
        return await self.provider.generate_summary(prompt)

    async def _fetch_transcript(self) -> None:
        """Fetch and process the video transcript"""
        if not self.video_id:
            raise SummarizerError("No video ID available to fetch transcript")

        try:
            # Create YouTubeTranscriptApi instance with proxy configuration if available
            if self.proxy_config and self.proxy_config.proxy_config:
                logger.debug("Using proxy configuration for transcript fetching")
                ytt_api = YouTubeTranscriptApi(proxy_config=self.proxy_config.proxy_config)
                transcript_list = ytt_api.get_transcript(self.video_id)
            else:
                logger.debug("Using direct connection for transcript fetching")
                transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            
            logger.debug(f"Raw transcript retrieved: {len(transcript_list)} segments")
            self.transcript = " ".join(item["text"] for item in transcript_list)
            self.transcript = self._clean_transcript(self.transcript)
            logger.debug(f"Processed transcript length: {len(self.transcript)} chars")
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}")

            # Provide more specific error messages
            error_msg = str(e).lower()
            if "no element found" in error_msg or "xml" in error_msg:
                raise SummarizerError(
                    f"Failed to fetch transcript for video {self.video_id}. "
                    "This may be due to: 1) The video has no transcripts available, "
                    "2) The video is private/restricted, 3) Invalid video ID, "
                    "4) YouTube API issues, 5) IP blocking (consider using proxy configuration). "
                    "Please verify the video ID and try again."
                )
            elif "could not retrieve" in error_msg or "transcript" in error_msg:
                raise SummarizerError(
                    f"Transcript not available for video {self.video_id}. "
                    "The video may not have subtitles, may be private/restricted, "
                    "or your IP may be blocked (consider using proxy configuration)."
                )
            elif "blocked" in error_msg or "403" in error_msg:
                raise SummarizerError(
                    f"Access to video {self.video_id} is blocked. "
                    "This may be due to IP restrictions. Try using a proxy configuration."
                )
            else:
                raise SummarizerError(f"Error fetching transcript: {str(e)}")

    async def _fetch_youtube_metadata(self) -> None:
        """Fetch metadata for the YouTube video using the YouTube Data API"""
        if not self.video_id or not self.youtube_api_key:
            return

        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    f"https://www.googleapis.com/youtube/v3/videos"
                    f"?id={self.video_id}&key={self.youtube_api_key}"
                    f"&part=snippet,contentDetails,statistics"
                )
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("items"):
                            self.metadata = data["items"][0]
                            logger.debug("Successfully fetched YouTube metadata")
                        else:
                            logger.warning("No metadata found for video")
                    else:
                        logger.warning(
                            f"Failed to fetch metadata: {response.status} {await response.text()}"
                        )
        except Exception as e:
            logger.error(f"Error fetching YouTube metadata: {str(e)}")
            # Non-critical error, continue without metadata

    def _clean_transcript(self, text: str) -> str:
        """Clean and normalize transcript text"""
        # Remove special characters and normalize whitespace
        text = text.replace("\n", " ")
        return " ".join(text.split())

    def _create_summary_prompt(self, text: str) -> str:
        """Create prompt for full text summarization"""
        return (
            "Please provide a clear and concise summary of the following transcript. "
            "Focus on the main points and key insights:\n\n"
            f"{text}\n\n"
            "Summary:"
        )

    def _create_chunk_prompt(self, chunk: str) -> str:
        """Create prompt for chunk summarization"""
        return (
            "Please summarize this section of the transcript, "
            "capturing the key points:\n\n"
            f"{chunk}\n\n"
            "Section Summary:"
        )

    def _create_combine_prompt(self, summaries: str) -> str:
        """Create prompt for combining chunk summaries"""
        return (
            "Below are summaries of different sections. "
            "Please combine them into a single, coherent summary that captures "
            "the main points and flows naturally:\n\n"
            f"{summaries}\n\n"
            "Combined Summary:"
        )

    async def close(self):
        """Close any resources held by the summarizer"""
        if hasattr(self, "provider"):
            await self.provider.close()
        
        # Cancel any active tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        
        self._active_tasks.clear()

    async def export_summary(self, output_path: str) -> None:
        """Export summary and metadata to a file"""
        import json
        
        if not self.summary:
            raise SummarizerError("No summary available to export")
        
        export_data = {
            "summary": self.summary,
            "video_id": self.video_id,
            "metadata": self.metadata,
            "provider": self.provider_name,
            "model": self.provider.model,
        }
        
        try:
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Summary exported to {output_path}")
        except Exception as e:
            raise SummarizerError(f"Failed to export summary: {str(e)}")
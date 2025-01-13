import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from youtube_transcript_api import YouTubeTranscriptApi

from ..utils.url_parser import extract_video_id
from .providers.base import ProviderError
from .providers.cerebras import CerebrasProvider
from .providers.groq import GroqProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """Base exception for Summarizer errors"""

    pass


class Summarizer:
    """Main class for generating summaries from YouTube video transcripts or direct text input"""

    PROVIDERS = {
        "openai": OpenAIProvider,
        "groq": GroqProvider,
        "cerebras": CerebrasProvider,
        "ollama": OllamaProvider,
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        chunk_size: int = 3000,
        chunk_overlap: int = 200,
        use_full_context: bool = False,
        youtube_api_key: Optional[str] = None,
    ):
        self.provider_name = provider.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_full_context = use_full_context
        self.youtube_api_key = youtube_api_key
        self._lock = asyncio.Lock()
        self._active_tasks: set[asyncio.Task] = set()

        # Initialize provider
        provider_class = self.PROVIDERS.get(self.provider_name)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider_class(
            model=model,
            temperature=temperature,
            use_full_context=use_full_context,
            api_key=self._get_provider_api_key(self.provider_name),
        )

        # State variables
        self.video_id: Optional[str] = None
        self.transcript: Optional[str] = None
        self.summary: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def _get_provider_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "ollama": None,  # Local provider doesn't need API key
        }
        env_var = env_vars.get(provider)
        return os.environ.get(env_var) if env_var else None

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
                await self._fetch_metadata()

            return await self._generate_summary()
        except Exception as e:
            logger.error(f"Error in get_summary: {str(e)}")
            raise
        finally:
            await self.close()

    async def _generate_summary(self) -> str:
        """Generate summary using either full context or chunked approach"""
        if not self.transcript:
            raise SummarizerError("No transcript available to summarize")

        if (
            self.use_full_context
            and len(self.transcript) <= self.provider.context_window
        ):
            logger.info("Using full context for summary")
            self.summary = await self._generate_full_summary()
        else:
            logger.info("Using chunked approach for summary")
            self.summary = await self._generate_chunked_summary()

        return self.summary

    async def _generate_chunked_summary(self) -> str:
        """Generate summary using chunked processing with improved error handling"""
        chunks = self._split_into_chunks(
            self.transcript, self.chunk_size, self.chunk_overlap
        )
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

        # Create and track tasks with improved management
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

    # async def _generate_chunked_summary(self) -> str:
    #     """Generate summary using chunked processing"""
    #     chunks = self._split_into_chunks(
    #         self.transcript, self.chunk_size, self.chunk_overlap
    #     )
    #     logger.info(f"Split transcript into {len(chunks)} chunks")

    #     # Use provider's recommended concurrency
    #     semaphore = asyncio.Semaphore(self.provider.max_concurrent_requests)
    #     chunk_results: List[Tuple[int, str]] = []
    #     errors: List[str] = []

    #     async def process_chunk(chunk: str, index: int) -> Tuple[int, str]:
    #         """Process a single chunk with error handling"""
    #         async with semaphore:
    #             try:
    #                 result = await self._process_chunk_with_backoff(chunk, index)
    #                 return (index, result)
    #             except Exception as e:
    #                 logger.error(f"Error processing chunk {index}: {str(e)}")
    #                 raise

    #     # Create processing tasks
    #     tasks = []
    #     async with self._lock:
    #         for i, chunk in enumerate(chunks):
    #             task = asyncio.create_task(process_chunk(chunk, i))
    #             self._active_tasks.add(task)
    #             task.add_done_callback(self._active_tasks.discard)
    #             tasks.append(task)

    #     # Process chunks with improved error handling
    #     for completed_task in asyncio.as_completed(tasks):
    #         try:
    #             result = await completed_task
    #             chunk_results.append(result)
    #         except Exception as e:
    #             errors.append(str(e))

    #     if not chunk_results:
    #         raise SummarizerError("All chunks failed to process")

    #     if errors:
    #         logger.warning(f"Completed with {len(errors)} chunk errors")

    #     # Sort results by original chunk order
    #     chunk_results.sort(key=lambda x: x[0])
    #     summaries = [result[1] for result in chunk_results]

    #     if len(summaries) == 1:
    #         return summaries[0]

    #     logger.info("Generating final combined summary")
    #     combined_prompt = self._create_combine_prompt("\n\n".join(summaries))
    #     return await self.provider.generate_summary(combined_prompt)

    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks with smart sentence boundary detection"""
        if not text:
            return []

        # Common sentence endings including ellipsis
        sentence_endings = ".!?..."
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate the ideal end point
            ideal_end = min(start + chunk_size, text_length)

            # If we're not at the text end, look for a good breaking point
            if ideal_end < text_length:
                # First try to break at a sentence boundary within a window
                window_size = min(200, chunk_size // 10)  # Look back up to 200 chars
                window_start = max(ideal_end - window_size, start)

                # Find the last sentence boundary in the window
                found_boundary = False
                for i in range(ideal_end, window_start - 1, -1):
                    if i < text_length and text[i - 1] in sentence_endings:
                        chunk = text[start:i].strip()
                        if chunk:  # Ensure we don't add empty chunks
                            chunks.append(chunk)
                            logger.debug(
                                f"Created chunk {len(chunks)}: {len(chunk)} chars"
                            )
                        start = max(i - overlap, 0)
                        found_boundary = True
                        break

                # If no sentence boundary found, break at a space
                if not found_boundary:
                    window_text = text[window_start:ideal_end]
                    last_space = window_text.rfind(" ")
                    if last_space != -1:
                        break_point = window_start + last_space
                        chunk = text[start:break_point].strip()
                        if chunk:
                            chunks.append(chunk)
                            logger.debug(
                                f"Created chunk {len(chunks)}: {len(chunk)} chars"
                            )
                        start = max(break_point - overlap, 0)
                    else:
                        # If no space found, break at ideal_end
                        chunk = text[start:ideal_end].strip()
                        if chunk:
                            chunks.append(chunk)
                            logger.debug(
                                f"Created chunk {len(chunks)}: {len(chunk)} chars"
                            )
                        start = max(ideal_end - overlap, 0)
            else:
                # Add the final chunk
                final_chunk = text[start:].strip()
                if final_chunk:
                    chunks.append(final_chunk)
                    logger.debug(f"Created final chunk: {len(final_chunk)} chars")
                break

        logger.info(f"Successfully created {len(chunks)} chunks")
        return chunks

    async def _generate_full_summary(self) -> str:
        """Generate summary using the full text"""
        prompt = self._create_summary_prompt(self.transcript)
        return await self.provider.generate_summary(prompt)

    async def _fetch_transcript(self) -> None:
        """Fetch and process the video transcript"""
        if not self.video_id:
            raise SummarizerError("No video ID available to fetch transcript")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            logger.debug(f"Raw transcript retrieved: {len(transcript_list)} segments")
            self.transcript = " ".join(item["text"] for item in transcript_list)
            self.transcript = self._clean_transcript(self.transcript)
            logger.debug(f"Processed transcript length: {len(self.transcript)} chars")
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}")
            raise SummarizerError(f"Error fetching transcript: {str(e)}")

    async def _fetch_metadata(self) -> None:
        """Fetch video metadata if YouTube API key is available"""
        if not self.youtube_api_key or not self.video_id:
            return
        # TODO: Implement YouTube API metadata fetching
        pass

    async def _process_chunk_with_backoff(
        self, chunk: str, index: int, max_retries: int = 3
    ) -> str:
        """Process a chunk with exponential backoff retry"""
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing chunk {index + 1}, attempt {attempt + 1}")
                prompt = self._create_chunk_prompt(chunk)
                summary = await self.provider.generate_summary(prompt)
                logger.debug(f"Completed chunk {index + 1}")
                return summary
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    jitter = (hash(str(index) + str(attempt)) % 1000) / 1000.0
                    await asyncio.sleep(delay + jitter)
                    logger.warning(
                        f"Retry {attempt + 1} for chunk {index + 1}: {str(e)}"
                    )
                else:
                    raise

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
        """Cleanup resources and active tasks"""
        async with self._lock:
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        if hasattr(self.provider, "close"):
            await self.provider.close()

    def export_summary(self, file_path: str) -> None:
        """Export the summary and metadata to a file"""
        if not self.summary:
            raise SummarizerError("No summary available to export")

        data = {
            "video_id": self.video_id,  # Will be None for direct transcript input
            "transcript": self.transcript,
            "summary": self.summary,
            "metadata": self.metadata,
            "provider": self.provider_name,
            "model": self.provider.model,
            "settings": {
                "use_full_context": self.use_full_context,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        }

        import json

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


# import asyncio
# import logging
# import os
# from typing import Any, Dict, Optional

# from youtube_transcript_api import YouTubeTranscriptApi

# from ..utils.url_parser import extract_video_id
# from .providers.cerebras import CerebrasProvider
# from .providers.groq import GroqProvider
# from .providers.ollama import OllamaProvider
# from .providers.openai import OpenAIProvider

# logger = logging.getLogger(__name__)


# class SummarizerError(Exception):
#     """Base exception for Summarizer errors"""

#     pass


# class Summarizer:
#     """Main class for generating summaries from YouTube video transcripts or direct text input"""

#     # Provider mapping
#     PROVIDERS = {
#         "openai": OpenAIProvider,
#         "groq": GroqProvider,
#         "cerebras": CerebrasProvider,
#         "ollama": OllamaProvider,
#     }

#     def __init__(
#         self,
#         provider: str = "openai",
#         model: Optional[str] = None,
#         temperature: float = 0.7,
#         chunk_size: int = 4000,
#         chunk_overlap: int = 200,
#         use_full_context: bool = False,
#         youtube_api_key: Optional[str] = None,
#     ):
#         """Initialize the summarizer with the specified provider and settings.

#         Args:
#             provider: Name of the LLM provider to use ("openai", "groq", "cerebras", "ollama")
#             model: Model identifier to use. Provider-specific options:
#                   - cerebras: "llama3.1-8b", "llama3.1-70b", "llama-3.3-70b"
#                   - openai: "gpt-3.5-turbo", "gpt-4o", "gpt-4o-32k", "gpt-4o-turbo-preview"
#                   - groq: [your supported models]
#                   - ollama: [your supported models]
#             temperature: Temperature for generation (0.0 to 1.0)
#             chunk_size: Maximum size of text chunks for processing
#             chunk_overlap: Number of tokens to overlap between chunks
#             use_full_context: Whether to use the model's full context window
#             youtube_api_key: Optional API key for fetching video metadata
#         """
#         self.provider_name = provider.lower()
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.use_full_context = use_full_context
#         self.youtube_api_key = youtube_api_key
#         self._lock = asyncio.Lock()
#         self._active_tasks: set[asyncio.Task] = set()

#         # Initialize the provider
#         if self.provider_name not in self.PROVIDERS:
#             raise ValueError(
#                 f"Unsupported provider. Choose from: {', '.join(self.PROVIDERS.keys())}"
#             )

#         provider_class = self.PROVIDERS[self.provider_name]

#         self.provider = provider_class(
#             model=model,
#             temperature=temperature,
#             use_full_context=use_full_context,
#             api_key=self._get_provider_api_key(self.provider_name),
#         )

#         # State variables
#         self.video_id: Optional[str] = None
#         self.transcript: Optional[str] = None
#         self.summary: Optional[str] = None
#         self.metadata: Dict[str, Any] = {}

#     async def __aenter__(self):
#         """Async context manager entry"""
#         return self

#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         """Async context manager exit with cleanup"""
#         try:
#             # Cancel any active tasks
#             async with self._lock:
#                 for task in self._active_tasks:
#                     if not task.done():
#                         task.cancel()
#                 await asyncio.gather(*self._active_tasks, return_exceptions=True)

#             # Close the provider
#             if hasattr(self.provider, "close"):
#                 await self.provider.close()
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")

#     def _get_provider_api_key(self, provider: str) -> Optional[str]:
#         """Get API key from environment variables"""
#         env_vars = {
#             "openai": "OPENAI_API_KEY",
#             "groq": "GROQ_API_KEY",
#             "cerebras": "CEREBRAS_API_KEY",
#         }
#         return os.environ.get(env_vars.get(provider))

#     async def get_summary(
#         self,
#         video_id: Optional[str] = None,
#         url: Optional[str] = None,
#         transcript_text: Optional[str] = None,
#     ) -> str:
#         """Generate a summary from either a YouTube video or direct transcript input."""
#         try:
#             if transcript_text is not None:
#                 logging.info("Using direct transcript input")
#                 self.transcript = self._clean_transcript(transcript_text)
#                 self.video_id = None
#                 return await self._generate_summary()

#             if url:
#                 logging.info("Using YouTube video URL input")
#                 video_id = extract_video_id(url)
#                 if not video_id:
#                     raise ValueError("Invalid YouTube URL")

#             if not video_id:
#                 raise ValueError(
#                     "Must provide either video_id, valid YouTube URL, or transcript_text"
#                 )

#             self.video_id = video_id
#             await self._fetch_transcript()

#             if self.youtube_api_key:
#                 await self._fetch_metadata()

#             logging.info("Starting summary generation")
#             return await self._generate_summary()
#         except Exception as e:
#             logger.error(f"Error in get_summary: {str(e)}")
#             raise
#         finally:
#             # Ensure cleanup happens even without context manager
#             if hasattr(self.provider, "close"):
#                 await self.provider.close()

#     async def _process_chunk_with_backoff(
#         self, chunk: str, index: int, max_retries: int = 3
#     ) -> str:
#         """Process a chunk with exponential backoff retry"""
#         for attempt in range(max_retries):
#             try:
#                 logger.info(f"Processing chunk {index + 1}, attempt {attempt + 1}")
#                 prompt = self._create_chunk_prompt(chunk)
#                 summary = await self.provider.generate_summary(prompt)
#                 logger.debug(f"Completed chunk {index + 1}")
#                 return summary
#             except Exception as e:
#                 if "rate limit" in str(e).lower() and attempt < max_retries - 1:
#                     wait_time = (2**attempt) * 1.0  # Exponential backoff
#                     logger.warning(
#                         f"Rate limit hit on chunk {index + 1}. Waiting {wait_time}s"
#                     )
#                     await asyncio.sleep(wait_time)
#                 else:
#                     raise

#     async def _fetch_transcript(self) -> None:
#         """Fetch and process the video transcript"""
#         if not self.video_id:
#             raise SummarizerError("No video ID available to fetch transcript")

#         try:
#             transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
#             logger.debug(f"Raw transcript retrieved: {len(transcript_list)} segments")
#             self.transcript = " ".join(item["text"] for item in transcript_list)
#             self.transcript = self._clean_transcript(self.transcript)
#             logger.debug(f"Processed transcript length: {len(self.transcript)} chars")
#         except Exception as e:
#             logger.error(f"Error fetching transcript: {str(e)}")
#             raise SummarizerError(f"Error fetching transcript: {str(e)}")

#     async def _fetch_metadata(self) -> None:
#         """Fetch video metadata if YouTube API key is available"""
#         if not self.youtube_api_key:
#             return

#         # TODO: Implement YouTube API metadata fetching
#         pass

#     async def _generate_full_summary(self) -> str:
#         """Generate summary using the full transcript"""
#         prompt = self._create_summary_prompt(self.transcript)
#         return await self.provider.generate_summary(prompt)

#     async def _generate_summary(self) -> str:
#         """Generate summary from the current transcript"""
#         if not self.transcript:
#             raise SummarizerError("No transcript available to summarize")

#         logger.debug(
#             f"Checking context length. Full context mode: {self.use_full_context}"
#         )
#         if self.use_full_context and self.provider.can_use_full_context(
#             self.transcript
#         ):
#             logger.info("Using full context for summary")
#             self.summary = await self._generate_full_summary()
#         else:
#             logger.info("Using chunked approach for summary")
#             token_count = self.provider.count_tokens(self.transcript)
#             logger.debug(f"Total tokens in transcript: {token_count}")
#             self.summary = await self._generate_chunked_summary()

#         return self.summary

#     async def _generate_chunked_summary(self) -> str:
#         """Generate summary by chunking the transcript with concurrent processing"""
#         chunks = self._split_into_chunks(
#             self.transcript, self.chunk_size, self.chunk_overlap
#         )
#         logger.info(f"Split transcript into {len(chunks)} chunks")

#         # Process chunks with backoff and rate limiting
#         semaphore = asyncio.Semaphore(10)

#         async def process_with_semaphore(chunk: str, index: int) -> str:
#             async with semaphore:
#                 return await self._process_chunk_with_backoff(chunk, index)

#         # Create and track tasks
#         tasks = []
#         async with self._lock:
#             for i, chunk in enumerate(chunks):
#                 task = asyncio.create_task(process_with_semaphore(chunk, i))
#                 self._active_tasks.add(task)
#                 task.add_done_callback(self._active_tasks.discard)
#                 tasks.append(task)

#         # Wait for all tasks to complete
#         chunk_summaries = await asyncio.gather(*tasks)

#         # If only one chunk, return its summary
#         if len(chunk_summaries) == 1:
#             return chunk_summaries[0]

#         # Generate final summary from chunk summaries
#         logger.info("Generating final combined summary")
#         combined_prompt = self._create_combine_prompt("\n\n".join(chunk_summaries))
#         return await self.provider.generate_summary(combined_prompt)

#     def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> list[str]:
#         """Split text into chunks based on character count with overlap"""
#         logger.debug(f"Starting chunk split with size={chunk_size}, overlap={overlap}")

#         if not text:
#             return []

#         chunks = []
#         start = 0
#         text_length = len(text)

#         while start < text_length:
#             # Get the chunk, ensuring we don't exceed text length
#             end = min(start + chunk_size, text_length)
#             chunk = text[start:end]

#             # If this isn't the last chunk, try to break at a space
#             if end < text_length:
#                 # Look for the last space within the chunk
#                 last_space = chunk.rfind(" ")
#                 if last_space != -1:
#                     end = start + last_space
#                     chunk = text[start:end]

#             chunks.append(chunk)
#             logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} chars")

#             # Move start position, accounting for overlap
#             start = end - overlap if end < text_length else text_length

#         logger.info(f"Successfully created {len(chunks)} chunks")
#         return chunks

#     def _clean_transcript(self, text: str) -> str:
#         """Clean and normalize transcript text"""
#         # Remove special characters and normalize whitespace
#         text = text.replace("\n", " ")
#         text = " ".join(text.split())
#         return text

#     def _create_summary_prompt(self, text: str) -> str:
#         """Create prompt for full text summarization"""
#         return (
#             "Please provide a clear and concise summary of the following transcript. "
#             "Focus on the main points and key insights:\n\n"
#             f"{text}\n\n"
#             "Summary:"
#         )

#     def _create_chunk_prompt(self, chunk: str) -> str:
#         """Create prompt for chunk summarization"""
#         return (
#             "Please summarize this section of the transcript, "
#             "capturing the key points:\n\n"
#             f"{chunk}\n\n"
#             "Section Summary:"
#         )

#     def _create_combine_prompt(self, summaries: str) -> str:
#         """Create prompt for combining chunk summaries"""
#         return (
#             "Below are summaries of different sections. "
#             "Please combine them into a single, coherent summary that captures "
#             "the main points and flows naturally:\n\n"
#             f"{summaries}\n\n"
#             "Combined Summary:"
#         )

#     def export_summary(self, file_path: str) -> None:
#         """Export the summary and metadata to a file"""
#         if not self.summary:
#             raise SummarizerError("No summary available to export")

#         data = {
#             "video_id": self.video_id,  # Will be None for direct transcript input
#             "summary": self.summary,
#             "metadata": self.metadata,
#             "provider": self.provider_name,
#             "model": self.provider.model,
#             "settings": {
#                 "use_full_context": self.use_full_context,
#                 "chunk_size": self.chunk_size,
#                 "chunk_overlap": self.chunk_overlap,
#             },
#         }

#         import json

#         with open(file_path, "w") as f:
#             json.dump(data, f, indent=2)

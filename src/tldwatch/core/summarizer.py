import os
from typing import Any, Dict, Optional

from youtube_transcript_api import YouTubeTranscriptApi

from .providers.cerebras import CerebrasProvider
from .providers.groq import GroqProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAIProvider
from .utils.url_parser import extract_video_id


class SummarizerError(Exception):
    """Base exception for Summarizer errors"""

    pass


class Summarizer:
    """Main class for generating summaries from YouTube video transcripts"""

    # Provider mapping
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
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        use_full_context: bool = False,
        youtube_api_key: Optional[str] = None,
    ):
        """
        Initialize the summarizer with the specified provider and settings.

        Args:
            provider: Name of the LLM provider to use
            model: Model identifier (if None, use provider's default)
            temperature: Temperature for generation (0.0 to 1.0)
            chunk_size: Maximum size of text chunks for processing
            chunk_overlap: Number of tokens to overlap between chunks
            use_full_context: Whether to use the model's full context window
            youtube_api_key: Optional API key for fetching video metadata
        """
        self.provider_name = provider.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_full_context = use_full_context
        self.youtube_api_key = youtube_api_key

        # Initialize the provider
        if self.provider_name not in self.PROVIDERS:
            raise ValueError(
                f"Unsupported provider. Choose from: {', '.join(self.PROVIDERS.keys())}"
            )

        provider_class = self.PROVIDERS[self.provider_name]
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
        }
        return os.environ.get(env_vars.get(provider))

    async def get_summary(
        self, video_id: Optional[str] = None, url: Optional[str] = None
    ) -> str:
        """
        Generate a summary for the specified YouTube video.

        Args:
            video_id: YouTube video ID
            url: YouTube video URL (alternative to video_id)

        Returns:
            Generated summary text
        """
        # Get video ID from URL if provided
        if url:
            video_id = extract_video_id(url)

        if not video_id:
            raise ValueError("Must provide either video_id or valid YouTube URL")

        self.video_id = video_id

        # Fetch transcript and metadata
        await self._fetch_transcript()
        if self.youtube_api_key:
            await self._fetch_metadata()

        # Generate summary
        if self.use_full_context and self.provider.can_use_full_context(
            self.transcript
        ):
            self.summary = await self._generate_full_summary()
        else:
            self.summary = await self._generate_chunked_summary()

        return self.summary

    async def _fetch_transcript(self) -> None:
        """Fetch and process the video transcript"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            self.transcript = " ".join(item["text"] for item in transcript_list)
            self.transcript = self._clean_transcript(self.transcript)
        except Exception as e:
            raise SummarizerError(f"Error fetching transcript: {str(e)}")

    async def _fetch_metadata(self) -> None:
        """Fetch video metadata if YouTube API key is available"""
        if not self.youtube_api_key:
            return

        # TODO: Implement YouTube API metadata fetching
        pass

    async def _generate_full_summary(self) -> str:
        """Generate summary using the full transcript"""
        prompt = self._create_summary_prompt(self.transcript)
        return await self.provider.generate_summary(prompt)

    async def _generate_chunked_summary(self) -> str:
        """Generate summary by chunking the transcript"""
        # Split transcript into chunks
        chunks = self._split_into_chunks(
            self.transcript, self.chunk_size, self.chunk_overlap
        )

        # Generate summaries for each chunk
        chunk_summaries = []
        for chunk in chunks:
            prompt = self._create_chunk_prompt(chunk)
            summary = await self.provider.generate_summary(prompt)
            chunk_summaries.append(summary)

        # Combine chunk summaries
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Generate final summary from chunk summaries
        combined_prompt = self._create_combine_prompt("\n\n".join(chunk_summaries))
        return await self.provider.generate_summary(combined_prompt)

    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            # Find chunk end considering token count
            chunk = words[start : start + chunk_size]
            while len(" ".join(chunk)) > chunk_size:
                chunk.pop()

            chunks.append(" ".join(chunk))
            start += len(chunk) - overlap

        return chunks

    def _clean_transcript(self, text: str) -> str:
        """Clean and normalize transcript text"""
        # Remove special characters and normalize whitespace
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    def _create_summary_prompt(self, text: str) -> str:
        """Create prompt for full text summarization"""
        return (
            "Please provide a clear and concise summary of the following video transcript. "
            "Focus on the main points and key insights:\n\n"
            f"{text}\n\n"
            "Summary:"
        )

    def _create_chunk_prompt(self, chunk: str) -> str:
        """Create prompt for chunk summarization"""
        return (
            "Please summarize this section of the video transcript, "
            "capturing the key points:\n\n"
            f"{chunk}\n\n"
            "Section Summary:"
        )

    def _create_combine_prompt(self, summaries: str) -> str:
        """Create prompt for combining chunk summaries"""
        return (
            "Below are summaries of different sections of a video. "
            "Please combine them into a single, coherent summary that captures "
            "the main points and flows naturally:\n\n"
            f"{summaries}\n\n"
            "Combined Summary:"
        )

    def export_summary(self, file_path: str) -> None:
        """Export the summary and metadata to a file"""
        if not self.summary:
            raise SummarizerError("No summary available to export")

        data = {
            "video_id": self.video_id,
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

"""
Text chunking strategies for processing long transcripts.

This module provides different strategies for splitting text into manageable chunks
for processing by LLM models, with options for different transcript lengths and use cases.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Enumeration of available chunking strategies"""

    STANDARD = "standard"  # Default strategy with balanced chunk size
    SMALL = "small"  # Smaller chunks for more granular processing
    LARGE = "large"  # Larger chunks for more context
    ADAPTIVE = "adaptive"  # Adapts chunk size based on transcript length
    PARAGRAPH = "paragraph"  # Chunks based on paragraph boundaries
    SEMANTIC = "semantic"  # Chunks based on semantic boundaries (requires additional processing)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""

    chunk_size: int
    chunk_overlap: int
    strategy: ChunkingStrategy
    min_chunk_size: int = 500  # Minimum chunk size to avoid tiny chunks
    max_chunk_size: int = 8000  # Maximum chunk size to avoid context window issues


def get_default_chunking_config(
    text_length: int, context_window: int
) -> ChunkingConfig:
    """
    Get a default chunking configuration based on text length and model context window.

    Args:
        text_length: Length of the text in characters
        context_window: Model's context window size in tokens

    Returns:
        ChunkingConfig with appropriate settings
    """
    # Estimate tokens (rough approximation: 4 chars ≈ 1 token)
    estimated_tokens = text_length // 4

    # Determine appropriate strategy based on text length
    if estimated_tokens <= context_window * 0.8:
        # Text fits in context window with room for prompt and response
        return ChunkingConfig(
            chunk_size=text_length,
            chunk_overlap=0,
            strategy=ChunkingStrategy.STANDARD,
        )
    elif text_length < 20000:  # Short transcript (~5 minutes)
        return ChunkingConfig(
            chunk_size=2000,
            chunk_overlap=200,
            strategy=ChunkingStrategy.STANDARD,
        )
    elif text_length < 60000:  # Medium transcript (~15 minutes)
        return ChunkingConfig(
            chunk_size=4000,
            chunk_overlap=400,
            strategy=ChunkingStrategy.STANDARD,
        )
    else:  # Long transcript
        return ChunkingConfig(
            chunk_size=6000,
            chunk_overlap=600,
            strategy=ChunkingStrategy.STANDARD,
        )


def prompt_for_chunking_strategy(text_length: int, context_window: int) -> ChunkingConfig:
    """
    Prompt the user to select a chunking strategy based on transcript length.

    Args:
        text_length: Length of the text in characters
        context_window: Model's context window size in tokens

    Returns:
        ChunkingConfig with user-selected settings
    """
    # Get the default config as a starting point
    default_config = get_default_chunking_config(text_length, context_window)
    
    # Estimate transcript duration (rough approximation: 1000 chars ≈ 1 minute of speech)
    estimated_minutes = text_length // 1000
    
    print("\nTranscript Information:")
    print(f"- Approximate length: {estimated_minutes} minutes")
    print(f"- Character count: {text_length}")
    
    print("\nRecommended chunking strategy:")
    print(f"- Strategy: {default_config.strategy.value}")
    print(f"- Chunk size: {default_config.chunk_size} characters")
    print(f"- Chunk overlap: {default_config.chunk_overlap} characters")
    
    use_default = input("\nUse recommended settings? (Y/n): ").strip().lower() != "n"
    
    if use_default:
        return default_config
    
    # Let user select a strategy
    print("\nSelect chunking strategy:")
    for i, strategy in enumerate(ChunkingStrategy):
        print(f"{i+1}. {strategy.value} - ", end="")
        if strategy == ChunkingStrategy.STANDARD:
            print("Balanced approach (recommended for most transcripts)")
        elif strategy == ChunkingStrategy.SMALL:
            print("Smaller chunks for more detailed processing")
        elif strategy == ChunkingStrategy.LARGE:
            print("Larger chunks for better context preservation")
        elif strategy == ChunkingStrategy.ADAPTIVE:
            print("Automatically adjusts chunk size based on content")
        elif strategy == ChunkingStrategy.PARAGRAPH:
            print("Splits by paragraphs or natural breaks")
        elif strategy == ChunkingStrategy.SEMANTIC:
            print("Splits by semantic meaning (may be slower)")
    
    strategy_choice = 1  # Default to standard
    try:
        choice = input("Enter choice (1-6) [1]: ").strip()
        if choice:
            strategy_choice = int(choice)
            if not 1 <= strategy_choice <= 6:
                print("Invalid choice, using standard strategy")
                strategy_choice = 1
    except ValueError:
        print("Invalid input, using standard strategy")
    
    selected_strategy = list(ChunkingStrategy)[strategy_choice - 1]
    
    # Let user customize chunk size and overlap
    try:
        chunk_size_input = input(f"Chunk size ({default_config.chunk_size}): ").strip()
        chunk_size = int(chunk_size_input) if chunk_size_input else default_config.chunk_size
        
        chunk_overlap_input = input(f"Chunk overlap ({default_config.chunk_overlap}): ").strip()
        chunk_overlap = int(chunk_overlap_input) if chunk_overlap_input else default_config.chunk_overlap
    except ValueError:
        print("Invalid input, using default values")
        chunk_size = default_config.chunk_size
        chunk_overlap = default_config.chunk_overlap
    
    return ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=selected_strategy,
    )


def split_text(text: str, config: ChunkingConfig) -> List[str]:
    """
    Split text into chunks based on the specified strategy.

    Args:
        text: The text to split
        config: Chunking configuration

    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Select the appropriate chunking function based on strategy
    chunking_functions = {
        ChunkingStrategy.STANDARD: _split_standard,
        ChunkingStrategy.SMALL: _split_small,
        ChunkingStrategy.LARGE: _split_large,
        ChunkingStrategy.ADAPTIVE: _split_adaptive,
        ChunkingStrategy.PARAGRAPH: _split_paragraph,
        ChunkingStrategy.SEMANTIC: _split_semantic,
    }
    
    chunking_function = chunking_functions.get(config.strategy, _split_standard)
    
    # Apply the selected chunking function
    return chunking_function(text, config)


def _split_standard(text: str, config: ChunkingConfig) -> List[str]:
    """
    Standard chunking strategy with sentence boundary detection.
    
    This is the default strategy that tries to split at sentence boundaries.
    """
    return _split_with_sentence_boundaries(
        text, config.chunk_size, config.chunk_overlap, config.min_chunk_size
    )


def _split_small(text: str, config: ChunkingConfig) -> List[str]:
    """
    Small chunking strategy for more granular processing.
    
    Uses smaller chunks with less overlap.
    """
    # Use 2/3 of the standard chunk size
    adjusted_size = max(config.min_chunk_size, config.chunk_size // 3 * 2)
    adjusted_overlap = max(50, config.chunk_overlap // 2)
    
    return _split_with_sentence_boundaries(
        text, adjusted_size, adjusted_overlap, config.min_chunk_size
    )


def _split_large(text: str, config: ChunkingConfig) -> List[str]:
    """
    Large chunking strategy for better context preservation.
    
    Uses larger chunks with more overlap.
    """
    # Use 4/3 of the standard chunk size, but cap at max_chunk_size
    adjusted_size = min(config.max_chunk_size, config.chunk_size * 4 // 3)
    adjusted_overlap = min(config.chunk_size // 3, config.chunk_overlap * 2)
    
    return _split_with_sentence_boundaries(
        text, adjusted_size, adjusted_overlap, config.min_chunk_size
    )


def _split_adaptive(text: str, config: ChunkingConfig) -> List[str]:
    """
    Adaptive chunking strategy that adjusts based on text length.
    
    For shorter texts, uses larger chunks with more overlap.
    For longer texts, uses smaller chunks with less overlap.
    """
    text_length = len(text)
    
    if text_length < 20000:  # Short text
        adjusted_size = min(config.max_chunk_size, config.chunk_size * 3 // 2)
        adjusted_overlap = config.chunk_overlap * 2
    elif text_length < 60000:  # Medium text
        adjusted_size = config.chunk_size
        adjusted_overlap = config.chunk_overlap
    else:  # Long text
        adjusted_size = max(config.min_chunk_size, config.chunk_size * 2 // 3)
        adjusted_overlap = max(50, config.chunk_overlap // 2)
    
    return _split_with_sentence_boundaries(
        text, adjusted_size, adjusted_overlap, config.min_chunk_size
    )


def _split_paragraph(text: str, config: ChunkingConfig) -> List[str]:
    """
    Paragraph-based chunking strategy.
    
    Tries to split at paragraph boundaries, then combines paragraphs to reach
    the target chunk size.
    """
    # Split by paragraphs (double newlines or multiple line breaks)
    paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph)
        
        # If adding this paragraph would exceed the chunk size and we already have content,
        # finalize the current chunk and start a new one
        if current_size + paragraph_size > config.chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap by including some paragraphs from the previous chunk
            overlap_size = 0
            overlap_paragraphs = []
            
            # Add paragraphs from the end of the previous chunk until we reach the desired overlap
            for p in reversed(current_chunk):
                if overlap_size + len(p) <= config.chunk_overlap:
                    overlap_paragraphs.insert(0, p)
                    overlap_size += len(p)
                else:
                    break
            
            current_chunk = overlap_paragraphs
            current_size = overlap_size
        
        current_chunk.append(paragraph)
        current_size += paragraph_size
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def _split_semantic(text: str, config: ChunkingConfig) -> List[str]:
    """
    Semantic chunking strategy.
    
    This is a placeholder for a more sophisticated semantic chunking approach.
    Currently falls back to standard chunking.
    """
    # For now, this is just a placeholder that falls back to standard chunking
    # In a real implementation, this would use NLP techniques to identify semantic boundaries
    logger.warning("Semantic chunking not fully implemented, falling back to standard chunking")
    return _split_standard(text, config)


def _split_with_sentence_boundaries(
    text: str, chunk_size: int, overlap: int, min_chunk_size: int
) -> List[str]:
    """
    Split text into chunks with smart sentence boundary detection.
    
    This is a helper function used by multiple chunking strategies.
    """
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
                    if chunk and len(chunk) >= min_chunk_size:  # Ensure we don't add tiny chunks
                        chunks.append(chunk)
                        logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} chars")
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
                    if chunk and len(chunk) >= min_chunk_size:
                        chunks.append(chunk)
                        logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} chars")
                    start = max(break_point - overlap, 0)
                else:
                    # If no space found, break at ideal_end
                    chunk = text[start:ideal_end].strip()
                    if chunk and len(chunk) >= min_chunk_size:
                        chunks.append(chunk)
                        logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} chars")
                    start = max(ideal_end - overlap, 0)
        else:
            # Add the final chunk
            final_chunk = text[start:].strip()
            if final_chunk and len(final_chunk) >= min_chunk_size:
                chunks.append(final_chunk)
                logger.debug(f"Created final chunk: {len(final_chunk)} chars")
            break

    logger.info(f"Successfully created {len(chunks)} chunks")
    return chunks
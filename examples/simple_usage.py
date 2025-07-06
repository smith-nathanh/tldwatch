#!/usr/bin/env python3
"""
Simple usage examples for the new unified tldwatch interface.

This demonstrates the greatly simplified API that consolidates all providers
into a single, easy-to-use interface.
"""

import asyncio
import os
from tldwatch import SimpleSummarizer, summarize_video, ChunkingStrategy


async def main():
    """Demonstrate the simplified tldwatch interface"""
    
    # Example 1: Quick summarization with convenience function
    print("=== Example 1: Quick summarization ===")
    try:
        # Just provide a YouTube URL or video ID - that's it!
        summary = await summarize_video("dQw4w9WgXcQ")  # Rick Roll video ID
        print(f"Summary: {summary[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using the SimpleSummarizer class with options
    print("=== Example 2: Using SimpleSummarizer with options ===")
    
    summarizer = SimpleSummarizer()
    
    try:
        # You can specify provider, model, and chunking strategy
        summary = await summarizer.summarize(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            provider="openai",  # or "anthropic", "google", "groq", etc.
            model="gpt-4o-mini",  # optional - uses provider default if not specified
            chunking_strategy="standard",  # "none", "standard", "small", "large"
            temperature=0.7
        )
        print(f"Summary: {summary[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Different providers and chunking strategies
    print("=== Example 3: Different providers and strategies ===")
    
    # List available providers
    print("Available providers:", summarizer.list_providers())
    print("Available chunking strategies:", summarizer.list_chunking_strategies())
    
    # Try different combinations
    test_cases = [
        {"provider": "openai", "chunking_strategy": "none"},  # Submit entire transcript
        {"provider": "anthropic", "chunking_strategy": "small"},  # Small chunks
        {"provider": "google", "chunking_strategy": "large"},  # Large chunks
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {case}")
        try:
            # For demo purposes, using a short video ID
            summary = await summarizer.summarize("dQw4w9WgXcQ", **case)
            print(f"Result: {summary[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Direct text summarization
    print("=== Example 4: Direct text summarization ===")
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines that mimic 
    "cognitive" functions that humans associate with the human mind, such as 
    "learning" and "problem solving". As machines become increasingly capable, 
    tasks considered to require "intelligence" are often removed from the 
    definition of AI, a phenomenon known as the AI effect. A quip in Tesler's 
    Theorem says "AI is whatever hasn't been done yet."
    """
    
    try:
        summary = await summarizer.summarize(
            sample_text,
            provider="openai",
            chunking_strategy="none"  # No chunking needed for short text
        )
        print(f"Summary of direct text: {summary}")
    except Exception as e:
        print(f"Error: {e}")


def show_environment_setup():
    """Show what environment variables are needed"""
    print("=== Environment Setup ===")
    print("Set these environment variables for the providers you want to use:")
    print("- OPENAI_API_KEY (for OpenAI)")
    print("- ANTHROPIC_API_KEY (for Anthropic)")
    print("- GEMINI_API_KEY (for Google)")
    print("- GROQ_API_KEY (for Groq)")
    print("- DEEPSEEK_API_KEY (for DeepSeek)")
    print("- CEREBRAS_API_KEY (for Cerebras)")
    print("- No API key needed for Ollama (local)")
    print()
    
    # Check which keys are set
    keys_to_check = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", 
        "GROQ_API_KEY", "DEEPSEEK_API_KEY", "CEREBRAS_API_KEY"
    ]
    
    print("Currently set API keys:")
    for key in keys_to_check:
        status = "✓" if os.getenv(key) else "✗"
        print(f"  {status} {key}")
    print()


if __name__ == "__main__":
    show_environment_setup()
    
    # Run the examples
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
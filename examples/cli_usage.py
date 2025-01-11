"""
Example usage of the tldwatch CLI.

This file provides examples of different ways to use the tldwatch command-line interface.
All examples assume you have tldwatch installed and the necessary API keys set in your environment.

Basic Usage:
-----------
# Get summary using YouTube URL
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc

# Get summary using video ID directly
tldwatch --video-id QAgR4uQ15rc

# Read YouTube URL from stdin
echo "https://www.youtube.com/watch?v=QAgR4uQ15rc" | tldwatch --stdin

Output Options:
--------------
# Save summary to a text file
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc --out summary.txt

# Save summary as JSON
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc --out summary.json

# Process multiple videos and save summaries
for video in QAgR4uQ15rc QkGwxtALTLU; do
    tldwatch --video-id $video --out "summaries/${video}.txt"
done

Provider Selection:
-----------------
# Use Groq with specific model
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc \
    --provider groq \
    --model mixtral-8x7b-32768

# Use OpenAI with GPT-4
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc \
    --provider openai \
    --model gpt-4

# Use local Ollama model
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc \
    --provider ollama \
    --model mistral

Advanced Options:
---------------
# Use full context window
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc --full-context

# Adjust chunk size and temperature
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc \
    --chunk-size 6000 \
    --temperature 0.8

Configuration Management:
----------------------
# Save current settings as default configuration
tldwatch --provider groq --model mixtral-8x7b-32768 --save-config

# Save configuration without processing a video
tldwatch --save-config --provider openai --model gpt-4

Batch Processing:
---------------
#!/bin/bash
# Example script for batch processing videos

# Process list of video IDs
video_ids=(
    "QAgR4uQ15rc"
    "QkGwxtALTLU"
    "another_video_id"
)

output_dir="summaries"
mkdir -p "$output_dir"

for video_id in "${video_ids[@]}"; do
    echo "Processing video: $video_id"
    tldwatch --video-id "$video_id" --out "$output_dir/$video_id.txt"
done

Pipeline Usage:
-------------
# Process a list of URLs from a file
cat video_urls.txt | while read url; do
    tldwatch "$url" --out "summaries/$(date +%Y%m%d_%H%M%S).txt"
done

Environment Setup:
----------------
# Example of setting up environment variables
export OPENAI_API_KEY="your-api-key"
export GROQ_API_KEY="your-api-key"
export CEREBRAS_API_KEY="your-api-key"
export YOUTUBE_API_KEY="your-api-key"  # Optional, for video metadata

Error Handling:
-------------
# Redirect stderr to a log file
tldwatch https://www.youtube.com/watch?v=QAgR4uQ15rc 2> error.log

# Continue batch processing even if some videos fail
for video in QAgR4uQ15rc QkGwxtALTLU; do
    tldwatch --video-id $video --out "summaries/${video}.txt" || \
        echo "Failed to process $video" >> failed.log
done
"""

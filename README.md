# YouTube Transcript Summarizer

This repository provides a module to summarize YouTube video transcripts using various AI models including OpenAI, Cerebras, and Ollama. It retrieves transcripts from YouTube videos, processes them, and generates concise summaries.

## Features

- **Transcript Retrieval**: Fetches transcripts from YouTube videos using the `YouTubeTranscriptApi`.
- **Text Processing**: Cleans and splits the transcript text into manageable chunks for processing.
- **Multiple Model Support**: 
  - OpenAI models (GPT-3.5, GPT-4)
  - Cerebras models (llama3.1-8b, llama3.1-70b)
  - Local Ollama models (Llama, Mistral, etc.)
- **Output**: Saves the original transcript and the generated summary to specified directories.
- **Twitter Integration**: Posts the generated summary as a thread on Twitter.

## File Structure

- **main.py**: Entry point to summarize YouTube video transcripts and post to Twitter.
- **get_transcript.py**: Module to retrieve transcripts from YouTube videos.
- **summarizer.py**: Module to process and summarize transcripts.
- **tests/x_api_test.py**: Unit tests and X api test.
- **requirements.txt**: List of dependencies required for the project.
- **README.md**: This file.
- **prompt.json**: JSON file containing prompts for summarization.
- **schedule/schedule.json**: JSON file containing the schedule for video summarization.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

To summarize a YouTube video transcript run the following command (option to not post a thread to t)

```sh
python main.py --channel <channel_name> --video_id <video_id> --title <video_title> --model <model_name>
```

### Options

- `--channel`: The channel you are pulling from.
- `--video_id`: The video ID of the YouTube video.
- `--title`: The title to insert into the final text.
- `--model`: The model to use for the completion (default: "gpt-4o").
- `--prompt`: The prompt to use for the completion (default: "prompt.json").
- `--temperature`: Temperature parameter of the model (default: 0.3).
- `--chunk_size`: The maximum number of tokens to send to the model at once (default: 4000).
- `--post`: Generate and post thread to Twitter.
- `--verbose` or `-v`: Enable verbose output.

### Example

```sh
python main.py --channel "Berkeley RDI Center on Decentralization & AI" --video_id "-yf-e-9FvOc" --title "CS 194/294-196 (LLM Agents) - Lecture 7, Nicolas Chapados and Alexandre Drouin"
```

## Configuration

Ensure you have your OpenAI API key and Twitter API keys set in a `.env` file:

Create a `.env` file in the root of your project and add the following lines (langchain keys and endpoints are optional):

```sh
OPENAI_API_KEY="your key"
CEREBRAS_API_KEY="your key"
LANGCHAIN_API_KEY="your key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="yt-transcripts"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
X_API_KEY="your key"
X_API_KEY_SECRET="your key"
X_ACCESS_TOKEN="your token"
X_ACCESS_TOKEN_SECRET="your token"
```
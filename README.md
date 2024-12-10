# YouTube Transcript Summarizer

This repository provides a module to summarize YouTube video transcripts using the OpenAI API. It retrieves transcripts from YouTube videos, processes them, and generates concise summaries.

## Features

- **Transcript Retrieval**: Fetches transcripts from YouTube videos using the `YouTubeTranscriptApi`.
- **Text Processing**: Cleans and splits the transcript text into manageable chunks for processing.
- **Summarization**: Utilizes OpenAI's language models to generate summaries of the transcripts.
- **Output**: Saves the original transcript and the generated summary to specified directories.

## File Structure

- **summarize.py**: Main script to fetch transcripts and generate summaries.
- **get_transcript.py**: Module to retrieve transcripts from YouTube videos.
- **utils.py**: Utility functions for processing and summarizing text.
- **requirements.txt**: List of dependencies required for the project.
- **README.md**: This file.

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

To summarize a YouTube video transcript, run the following command:

```sh
python summarize.py --channel <channel_name> --video_id <video_id>
```

## Example

```sh
python summarize.py --channel patel --video_id "UakqL6Pj9xo"
```

## Configuration

Ensure you have your OpenAI API key set in a `.env` file:

Create a `.env` file in the root of your project and add the following lines (langchain keys and endpoints are optional):

```sh
OPENAI_API_KEY="your key"
LANGCHAIN_API_KEY="your key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="yt-transcripts"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

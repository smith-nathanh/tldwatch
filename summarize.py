import argparse
import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
from utils import clean_transcript_string, save_response
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", required=True, help="The channel you are pulling from.")
    parser.add_argument("--video_id", help="The video ID of the YouTube video.")
    parser.add_argument("--title", default='', help="The title to insert into the final text")
    parser.add_argument('--model', default="gpt-4o", help="The model to use for the completion")
    parser.add_argument('--prompt', default="prompt.json", help="The prompt to use for the completion")
    parser.add_argument('--temperature', default=0.3, type=float, help="Temperature parameter of the model")
    parser.add_argument('--chunk_size', default=4000, type=int, help="The maximum number of tokens to send to the model at once")
    parser.add_argument("--create_thread", action="store_true", help="Create a twitter thread from the summary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    video_id = args.video_id

    llm = ChatOpenAI(
        temperature=args.temperature, 
        model_name=args.model,
        verbose=args.verbose
    )

    # Get transcript
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([line['text'] for line in raw_transcript])
    transcript = clean_transcript_string(transcript)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(transcript)

    # Load prompts
    with open(args.prompt, 'r') as f:
        prompts = json.load(f)

    map_template = prompts.get("map_prompt", 
        """Write a concise summary of this video transcript section:
        {text}
        CONCISE SUMMARY:""")

    combine_template = prompts.get("combine_prompt", 
        """Below are summaries from different sections of the same video. 
        Create a single coherent summary that captures the key points (avoid saying the word 'delves' during your summarization):

        {text}

        FINAL SUMMARY:""")

    # Process chunks using updated invoke method
    summaries = []
    for chunk in chunks:
        response = llm.invoke(map_template.format(text=chunk))
        # Extract content from AIMessage
        if hasattr(response, 'content'):
            summaries.append(response.content)
        else:
            summaries.append(str(response))

    # Combine summaries
    final_summary = llm.invoke(combine_template.format(
        text="\n".join(summaries)
    ))
    
    # Extract final summary content
    final_text = final_summary.content if hasattr(final_summary, 'content') else str(final_summary)
    
    # save response
    save_response(args, transcript, final_text)

    # add title to final text
    final_text = f"{args.title}\n\n" + final_text if args.title else final_text
    if args.verbose:
        print('\n', final_text, '\n')

if __name__ == "__main__":
    load_dotenv() # load environment variables from .env file
    main()
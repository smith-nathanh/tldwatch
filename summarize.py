import argparse
import os
from youtube_transcript_api import YouTubeTranscriptApi
from utils import clean_transcript_string, save_response
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Dictionary mapping model names to their maximum context length in tokens
MODEL_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    'gpt-4-turbo-2024-04-09': 128000
}

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=os.environ['LANGCHAIN_API_KEY']
LANGCHAIN_PROJECT="pr-plaintive-scow-13"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", "-p", required=True, help="The channel you are pulling from.")
    parser.add_argument("--video_id", help="The video ID of the YouTube video.")
    parser.add_argument('--model', default="gpt-3.5-turbo-16k", help="The model to use for the completion")
    parser.add_argument('--temperature', default=0.3, help="Temperature parameter of the model")
    parser.add_argument("--prompt", "-t", default="prompt.json", help="The JSON file containing the prompt template")
    parser.add_argument("--outdir", "-o", required=False, help="The directory to save the transcript and the summary as json.")
    args = parser.parse_args()

    # get command line arguments
    video_id = args.video_id
    outdir = os.path.join(os.getcwd(), f'texts/{args.channel}') if args.outdir is None else args.outdir

    # make the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Define LLM
    llm = ChatOpenAI(temperature=args.temperature, model_name=args.model)

    # get the transcript
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([line['text'] for line in raw_transcript])
    transcript = clean_transcript_string(transcript)

    # split the text up into chunks if necessary
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MODEL_TOKEN_LIMITS[args.model] - 500,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False
        )
    texts = text_splitter.create_documents([transcript])

    # generate summary
    chain = load_summarize_chain(llm, chain_type="refine")
    result = chain.invoke(texts)

    # print and save response
    print('\n', result["output_text"], '\n')
    save_response(transcript, result["output_text"], outdir, args)


if __name__ == "__main__":
   main()
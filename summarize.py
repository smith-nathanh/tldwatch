import argparse
import json
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", "-p", required=True, help="The channel you are pulling from.")
    parser.add_argument("--video_id", help="The video ID of the YouTube video.")
    parser.add_argument('--model', default="gpt-3.5-turbo", help="The model to use for the completion")
    parser.add_argument("--prompt", "-t", default="prompt.json", help="The JSON file containing the prompt template")
    parser.add_argument("--outdir", "-o", required=False, help="The directory to save the transcript and the summary as json.")
    args = parser.parse_args()

    # get command line arguments
    video_id = args.video_id
    outdir = os.path.join(os.getcwd(), f'texts/{args.channel}') if args.outdir is None else args.outdir

    # make the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # get the transcript
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([line['text'] for line in raw_transcript])

    # Load the prompt template
    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Prompt template file {args.prompt} not found!")
    with open(args.prompt, "r") as f:
        prompt = json.load(f)
    
    # update the model if necesssary and add the transcript to the prompt
    prompt["model"] = args.model
    prompt["messages"].append({"role": "user", "content": f"Please summarize the main points in the following: {transcript}"})
    
    # Send the input data as a prompt to the OpenAI API and get the response
    client = OpenAI(api_key=os.getenv("OPENAI_LI_KEY"))
    response = client.chat.completions.create(**prompt)

    # Save the response to a json file
    output = {'transcript': transcript, 'summary': response.choices[0].message.content}
    print('\n', output['summary'], '\n')
    output_file = os.path.join(outdir, f"{args.video_id}.json")
    with open(output_file, "w", encoding="utf-8") as f: 
        json.dump(output, f)

    print(f"Response written to {output_file}\n")

if __name__ == "__main__":
   main()
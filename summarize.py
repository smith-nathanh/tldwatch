import argparse
import json
import os
import re
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import tiktoken

def clean_string(text):
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Remove any escape sequences like \'
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Normalize multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

# Dictionary mapping model names to their maximum context length in tokens
# is it actually cheaper to use the smaller models? need to look into this
MODEL_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    #"gpt-4-32k": 32768
}

# List of models ordered by their maximum context length
ORDERED_MODELS = sorted(MODEL_TOKEN_LIMITS.items(), key=lambda item: item[1])

# Function to calculate the number of tokens in a message
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Every message has a minimum of 4 tokens
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # If there's a name, it counts for an extra token
                num_tokens -= 1
    num_tokens += 2  # Every reply also has 2 tokens
    return num_tokens

# Function to truncate the last user message to fit within the maximum context length
def truncate_last_user_message(messages, max_tokens, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = num_tokens_from_messages(messages, model=model)
    
    while total_tokens > max_tokens:
        last_message = messages[-1]
        if last_message["role"] == "user":
            original_content = last_message["content"]
            encoded_content = encoding.encode(original_content)
            tokens_to_remove = total_tokens - max_tokens
            truncated_content = encoding.decode(encoded_content[:-tokens_to_remove])
            last_message["content"] = truncated_content
            total_tokens = num_tokens_from_messages(messages, model=model)
            if len(truncated_content) == 0:
                raise ValueError("User message cannot be truncated further without losing all content.")
        else:
            raise ValueError("The last message is not from the user; unable to truncate further.")
    return messages

# Function to dynamically choose the appropriate model and trim messages
def choose_model_and_trim_messages(messages, response_tokens):
    for model, max_tokens in ORDERED_MODELS:
        if num_tokens_from_messages(messages, model=model) <= max_tokens:
            return model, messages

    # If no model fits, use the largest model and truncate the last user message
    largest_model, largest_max_tokens = ORDERED_MODELS[-1]
    truncated_messages = truncate_last_user_message(messages, max_tokens=largest_max_tokens - response_tokens, model=largest_model)
    return largest_model, truncated_messages


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
    transcript = clean_string(transcript)

    # Load the prompt template
    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Prompt template file {args.prompt} not found!")
    with open(args.prompt, "r") as f:
        prompt = json.load(f)
    
    # update the model if necesssary and add the transcript to the prompt
    prompt["model"] = args.model
    prompt["messages"].append({"role": "user", "content": f"Please summarize the main points in the following: {transcript}"})
    
    # Dynamically choose the model and trim messages
    chosen_model, trimmed_messages = choose_model_and_trim_messages(prompt["messages"], response_tokens=prompt["max_tokens"])
    prompt['model'] = chosen_model
    prompt['messages'] = trimmed_messages
    
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
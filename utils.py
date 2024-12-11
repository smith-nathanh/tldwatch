import re
import os
import json

def clean_transcript_string(text):
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


def retrieve_prompt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template file {path} not found!")
    with open(path, "r") as f:
        prompt = json.load(f)
    return prompt

def save_response(args, transcript, summary):
    
    # Sanitize channel name for filesystem
    safe_channel = (args.channel
                   .replace(' ', '_')
                   .replace('/', '_')
                   .replace('&', 'and'))
    
    # Create output directory path
    outdir = os.path.join(os.getcwd(), 'texts', safe_channel)
    os.makedirs(outdir, exist_ok=True)

    output = {'args': vars(args), 
              'transcript': transcript, 
              'summary': summary}

    output_file = os.path.join(outdir, f"{args.video_id}.json")
    with open(output_file, "w", encoding="utf-8") as f: 
        json.dump(output, f)
    print(f"Response written to {output_file}\n")
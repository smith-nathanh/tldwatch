import argparse
import json
import os
from openai import OpenAI



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", "-p", required=True, help="The channel you are pulling from.")
    parser.add_argument("--video_id", help="The video ID of the YouTube video.")
    parser.add_argument("--output_file", "-o", required=False, help="The output file to save the summary.")
    parser.add_argument('--model', default="gpt-3.5-turbo", help="The model to use for the completion")
    parser.add_argument("--write_to_txt", action="store_true", help="Write the response to a text file")
    args = parser.parse_args()

    full_path = os.path.join(os.getcwd(), f'texts/{args.channel}/{args.video_id}.txt')
    output_file = args.output_file if args.output_file else os.path.join(os.getcwd(), f'texts/{args.channel}/{args.video_id}_summary.txt')
    
    # Load the text from the input file
    with open(full_path, "r") as f:
        text = f.read()

    prompt = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a super insightful and helpful assistant who has a keen eye for detail. You are helping a user summarize a text."
            },
            {
                "role": "user",
                "content": f"Please summarize this main points of the text {text}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1
    }

    # Send the input data as a prompt to the OpenAI API and get the response
    client = OpenAI(api_key=os.getenv("OPENAI_LI_KEY"))
    response = client.chat.completions.create(**prompt)
    output = {'response': response.choices[0].message.content}
    
    # Write the response to the output JSON file
    # with open(output_file, "w") as f:
    #     json.dump(output, f)    
    # print("Response generated successfully!")

    #if args.write_to_txt:
    print()
    print(output['response'])
    print()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output['response'])
    print(f"Response written to {output_file}")

if __name__ == "__main__":
   main()
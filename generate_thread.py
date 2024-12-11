import json
import argparse
import textwrap
from pathlib import Path

def format_first_paragraph(title, channel):
    return (f"AI Research Highlights ✨ Distilling AI content into focused summaries "
            f"you can read in minutes. Today's video: {title} by {channel}\n"
            "Full summary in the 🧵 below 👇")

def create_thread_paragraphs(summary, title=None, channel=None, video_id=None):
    paragraphs = []
    
    if title and channel:
        paragraphs.append(format_first_paragraph(title, channel))
    
    # Split summary into sentences and group into paragraphs
    current_paragraph = ""
    for sentence in summary.split('. '):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        test_paragraph = current_paragraph + '. ' + sentence if current_paragraph else sentence
        if len(test_paragraph) <= 128:
            current_paragraph = test_paragraph
        else:
            if current_paragraph:
                paragraphs.append(current_paragraph + '.')
            current_paragraph = sentence
    
    if current_paragraph:
        paragraphs.append(current_paragraph + '...')
    
    paragraphs.append(f"🔗 Watch the full video here: https://www.youtube.com/watch?v={video_id}")
        
    return paragraphs

def main():
    parser = argparse.ArgumentParser(description='Format JSON summary into thread paragraphs')
    parser.add_argument('json_file', help='Path to JSON file containing transcript and summary')
    args = parser.parse_args()
    
    # Read JSON file
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
            summary = data.get('summary')
            if not summary:
                raise ValueError("JSON file must contain a 'summary' field")
            
            json_args = data.get('args')
            if not args:
                raise ValueError("JSON file must contain an 'args' key")

            title = json_args.get('title')
            channel = json_args.get('channel')
            video_id = json_args.get('video_id')

            if not all([title, channel, video_id]):
                raise ValueError("'args' must contain 'title', 'channel', and 'video_id' fields")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
    
    # Create thread paragraphs
    paragraphs = create_thread_paragraphs(summary, title, channel, video_id)
    
    # Print result
    print("\nFormatted Thread:\n")
    for p in paragraphs:
        print(p + "\n")
    
    # Save to file if output path provided
    try:
        output_file = args.json_file.replace('.json', '.txt')
        with open(output_file, 'w') as f:
            f.write('\n\n'.join(paragraphs))
        print(f"\nSaved to {output_file}")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    main()
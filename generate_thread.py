import json
import argparse

def retrieve_summary(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            summary = data.get('summary')
            if not summary:
                raise ValueError("JSON file must contain a 'summary' field")
            
            json_args = data.get('args')
            if not json_args:
                raise ValueError("JSON file must contain an 'args' key")

            title = json_args.get('title')
            channel = json_args.get('channel')
            video_id = json_args.get('video_id')

            if not all([title, channel, video_id]):
                raise ValueError("'args' must contain 'title', 'channel', and 'video_id' fields")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
    return summary, title, channel, video_id


def format_first_paragraph(title, channel):
    return (f"AI Research Highlights ✨ Distilling AI content into focused summaries "
            f"you can read in minutes. Today's video: {title} by {channel}\n"
            "Full summary in the 🧵 below 👇")

def create_thread_paragraphs(summary, title=None, channel=None, video_id=None, verbose=True):
    paragraphs = []
    
    if title and channel:
        paragraphs.append(format_first_paragraph(title, channel))
    
    # Clean the summary by removing newlines and extra spaces
    summary = ' '.join(summary.split())
    sentences = [s.strip() for s in summary.split('. ') if s.strip()]
    
    current_paragraph = []
    current_length = 0
    
    for sentence in sentences:
        # Calculate new length including the sentence and potential period/space
        new_length = current_length + len(sentence) + (2 if current_paragraph else 0)
        
        if new_length <= 275:  # Leave room for ellipsis
            current_paragraph.append(sentence)
            current_length = new_length
        else:
            if current_paragraph:
                # Join completed paragraph
                paragraph_text = '. '.join(current_paragraph) + '.'
                if verbose:
                    print(f"Paragraph length: {len(paragraph_text)}")
                paragraphs.append(paragraph_text)
            
            # Start new paragraph with current sentence
            current_paragraph = [sentence]
            current_length = len(sentence)
    
    # Add remaining paragraph if any
    if current_paragraph:
        paragraph_text = '. '.join(current_paragraph) + '..'
        if verbose:
            print(f"Final paragraph length: {len(paragraph_text)}")
        paragraphs.append(paragraph_text)
    
    # Add video link
    video_link = f"🔗 Watch the full video here: https://www.youtube.com/watch?v={video_id}"
    paragraphs.append(video_link)
    
    return paragraphs

def save_thread(output_file, paragraphs):
    try:
        with open(output_file, 'w') as f:
            f.write('\n\n'.join(paragraphs))
        print(f"\nSaved to {output_file}")
    except Exception as e:
        print(f"Error saving output: {e}")

def main():
    parser = argparse.ArgumentParser(description='Format JSON summary into thread paragraphs')
    parser.add_argument('json_file', help='Path to JSON file containing transcript and summary')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print paragraphs')
    args = parser.parse_args()
    
    # Retrieve the json file containing the summary
    summary, title, channel, video_id = retrieve_summary(args.json_file)
    
    # Create thread paragraphs
    paragraphs = create_thread_paragraphs(summary, title, channel, video_id, args.verbose)
    
    # Print result
    if args.verbose:
        print("\nFormatted Thread:\n")
        for p in paragraphs:
            print(p + "\n")
    
    # Save to file if output path provided
    output_file = args.json_file.replace('.json', '_thread.txt')
    save_thread(output_file, paragraphs)

if __name__ == "__main__":
    main()
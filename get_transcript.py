from youtube_transcript_api import YouTubeTranscriptApi
import argparse
import os



def main():
    """
    Entry point to get the transcript of a YouTube video using its video ID.
    """
    parser = argparse.ArgumentParser(description="Get the transcript of a YouTube video using its video ID.")
    parser.add_argument("--channel", "-p", required=True, help="The channel you are pulling from.")
    parser.add_argument("--video_id", help="The video ID of the YouTube video.")
    parser.add_argument("--output_file", "-o", required=False, help="The output file to save the transcript to.")
    args = parser.parse_args()
    
    # get command line arguments
    video_id = args.video_id
    full_path = os.path.join(os.getcwd(), f'texts/{args.channel}')
    os.makedirs(full_path, exist_ok=True)
    output_file = args.output_file if args.output_file else os.path.join(full_path, f"{video_id}.txt")

    # get the transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # save the transcript to a file
    text = ' '.join([line['text'] for line in transcript])
    with open(output_file, 'w') as f:
        f.write(text)
    print("Transcript saved to", output_file)


if __name__ == "__main__":
    main()
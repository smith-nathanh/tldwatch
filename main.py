import os
import logging
import json
from datetime import datetime
import time
import tweepy
import argparse
from dotenv import load_dotenv
from summarizer import TranscriptSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_twitter_api():
    
    api_key = os.getenv('X_API_KEY')
    api_key_secret = os.getenv('X_API_KEY_SECRET')
    access_token = os.getenv('X_ACCESS_TOKEN')
    access_token_secret = os.getenv('X_ACCESS_TOKEN_SECRET')
    
    return tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_key_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )

def post_thread(content_list, client):
    previous_tweet_id = None
    for content in content_list:
        try:
            response = client.create_tweet(
                text=content,
                in_reply_to_tweet_id=previous_tweet_id
            )
            previous_tweet_id = response.data['id']
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error posting tweet: {e}")
            break

def main():
    """Entry point to summarize YouTube video transcripts"""
    parser = argparse.ArgumentParser(description='Summarize YouTube video transcripts and post to Twitter.')
    parser.add_argument('--channel', help='The channel you are pulling from.')
    parser.add_argument('--video_id', help='The video ID of the YouTube video.')
    parser.add_argument('--title', help='The title to insert into the final text')
    parser.add_argument('--model', default="gpt-4o", help='The model to use for the completion')
    parser.add_argument('--prompt', default="prompt.json", help='The prompt to use for the completion')
    parser.add_argument('--temperature', default=0.3, type=float, help='Temperature parameter of the model')
    parser.add_argument('--chunk_size', default=4000, type=int, help='The maximum number of tokens to send to the model at once')
    parser.add_argument('--do_not_post', action='store_true', help='Do not post the thread to Twitter')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    if args.channel and args.video_id and args.title:
        channel = args.channel
        video_id = args.video_id
        title = args.title
    else:
        try:
            with open('schedule/schedule.json', 'r') as f:
                schedule = json.load(f)
        except FileNotFoundError:
            logging.error("schedule.json not found")
            return

        today = datetime.now().strftime('%Y-%m-%d')
        
        if today not in schedule:
            logging.error("No entry found for today")
            return

        today_entry = schedule[today]
        channel = today_entry['channel']
        video_id = today_entry['video_id']
        title = today_entry['title']

    model = args.model
    prompt = args.prompt
    temperature = args.temperature
    chunk_size = args.chunk_size
    verbose = args.verbose

    summarizer = TranscriptSummarizer(
        channel=channel,
        video_id=video_id,
        title=title,
        model=model,
        prompt=prompt,
        temperature=temperature,
        chunk_size=chunk_size,
        verbose=verbose
    )
    
    summarizer.fetch_transcript()
    summarizer.summarize()
    summarizer.save_summary()
    summarizer.generate_thread()

    # Load Twitter API client
    twitter_client = load_twitter_api()

    # Read the generated thread from the file
    thread_file = summarizer.output_file.replace('.json', '_thread.txt')
    with open(thread_file, 'r') as f:
        thread_content = f.read().split('\n\n')

    # Post the thread to Twitter
    if not args.do_not_post:
        post_thread(thread_content, twitter_client)

if __name__ == "__main__":
    load_dotenv()
    main()
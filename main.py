import os
import time
import json
from datetime import datetime
import logging
import argparse
import tweepy
from dotenv import load_dotenv
from summarizer import TranscriptSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_twitter_api():
    api_key = os.getenv('X_API_KEY')
    api_key_secret = os.getenv('X_API_KEY_SECRET')
    access_token = os.getenv('X_ACCESS_TOKEN')
    access_token_secret = os.getenv('X_ACCESS_TOKEN_SECRET')
    
    client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_key_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        wait_on_rate_limit=True
    )
    return client

def post_thread(content_list, client):
    previous_tweet_id = None
    
    for i, content in enumerate(content_list):
        retries = 3  # Reduced retries since wait_on_rate_limit handles most rate limit issues
        retry_count = 0
        
        while retry_count < retries:
            try:
                # Post the tweet - Tweepy will automatically wait if we hit rate limits
                response = client.create_tweet(
                    text=content,
                    in_reply_to_tweet_id=previous_tweet_id
                )
                previous_tweet_id = response.data['id']
                logging.info(f"Posted tweet {i+1}/{len(content_list)}")
                
                # Small delay between tweets for safety
                time.sleep(2)
                break
                
            except Exception as e:
                retry_count += 1
                if not isinstance(e, tweepy.errors.TooManyRequests):
                    logging.error(f"Error posting tweet {i+1}: {str(e)}")
                    if retry_count == retries:
                        logging.error(f"Failed to post tweet after {retries} attempts. Continuing with thread...")
                    time.sleep(5)  # Basic delay before retry for non-rate-limit errors

def main():
    """Entry point to summarize YouTube video transcripts"""
    parser = argparse.ArgumentParser(description='Summarize YouTube video transcripts and post to Twitter.')
    parser.add_argument('--channel', help='The channel you are pulling from.')
    parser.add_argument('--video_id', help='The video ID of the YouTube video.')
    parser.add_argument('--title', help='The title to insert into the final text')
    parser.add_argument('--model', default="gpt-4o", help='The model to use for the completion')
    parser.add_argument('--provider', default="openai", help='The provider of the model')
    parser.add_argument('--prompt', default="prompt.json", help='The prompt to use for the completion')
    parser.add_argument('--temperature', default=0.3, type=float, help='Temperature parameter of the model')
    parser.add_argument('--chunk_size', default=4000, type=int, help='The maximum number of tokens to send to the model at once')
    parser.add_argument('--post', action='store_true', help='Generate and post thread to Twitter')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    if args.channel and args.video_id and args.title:
        today_entry = {}
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

    summarizer = TranscriptSummarizer(
        channel=today_entry.get('channel', args.channel),
        video_id=today_entry.get('video_id', args.video_id),
        title=today_entry.get('title', args.title),
        model=today_entry.get('model', args.model),
        provider=today_entry.get('provider', args.provider),
        prompt=today_entry.get('prompt', args.prompt),
        temperature=today_entry.get('temperature', args.temperature),
        chunk_size=today_entry.get('chunk_size', args.chunk_size),
        verbose=today_entry.get('verbose', args.verbose)
    )

    summarizer.fetch_transcript()
    summarizer.summarize()
    summarizer.save_summary()

    # Only generate and post thread if --post flag is present
    if args.post:
        summarizer.generate_thread()
        thread_file = summarizer.output_file.replace('.json', '_thread.txt')
        with open(thread_file, 'r') as f:
            thread_content = f.read().split('\n\n')
        
        # Load Twitter API client and post
        twitter_client = load_twitter_api()
        post_thread(thread_content, twitter_client)

if __name__ == "__main__":
    load_dotenv()
    main()
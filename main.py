import os
import json
from datetime import datetime
import subprocess
from pathlib import Path
import tweepy
import time
from dotenv import load_dotenv

def load_twitter_api():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get X (Twitter) API credentials from environment variables
    api_key = os.getenv('X_API_KEY')  # This is the consumer key
    api_key_secret = os.getenv('X_API_KEY_SECRET')  # This is the consumer secret
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
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"Error posting tweet: {e}")
            break

def main():
    # Read schedule.json
    try:
        with open('schedule.json', 'r') as f:
            schedule = json.load(f)
    except FileNotFoundError:
        print("schedule.json not found")
        return

    # Get today's date in the format used in schedule.json
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if today's entry exists
    if today not in schedule:
        print("No entry found for today")
        return

    today_entry = schedule[today]
    
    # Run summarize.py
    cmd = [
        "summarize.py",
        "--channel", today_entry['channel'],
        "--video_id", today_entry['video_id'],
        "--title", today_entry['title'],
        "--create_thread", "-v"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running summarize.py: {e}")
        return

    # Read the generated thread file
    thread_file = Path(f"{today_entry['video_id']}_thread.txt")
    if not thread_file.exists():
        print("Thread file not generated")
        return

    with open(thread_file, 'r') as f:
        content = f.read()
    
    # Split into paragraphs and filter empty lines
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Initialize Twitter API and post thread
    twitter_client = load_twitter_api()
    post_thread(paragraphs, twitter_client)

if __name__ == "__main__":
    main()
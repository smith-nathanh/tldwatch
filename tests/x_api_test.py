import os
import time
import tweepy
from dotenv import load_dotenv

def load_twitter_api():
    load_dotenv()
    
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

def test_post_and_delete_tweet():
    client = load_twitter_api()
    
    # Post a tweet
    tweet_text = "This is a test tweet."
    response = client.create_tweet(text=tweet_text)
    tweet_id = response.data['id']
    print(f"Tweet posted with ID: {tweet_id}")
    
    # Wait for a few seconds
    time.sleep(60)
    
    # Delete the tweet
    client.delete_tweet(tweet_id)
    print(f"Tweet with ID: {tweet_id} deleted")

if __name__ == "__main__":
    test_post_and_delete_tweet()
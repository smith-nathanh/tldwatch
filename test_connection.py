from dotenv import load_dotenv
import os
import tweepy

load_dotenv()
api_key = os.getenv('X_API_KEY')
api_key_secret = os.getenv('X_API_KEY_SECRET')
access_token = os.getenv('X_ACCESS_TOKEN')
access_token_secret = os.getenv('X_ACCESS_TOKEN_SECRET')

client = tweepy.Client(
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

print(client.rate_limit_status())
import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import tweepy
from dotenv import load_dotenv

from tldwatch import Summarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_twitter_api():
    api_key = os.getenv("X_API_KEY")
    api_key_secret = os.getenv("X_API_KEY_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")

    client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_key_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        wait_on_rate_limit=True,
    )
    return client


def post_thread(content_list, client):
    previous_tweet_id = None

    if len(content_list) > 17:
        logging.warning(
            "Thread contains more than 17 tweets. Twitter thread limit is 17 tweets."
        )
        return

    for i, content in enumerate(content_list):
        retries = 3
        retry_count = 0

        while retry_count < retries:
            try:
                response = client.create_tweet(
                    text=content, in_reply_to_tweet_id=previous_tweet_id
                )
                previous_tweet_id = response.data["id"]
                logging.info(f"Posted tweet {i + 1}/{len(content_list)}")
                time.sleep(2)
                break

            except Exception as e:
                retry_count += 1
                if not isinstance(e, tweepy.errors.TooManyRequests):
                    logging.error(f"Error posting tweet {i + 1}: {str(e)}")
                    if retry_count == retries:
                        logging.error(
                            f"Failed to post tweet after {retries} attempts. Continuing with thread..."
                        )
                    time.sleep(5)


def format_first_paragraph(title, channel):
    return (
        f"AI Research Highlights ✨ Distilling AI content into focused summaries "
        f"you can read in minutes. Today's video: {title} by {channel}\n"
        "Full summary in the 🧵 below 👇"
    )


def create_thread_paragraphs(
    summary,
    title=None,
    channel=None,
    video_id=None,
    provider=None,
    model=None,
    verbose=True,
):
    paragraphs = []

    if title and channel:
        paragraphs.append(format_first_paragraph(title, channel))

    summary = " ".join(summary.split())
    sentences = [s.strip() for s in summary.split(". ") if s.strip()]

    current_paragraph = []
    current_length = 0

    for sentence in sentences:
        new_length = current_length + len(sentence) + (2 if current_paragraph else 0)

        if new_length <= 275:
            current_paragraph.append(sentence)
            current_length = new_length
        else:
            if current_paragraph:
                paragraph_text = ". ".join(current_paragraph) + "."
                if verbose:
                    logging.info(f"Paragraph length: {len(paragraph_text)}")
                paragraphs.append(paragraph_text)

            current_paragraph = [sentence]
            current_length = len(sentence)

    if current_paragraph:
        paragraph_text = ". ".join(current_paragraph) + "."
        if verbose:
            logging.info(f"Final paragraph length: {len(paragraph_text)}")
        paragraphs.append(paragraph_text)

    paragraphs.append(
        f"Summary powered by {provider} and {model}.\n"
        f"🔗 Watch the full video here: https://www.youtube.com/watch?v={video_id}"
    )

    return paragraphs


def save_thread(output_file, paragraphs):
    try:
        with open(output_file, "w") as f:
            f.write("\n\n".join(paragraphs))
        logging.info(f"Thread saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving thread: {e}")


def display_thread_preview(paragraphs):
    print("\n=== Thread Preview ===\n")
    for i, paragraph in enumerate(paragraphs, 1):
        print(f"Tweet {i}/{len(paragraphs)}:")
        print(f"{paragraph}")
        print(f"Length: {len(paragraph)} characters")
        print("-" * 50)


def get_user_confirmation():
    while True:
        response = input("\nDo you want to post this thread? (yes/no): ").lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        print("Please enter 'yes' or 'no'")


async def process_video(summarizer, video_id, output_path):
    """Process a single video and export its summary"""
    try:
        summary = await summarizer.get_summary(video_id=video_id)
        await summarizer.export_summary(str(output_path))
        return summary
    except Exception as e:
        logging.error(f"Error processing video {video_id}: {str(e)}")
        return None


async def main():
    """Entry point to summarize YouTube video transcripts"""
    parser = argparse.ArgumentParser(
        description="Summarize YouTube video transcripts and post to Twitter."
    )
    parser.add_argument("--channel", help="The channel you are pulling from.")
    parser.add_argument("--schedule", help="The schedule file to pull from.")
    parser.add_argument("--video_id", help="The video ID of the YouTube video.")
    parser.add_argument("--title", help="The title to insert into the final text")
    parser.add_argument(
        "--model", default="gpt-4", help="The model to use for the completion"
    )
    parser.add_argument(
        "--provider", default="openai", help="The provider of the model"
    )
    parser.add_argument(
        "--temperature",
        default=0.3,
        type=float,
        help="Temperature parameter of the model",
    )
    parser.add_argument(
        "--chunk_size",
        default=4000,
        type=int,
        help="The maximum number of tokens to send to the model at once",
    )
    parser.add_argument(
        "--chunk_overlap",
        default=200,
        type=int,
        help="The number of tokens to overlap between chunks",
    )
    parser.add_argument(
        "--use_full_context",
        action="store_true",
        help="Use full context window instead of chunking",
    )
    parser.add_argument(
        "--output_dir", default="summaries", help="Directory to store summary outputs"
    )
    parser.add_argument(
        "--post", action="store_true", help="Generate and post thread to Twitter"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Determine video details from schedule or arguments
    if args.channel and args.video_id and args.title:
        today_entry = {
            "channel": args.channel,
            "video_id": args.video_id,
            "title": args.title,
        }
    else:
        try:
            schedule_path = Path(args.schedule)
            with open(schedule_path, "r") as f:
                schedule = json.load(f)
        except FileNotFoundError:
            logging.error(f"Schedule file not found: {args.schedule}")
            return

        today = datetime.now().strftime("%Y-%m-%d")
        if today not in schedule:
            logging.error("No entry found for today")
            return

        today_entry = schedule[today]

    # Initialize summarizer with new API
    summarizer = Summarizer(
        provider=today_entry.get("provider", args.provider),
        model=today_entry.get("model", args.model),
        temperature=today_entry.get("temperature", args.temperature),
        chunk_size=today_entry.get("chunk_size", args.chunk_size),
        chunk_overlap=args.chunk_overlap,
        use_full_context=args.use_full_context,
        youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
    )

    # Process video and export summary
    video_id = today_entry.get("video_id", args.video_id)
    output_path = output_dir / f"{video_id}_summary.json"

    summary = await process_video(summarizer, video_id, output_path)

    if summary and args.post:
        # Load the exported summary file
        with open(output_path, "r") as f:
            summary_data = json.load(f)

        # Generate thread content
        thread_paragraphs = create_thread_paragraphs(
            summary=summary_data["summary"],
            title=today_entry.get("title"),
            channel=today_entry.get("channel"),
            video_id=video_id,
            provider=summary_data.get("provider", args.provider),
            model=summary_data.get("model", args.model),
            verbose=args.verbose,
        )

        # Save thread to file
        thread_file = output_path.parent / f"{video_id}_thread.txt"
        save_thread(thread_file, thread_paragraphs)

        # Display thread preview and get user confirmation
        display_thread_preview(thread_paragraphs)
        if get_user_confirmation():
            # Post to Twitter
            twitter_client = load_twitter_api()
            post_thread(thread_paragraphs, twitter_client)
        else:
            logging.info("Thread posting cancelled by user")

    await summarizer.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())

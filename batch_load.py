import os
import logging
import csv
import psycopg
from psycopg.rows import dict_row
from datetime import datetime
from dotenv import load_dotenv
from summarizer import TranscriptSummarizer
from scheduler import Scheduler
import argparse

def connect_db(use_dict_row=False):
    """Create database connection using environment variables"""
    load_dotenv()
    
    if os.getenv("DATABASE_URL"):
        return psycopg.connect(os.getenv("DATABASE_URL"))
    
    conn_params = {
        "dbname": os.getenv("PGDATABASE"),
        "host": os.getenv("PGHOST"),
        "port": os.getenv("PGPORT"),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
    }
    
    if use_dict_row:
        conn_params["row_factory"] = dict_row
        
    return psycopg.connect(**conn_params)

def update_summaries(csv_file, provider, model):
    """Update only summaries for existing videos"""
    conn = connect_db(use_dict_row=True)
    # Create cursor with dict_row factory
    cur = conn.cursor(row_factory=dict_row)
    scheduler = Scheduler()

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                url = row[0]
                try:
                    video_id = scheduler._extract_video_id(url)
                    
                    # Check if video exists
                    cur.execute("""
                        SELECT id, title, channel 
                        FROM videos 
                        WHERE video_id = %s
                    """, (video_id,))
                    video = cur.fetchone()
                    
                    if not video:
                        logging.warning(f"Video {url} not found in database, skipping")
                        continue

                    # Generate summary
                    summarizer = TranscriptSummarizer(
                        channel=video['channel'],
                        video_id=video_id,
                        title=video['title'],
                        provider=provider,
                        model=model,
                        db_conn=conn
                    )
                    summarizer.fetch_transcript()
                    summarizer.summarize()

                    # Update summary
                    cur.execute("""
                        UPDATE videos 
                        SET summary = %s
                        WHERE id = %s
                    """, (
                        summarizer.summary,
                        video['id']
                    ))

                    conn.commit()
                    logging.info(f"Updated summary for video {url}")

                except Exception as e:
                    logging.error(f"Error updating summary for video {url}: {str(e)}")
                    conn.rollback()

    finally:
        cur.close()
        conn.close()

def process_videos(csv_file, provider=None, model=None):
    """Process new videos and add them to database"""
    conn = connect_db()
    cur = conn.cursor()
    scheduler = Scheduler()

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                url = row[0]
                try:
                    video_id = scheduler._extract_video_id(url)
                    
                    # Check if video exists
                    cur.execute("SELECT 1 FROM videos WHERE video_id = %s", (video_id,))
                    if cur.fetchone():
                        logging.info(f"Video {url} already exists, skipping")
                        continue

                    # Get video info
                    video_info = scheduler._get_video_info(video_id)
                    
                    # Get transcript and generate summary if model is provided
                    summarizer = TranscriptSummarizer(
                        channel=video_info["channel"],
                        video_id=video_id,
                        title=video_info["title"],
                        provider=provider,
                        model=model
                    )
                    summarizer.fetch_transcript()
                    
                    summary = ""
                    if provider and model:
                        summarizer.summarize()
                        summary = summarizer.summary

                    # Insert into videos table
                    cur.execute("""
                        INSERT INTO videos 
                        (video_id, title, channel, summary, date_added, upvotes, downvotes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        video_id,
                        video_info["title"],
                        video_info["channel"],
                        summary,
                        datetime.now(),
                        0,
                        0
                    ))
                    video_db_id = cur.fetchone()[0]

                    # Insert transcript
                    cur.execute("""
                        INSERT INTO transcripts 
                        (video_id, transcript, created_at)
                        VALUES (%s, %s, %s)
                    """, (
                        video_db_id,
                        summarizer.transcript,
                        datetime.now()
                    ))

                    conn.commit()
                    logging.info(f"Processed video {url}")

                except Exception as e:
                    logging.error(f"Error processing video {url}: {str(e)}")
                    conn.rollback()

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to CSV file containing video URLs")
    parser.add_argument("--update-summaries", action="store_true", 
                       help="Only update summaries for existing videos")
    parser.add_argument("--provider", help="Provider for summarization (e.g., 'openai', 'ollama')")
    parser.add_argument("--model", help="Model name for summarization")
    args = parser.parse_args()
    
    if args.update_summaries:
        if not (args.provider and args.model):
            parser.error("--provider and --model are required when updating summaries")
        update_summaries(args.csv_file, args.provider, args.model)
    else:
        process_videos(args.csv_file, args.provider, args.model)
    
    # Just load videos
    # python batch_load.py videos.csv

    # Load videos and generate summaries
    # python batch_load.py videos.csv --provider openai --model gpt-4

    # Update summaries for existing videos
    # python batch_load.py videos.csv --update-summaries --provider ollama --model llama2

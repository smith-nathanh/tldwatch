import os
import logging
import csv
import psycopg
from psycopg.rows import dict_row
from datetime import datetime
from dotenv import load_dotenv
from summarizer import TranscriptSummarizer
from scheduler import Scheduler

def connect_db():
    """Create database connection using environment variables"""
    load_dotenv()
    
    if os.getenv("DATABASE_URL"):
        return psycopg.connect(os.getenv("DATABASE_URL"))
    
    return psycopg.connect(
        dbname=os.getenv("PGDATABASE"),
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        row_factory=dict_row
    )

def process_videos(csv_file):
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
                    
                    # Get transcript
                    summarizer = TranscriptSummarizer(
                        channel=video_info["channel"],
                        video_id=video_id,
                        title=video_info["title"]
                    )
                    summarizer.fetch_transcript()

                    # Insert into videos table
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
                        "",  # Empty summary as placeholder
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

                except Exception as e:
                    logging.error(f"Error processing video {url}: {str(e)}")
                    conn.rollback()

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # usage: 
    # python batch_processor.py videos.csv
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to CSV file containing video URLs")
    args = parser.parse_args()
    
    process_videos(args.csv_file)
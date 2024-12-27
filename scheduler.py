import os
import json
from datetime import datetime
from typing import Optional
import re
import argparse
import logging
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

# Configure module-level logger
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, schedule_path: str = "schedule/schedule.json"):
        """Initialize the Scheduler with a path to the schedule file."""
        # Create schedule directory if it doesn't exist
        os.makedirs(os.path.dirname(schedule_path), exist_ok=True)
        
        self.schedule_path = schedule_path
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")
        
        # Initialize YouTube API client
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
    
    def _load_schedule(self) -> dict:
        """Load the current schedule from file."""
        try:
            with open(self.schedule_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_schedule(self, schedule: dict):
        """Save the updated schedule to file."""
        with open(self.schedule_path, 'w') as f:
            json.dump(schedule, f, indent=4)

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu.be\/)([A-Za-z0-9_-]+)',
            r'youtube\.com\/embed\/([A-Za-z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError("Invalid YouTube URL format")

    def _get_video_info(self, video_id: str) -> dict:
        """Fetch video information from YouTube API."""
        try:
            response = self.youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()
            
            if not response["items"]:
                raise ValueError(f"No video found with ID: {video_id}")
            
            video_info = response["items"][0]["snippet"]
            return {
                "channel": video_info["channelTitle"],
                "video_id": video_id,
                "title": video_info["title"],
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
            
        except HttpError as e:
            raise Exception(f"Error fetching video info: {str(e)}")

    def _find_next_date(self) -> str:
        """Find the next day after the last scheduled date."""
        schedule = self._load_schedule()
        if not schedule:
            return datetime.now().strftime("%Y-%m-%d")
            
        last_date = max(schedule.keys())
        next_date = datetime.strptime(last_date, "%Y-%m-%d")
        next_date = next_date.replace(day=next_date.day + 1)
        return next_date.strftime("%Y-%m-%d")

    def add_video(self, url: str, date: Optional[str] = None, model: str = "llama3.1-70b"):
        """
        Add a video to the schedule. If no date provided, adds to next available date.
        
        Args:
            url (str): YouTube video URL
            date (Optional[str]): Date in YYYY-MM-DD format. If None, uses next available date
            model (str, optional): Model name. Defaults to "llama3.1-70b"
        """
        if date is None:
            date = self._find_next_date()
        else:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")

        # Get current schedule
        schedule = self._load_schedule()
        
        # Extract video ID and get info
        video_id = self._extract_video_id(url)
        video_info = self._get_video_info(video_id)
        
        # Add model information
        video_info["model"] = model
        
        # Update schedule
        schedule[date] = video_info
        
        # Save updated schedule
        self._save_schedule(schedule)
        logger.info(f"Added video {video_id} scheduled for {date}")

    def remove_video(self, date: str):
        """Remove a video from the schedule for a specific date."""
        schedule = self._load_schedule()
        if date in schedule:
            del schedule[date]
            self._save_schedule(schedule)
            logger.info(f"Removed video scheduled for {date}")
        else:
            logger.warning(f"Attempted to remove non-existent video scheduled for {date}")

def main():
    parser = argparse.ArgumentParser(description='Manage YouTube video schedule')
    parser.add_argument('action', choices=['add', 'remove'], help='Action to perform')
    parser.add_argument('--date', help='Date in YYYY-MM-DD format (optional)')
    parser.add_argument('--url', help='YouTube video URL (required for add)', required=False)
    parser.add_argument('--model', default='llama3.1-70b', help='Model name (default: llama3.1-70b)')
    
    args = parser.parse_args()
    
    try:
        scheduler = Scheduler()
        
        if args.action == 'add':
            if not args.url:
                parser.error("--url is required for add action")
            scheduler.add_video(args.url, args.date, args.model)
        else:  # remove
            if not args.date:
                parser.error("--date is required for remove action")
            scheduler.remove_video(args.date)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()

# Run the script with the following commands:

# Add video to next available date
# python scheduler.py add --url "https://youtube.com/watch?v=example"

# # Add video with specific date and model
# python scheduler.py add --url "https://youtube.com/watch?v=example" --date "2024-12-31" --model "gpt-4"

# # Add video with different model but auto date
# python scheduler.py add --url "https://youtube.com/watch?v=example" --model "qwen2.5:1.5b"

# # Remove video for specific date
# python scheduler.py add --date "2024-12-31"

    # "2024-12-29": {
    #     "model": "llama3.1-70b",
    #     "channel": "Machine Learning Street Talk",
    #     "video_id": "s7_NlkBwdj8",
    #     "title": "It's Not About Scale, It's About Abstraction",
    #     "url": "https://www.youtube.com/watch?v=s7_NlkBwdj8"
    # }
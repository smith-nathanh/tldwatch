"""
Quickstart example for tldwatch library usage.
Shows the most common ways to use the library.
"""

import asyncio
import logging

from tldwatch import Summarizer

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    try:
        # Initialize summarizer
        logger.info("Initializing summarizer")
        summarizer = Summarizer(
            provider="openai",
            model="gpt-4o-mini",
            use_full_context=False,
            temperature=0.7,
            chunk_size=5000,
            chunk_overlap=200,
        )
        logger.info("Summarizer initialized successfully")

        # Summarize from YouTube video ID
        video_id = "QAgR4uQ15rc"
        logger.info(f"Starting summary process for video ID: {video_id}")

        try:
            summary = await summarizer.get_summary(video_id=video_id)
            logger.info("Summary generation completed")
            logger.info("\nSummary from video ID: %s", summary)

            # Export summary to file
            await summarizer.export_summary("summary.json")
            logger.info("Summary exported to summary.json")

        except Exception as e:
            logger.error("Error processing video: %s", str(e))
            raise

    except Exception as e:
        logger.error("Fatal error: %s", str(e))
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error("Process failed: %s", str(e))

"""RSS feed scheduler using APScheduler."""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from typing import Dict, Optional
from app.ingestion.pipeline import IngestionPipeline
import logging

logger = logging.getLogger(__name__)


class FeedScheduler:
    """Scheduler for RSS feed polling."""

    def __init__(self):
        """Initialize scheduler."""
        self.scheduler = AsyncIOScheduler()
        self.pipeline = IngestionPipeline()
        self._feeds: Dict[str, Dict] = {}
        self._running = False

    def add_feed(
        self,
        feed_id: str,
        feed_url: str,
        feed_name: str,
        user_id: str,
        update_frequency: str = "hourly",
    ) -> None:
        """
        Add a feed to the scheduler.

        Args:
            feed_id: Feed ID
            feed_url: RSS feed URL
            feed_name: Feed name
            user_id: User ID
            update_frequency: Update frequency (hourly, daily)
        """
        # Determine interval
        if update_frequency == "hourly":
            hours = 1
        elif update_frequency == "daily":
            hours = 24
        else:
            hours = 1  # Default to hourly

        # Store feed info
        self._feeds[feed_id] = {
            "feed_url": feed_url,
            "feed_name": feed_name,
            "user_id": user_id,
        }

        # Add job
        self.scheduler.add_job(
            self._ingest_feed_job,
            IntervalTrigger(hours=hours),
            id=feed_id,
            args=[feed_id],
            replace_existing=True,
        )

        logger.info(f"Added feed {feed_id} with {update_frequency} polling")

    def remove_feed(self, feed_id: str) -> None:
        """
        Remove a feed from the scheduler.

        Args:
            feed_id: Feed ID
        """
        if feed_id in self._feeds:
            self.scheduler.remove_job(feed_id)
            del self._feeds[feed_id]
            logger.info(f"Removed feed {feed_id} from scheduler")

    async def _ingest_feed_job(self, feed_id: str) -> None:
        """
        Job function to ingest a feed.

        Args:
            feed_id: Feed ID
        """
        if feed_id not in self._feeds:
            logger.warning(f"Feed {feed_id} not found in scheduler")
            return

        feed_info = self._feeds[feed_id]
        logger.info(f"Starting scheduled ingestion for feed {feed_id}")

        try:
            result = await self.pipeline.ingest_feed(
                feed_url=feed_info["feed_url"],
                feed_name=feed_info["feed_name"],
                user_id=feed_info["user_id"],
                feed_id=feed_id,
            )
            logger.info(
                f"Completed ingestion for feed {feed_id}: "
                f"{result.get('articles_ingested', 0)} articles ingested"
            )
        except Exception as e:
            logger.error(f"Error ingesting feed {feed_id}: {e}")

    def start(self) -> None:
        """Start the scheduler."""
        if not self._running:
            self.scheduler.start()
            self._running = True
            logger.info("Feed scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown()
            self._running = False
            logger.info("Feed scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


"""RSS feed scheduler using APScheduler."""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from typing import Dict, Optional
from datetime import datetime
from app.ingestion.pipeline import IngestionPipeline
from app.clients.supabase import SupabaseClient
from app.ingestion.rss_parser import RSSParser
import logging
import asyncio

logger = logging.getLogger(__name__)


class FeedScheduler:
    """Scheduler for RSS feed polling with database-backed feed management."""

    def __init__(self):
        """Initialize scheduler."""
        self.scheduler = AsyncIOScheduler()
        self.pipeline = IngestionPipeline()
        self.supabase_client = SupabaseClient()
        self.rss_parser = RSSParser()
        self._running = False
        self._refresh_interval = 300  # Refresh feed list every 5 minutes

    async def load_feeds(self) -> None:
        """Load active feeds from database and schedule them."""
        try:
            # Get all active feeds
            feeds = await self.supabase_client.list_feeds(status="active")

            # Remove feeds that are no longer active
            scheduled_job_ids = [job.id for job in self.scheduler.get_jobs()]
            for job_id in scheduled_job_ids:
                feed_exists = any(feed["id"] == job_id for feed in feeds)
                if not feed_exists:
                    self.scheduler.remove_job(job_id)
                    logger.info(f"Removed feed {job_id} from scheduler (no longer active)")

            # Add or update feeds
            for feed in feeds:
                feed_id = feed["id"]
                update_frequency = feed.get("update_frequency", "hourly")

                # Determine interval
                if update_frequency == "hourly":
                    hours = 1
                elif update_frequency == "daily":
                    hours = 24
                else:
                    hours = 1  # Default to hourly

                # Add or update job
                self.scheduler.add_job(
                    self._ingest_feed_job,
                    IntervalTrigger(hours=hours),
                    id=feed_id,
                    args=[feed_id],
                    replace_existing=True,
                )

            logger.info(f"Loaded {len(feeds)} active feeds into scheduler")

        except Exception as e:
            logger.error(f"Error loading feeds: {e}")

    async def _ingest_feed_job(self, feed_id: str) -> None:
        """
        Job function to ingest a feed with incremental ingestion support.

        Args:
            feed_id: Feed ID
        """
        try:
            # Get feed from database
            from uuid import UUID

            feed_uuid = UUID(feed_id)
            feed_data = await self.supabase_client.get_feed(feed_uuid)

            if not feed_data:
                logger.warning(f"Feed {feed_id} not found in database")
                return

            # Check feed status
            if feed_data["status"] != "active":
                logger.info(f"Feed {feed_id} is not active, skipping ingestion")
                return

            logger.info(f"Starting scheduled ingestion for feed {feed_id} ({feed_data['name']})")

            # Update feed status (optional - could mark as processing)
            # For now, we'll just ingest

            # Run ingestion pipeline
            result = await self.pipeline.ingest_feed(
                feed_url=feed_data["url"],
                feed_name=feed_data["name"],
                user_id=feed_data["user_id"],
                feed_id=feed_id,
            )

            # Update feed with results
            update_data = {
                "last_ingested_at": datetime.now(),
                "article_count": feed_data.get("article_count", 0) + result.get("articles_ingested", 0),
            }

            if result.get("status") == "error":
                update_data["status"] = "error"
                update_data["error_message"] = result.get("error", "Unknown error")
                logger.error(f"Feed {feed_id} ingestion failed: {result.get('error')}")
            else:
                update_data["status"] = "active"
                update_data["error_message"] = None
                logger.info(
                    f"Completed ingestion for feed {feed_id}: "
                    f"{result.get('articles_ingested', 0)} articles ingested, "
                    f"{result.get('articles_failed', 0)} failed"
                )

            await self.supabase_client.update_feed(feed_uuid, update_data)

        except Exception as e:
            logger.error(f"Error ingesting feed {feed_id}: {e}")
            # Update feed status to error
            try:
                from uuid import UUID

                feed_uuid = UUID(feed_id)
                await self.supabase_client.update_feed(
                    feed_uuid,
                    {
                        "status": "error",
                        "error_message": str(e),
                    },
                )
            except Exception as update_error:
                logger.error(f"Failed to update feed error status: {update_error}")

    async def _refresh_feeds_periodically(self) -> None:
        """Periodically refresh feed list from database."""
        while self._running:
            await asyncio.sleep(self._refresh_interval)
            if self._running:
                await self.load_feeds()

    def start(self) -> None:
        """Start the scheduler."""
        if not self._running:
            self.scheduler.start()
            self._running = True
            logger.info("Feed scheduler started")

            # Load feeds initially
            asyncio.create_task(self.load_feeds())

            # Start periodic refresh
            asyncio.create_task(self._refresh_feeds_periodically())

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown()
            self._running = False
            logger.info("Feed scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

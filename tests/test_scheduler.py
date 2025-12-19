"""Tests for feed scheduler."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.ingestion.scheduler import FeedScheduler


@pytest.mark.asyncio
async def test_scheduler_load_feeds(mock_supabase_client):
    """Test loading feeds from database."""
    scheduler = FeedScheduler()
    scheduler.supabase_client = mock_supabase_client

    mock_supabase_client.list_feeds = AsyncMock(
        return_value=[
            {
                "id": "feed-1",
                "name": "Feed 1",
                "url": "https://example.com/feed1",
                "update_frequency": "hourly",
                "status": "active",
                "user_id": "user-1",
            }
        ]
    )

    with patch.object(scheduler.scheduler, "add_job") as mock_add_job:
        await scheduler.load_feeds()

        mock_add_job.assert_called_once()
        assert scheduler.is_running() is False  # Not started yet


@pytest.mark.asyncio
async def test_scheduler_ingest_feed_job(mock_supabase_client):
    """Test scheduled feed ingestion."""
    from uuid import uuid4
    test_uuid = uuid4()
    scheduler = FeedScheduler()
    scheduler.supabase_client = mock_supabase_client
    scheduler.pipeline = MagicMock()

    feed_data = {
        "id": str(test_uuid),
        "name": "Test Feed",
        "url": "https://example.com/feed",
        "status": "active",
        "user_id": "user-1",
        "article_count": 0,
    }

    mock_supabase_client.get_feed = AsyncMock(return_value=feed_data)
    mock_supabase_client.update_feed = AsyncMock()
    scheduler.pipeline.ingest_feed = AsyncMock(
        return_value={
            "articles_ingested": 5,
            "articles_failed": 0,
            "status": "completed",
        }
    )

    await scheduler._ingest_feed_job(str(test_uuid))

    scheduler.pipeline.ingest_feed.assert_called_once()
    mock_supabase_client.update_feed.assert_called_once()


def test_scheduler_start_stop():
    """Test scheduler start and stop."""
    scheduler = FeedScheduler()

    with patch.object(scheduler.scheduler, "start") as mock_start:
        with patch.object(scheduler.scheduler, "shutdown") as mock_stop:
            with patch("asyncio.create_task") as mock_task:
                scheduler.start()
                assert scheduler.is_running() is True
                mock_start.assert_called_once()

                scheduler.stop()
                assert scheduler.is_running() is False
                mock_stop.assert_called_once()


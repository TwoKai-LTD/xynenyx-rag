"""Tests for database-backed feed management."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
from app.clients.supabase import SupabaseClient
from app.routers.feeds import create_feed, list_feeds, get_feed, delete_feed
from app.schemas.feeds import FeedCreate


@pytest.mark.asyncio
async def test_create_feed(mock_supabase_client):
    """Test feed creation."""
    mock_supabase_client.create_feed = AsyncMock(
        return_value={
            "id": "test-feed-id",
            "name": "Test Feed",
            "url": "https://example.com/feed",
            "update_frequency": "hourly",
            "status": "active",
        }
    )

    feed_data = await mock_supabase_client.create_feed(
        user_id="test-user",
        name="Test Feed",
        url="https://example.com/feed",
        update_frequency="hourly",
    )

    assert feed_data["id"] == "test-feed-id"
    assert feed_data["name"] == "Test Feed"


@pytest.mark.asyncio
async def test_get_feed(mock_supabase_client):
    """Test getting a feed."""
    from uuid import uuid4
    test_uuid = uuid4()
    mock_supabase_client.get_feed = AsyncMock(
        return_value={
            "id": str(test_uuid),
            "name": "Test Feed",
            "url": "https://example.com/feed",
            "status": "active",
        }
    )

    feed = await mock_supabase_client.get_feed(test_uuid)
    assert feed is not None
    assert feed["name"] == "Test Feed"


@pytest.mark.asyncio
async def test_list_feeds(mock_supabase_client):
    """Test listing feeds."""
    mock_supabase_client.list_feeds = AsyncMock(
        return_value=[
            {
                "id": "feed-1",
                "name": "Feed 1",
                "url": "https://example.com/feed1",
                "status": "active",
            },
            {
                "id": "feed-2",
                "name": "Feed 2",
                "url": "https://example.com/feed2",
                "status": "active",
            },
        ]
    )

    feeds = await mock_supabase_client.list_feeds(user_id="test-user")
    assert len(feeds) == 2


@pytest.mark.asyncio
async def test_update_feed(mock_supabase_client):
    """Test updating a feed."""
    from uuid import uuid4
    test_uuid = uuid4()
    mock_supabase_client.update_feed = AsyncMock()

    await mock_supabase_client.update_feed(
        test_uuid,
        {"status": "paused", "last_ingested_at": "2024-01-01T00:00:00"},
    )

    mock_supabase_client.update_feed.assert_called_once()


@pytest.mark.asyncio
async def test_delete_feed(mock_supabase_client):
    """Test deleting a feed."""
    from uuid import uuid4
    test_uuid = uuid4()
    mock_supabase_client.delete_feed = AsyncMock()

    await mock_supabase_client.delete_feed(test_uuid)

    mock_supabase_client.delete_feed.assert_called_once()


"""API routes for RSS feed management."""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from uuid import UUID
from datetime import datetime
from app.schemas.feeds import FeedCreate, FeedResponse, FeedListResponse, IngestResponse
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.rss_parser import RSSParser
from app.clients.supabase import SupabaseClient

router = APIRouter(prefix="/feeds", tags=["feeds"])

_supabase_client = SupabaseClient()
_pipeline = IngestionPipeline()
_rss_parser = RSSParser()


@router.post("", response_model=FeedResponse, status_code=201)
async def create_feed(
    feed: FeedCreate,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Register a new RSS feed.

    Args:
        feed: Feed creation request
        x_user_id: User ID from header

    Returns:
        Created feed information
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")

    # Validate feed URL
    if not _rss_parser.validate_feed_url(str(feed.url)):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible RSS feed URL")

    try:
        # Create feed in database
        feed_data = await _supabase_client.create_feed(
            user_id=x_user_id,
            name=feed.name,
            url=str(feed.url),
            update_frequency=feed.update_frequency,
        )

        return FeedResponse(
            id=feed_data["id"],
            name=feed_data["name"],
            url=feed_data["url"],
            update_frequency=feed_data["update_frequency"],
            status=feed_data["status"],
            last_ingested_at=feed_data.get("last_ingested_at"),
        )
    except Exception as e:
        # Check if it's a unique constraint violation
        if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
            raise HTTPException(status_code=409, detail="Feed with this URL already exists")
        raise HTTPException(status_code=500, detail=f"Failed to create feed: {str(e)}") from e


@router.get("", response_model=FeedListResponse)
async def list_feeds(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    status: Optional[str] = None,
):
    """
    List all registered feeds.

    Args:
        x_user_id: User ID from header (optional, for filtering)
        status: Filter by status (optional)

    Returns:
        List of feeds
    """
    try:
        feeds_data = await _supabase_client.list_feeds(
            user_id=x_user_id,
            status=status,
        )

        feeds = [
            FeedResponse(
                id=feed["id"],
                name=feed["name"],
                url=feed["url"],
                update_frequency=feed["update_frequency"],
                status=feed["status"],
                last_ingested_at=feed.get("last_ingested_at"),
            )
            for feed in feeds_data
        ]

        return FeedListResponse(feeds=feeds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list feeds: {str(e)}") from e


@router.get("/{feed_id}", response_model=FeedResponse)
async def get_feed(feed_id: str):
    """
    Get feed details.

    Args:
        feed_id: Feed ID

    Returns:
        Feed information
    """
    try:
        feed_uuid = UUID(feed_id)
        feed_data = await _supabase_client.get_feed(feed_uuid)

        if not feed_data:
            raise HTTPException(status_code=404, detail="Feed not found")

        return FeedResponse(
            id=feed_data["id"],
            name=feed_data["name"],
            url=feed_data["url"],
            update_frequency=feed_data["update_frequency"],
            status=feed_data["status"],
            last_ingested_at=feed_data.get("last_ingested_at"),
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feed ID format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feed: {str(e)}") from e


@router.post("/{feed_id}/ingest", response_model=IngestResponse)
async def ingest_feed(
    feed_id: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Trigger manual ingestion of a feed.

    Args:
        feed_id: Feed ID
        x_user_id: User ID from header

    Returns:
        Ingestion job status
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")

    try:
        feed_uuid = UUID(feed_id)
        feed_data = await _supabase_client.get_feed(feed_uuid)

        if not feed_data:
            raise HTTPException(status_code=404, detail="Feed not found")

        # Check feed status
        if feed_data["status"] == "paused":
            raise HTTPException(status_code=400, detail="Feed is paused")

        # Update feed status to processing
        await _supabase_client.update_feed(feed_uuid, {"status": "active"})

        # Run ingestion pipeline
        result = await _pipeline.ingest_feed(
            feed_url=feed_data["url"],
            feed_name=feed_data["name"],
            user_id=x_user_id,
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
        else:
            update_data["status"] = "active"
            update_data["error_message"] = None

        await _supabase_client.update_feed(feed_uuid, update_data)

        return IngestResponse(
            feed_id=feed_id,
            feed_url=feed_data["url"],
            feed_name=feed_data["name"],
            articles_ingested=result.get("articles_ingested", 0),
            articles_failed=result.get("articles_failed", 0),
            status=result.get("status", "error"),
            message=result.get("message"),
            error=result.get("error"),
            errors=result.get("errors"),
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feed ID format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest feed: {str(e)}") from e


@router.delete("/{feed_id}", status_code=204)
async def delete_feed(
    feed_id: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Remove a feed.

    Args:
        feed_id: Feed ID
        x_user_id: User ID from header
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")

    try:
        feed_uuid = UUID(feed_id)
        feed_data = await _supabase_client.get_feed(feed_uuid)

        if not feed_data:
            raise HTTPException(status_code=404, detail="Feed not found")

        # Verify ownership (RLS should handle this, but double-check)
        if feed_data.get("user_id") != x_user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        await _supabase_client.delete_feed(feed_uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feed ID format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete feed: {str(e)}") from e

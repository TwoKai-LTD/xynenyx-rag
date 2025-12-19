"""API routes for RSS feed management."""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.feeds import FeedCreate, FeedResponse, FeedListResponse, IngestResponse
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.rss_parser import RSSParser

router = APIRouter(prefix="/feeds", tags=["feeds"])

# In-memory feed storage (will be replaced with database in future)
_feeds: dict[str, dict] = {}
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

    # Generate feed ID (simple hash for now)
    import hashlib

    feed_id = hashlib.md5(str(feed.url).encode()).hexdigest()

    # Store feed
    feed_data = {
        "id": feed_id,
        "name": feed.name,
        "url": str(feed.url),
        "update_frequency": feed.update_frequency,
        "status": "active",
        "last_ingested_at": None,
    }
    _feeds[feed_id] = feed_data

    return FeedResponse(**feed_data)


@router.get("", response_model=FeedListResponse)
async def list_feeds():
    """
    List all registered feeds.

    Returns:
        List of feeds
    """
    feeds = [FeedResponse(**feed) for feed in _feeds.values()]
    return FeedListResponse(feeds=feeds)


@router.get("/{feed_id}", response_model=FeedResponse)
async def get_feed(feed_id: str):
    """
    Get feed details.

    Args:
        feed_id: Feed ID

    Returns:
        Feed information
    """
    if feed_id not in _feeds:
        raise HTTPException(status_code=404, detail="Feed not found")

    return FeedResponse(**_feeds[feed_id])


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

    if feed_id not in _feeds:
        raise HTTPException(status_code=404, detail="Feed not found")

    feed_data = _feeds[feed_id]

    # Run ingestion pipeline
    result = await _pipeline.ingest_feed(
        feed_url=feed_data["url"],
        feed_name=feed_data["name"],
        user_id=x_user_id,
        feed_id=feed_id,
    )

    # Update last ingested timestamp
    from datetime import datetime

    _feeds[feed_id]["last_ingested_at"] = datetime.now()

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


@router.delete("/{feed_id}", status_code=204)
async def delete_feed(feed_id: str):
    """
    Remove a feed.

    Args:
        feed_id: Feed ID
    """
    if feed_id not in _feeds:
        raise HTTPException(status_code=404, detail="Feed not found")

    del _feeds[feed_id]


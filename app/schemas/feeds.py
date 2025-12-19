"""Pydantic models for RSS feed management."""
from pydantic import BaseModel, HttpUrl, Field, ConfigDict
from typing import Optional
from datetime import datetime


class FeedCreate(BaseModel):
    """Request model for creating a feed."""

    name: str = Field(..., description="Feed name")
    url: HttpUrl = Field(..., description="RSS feed URL")
    update_frequency: str = Field(
        default="hourly", description="Update frequency (hourly, daily)"
    )


class FeedResponse(BaseModel):
    """Response model for feed information."""

    id: str = Field(..., description="Feed ID")
    name: str = Field(..., description="Feed name")
    url: str = Field(..., description="RSS feed URL")
    update_frequency: str = Field(..., description="Update frequency")
    status: str = Field(default="active", description="Feed status")
    last_ingested_at: Optional[datetime] = Field(None, description="Last ingestion timestamp")

    model_config = ConfigDict(from_attributes=True)


class FeedListResponse(BaseModel):
    """Response model for feed list."""

    feeds: list[FeedResponse] = Field(..., description="List of feeds")


class IngestResponse(BaseModel):
    """Response model for ingestion job."""

    feed_id: str = Field(..., description="Feed ID")
    feed_url: str = Field(..., description="Feed URL")
    feed_name: str = Field(..., description="Feed name")
    articles_ingested: int = Field(..., description="Number of articles ingested")
    articles_failed: int = Field(default=0, description="Number of articles that failed")
    status: str = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")
    errors: Optional[list[str]] = Field(None, description="List of errors")


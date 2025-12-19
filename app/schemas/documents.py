"""Pydantic models for document management."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime


class DocumentResponse(BaseModel):
    """Response model for document information."""

    id: str = Field(..., description="Document ID")
    name: str = Field(..., description="Document name")
    status: str = Field(..., description="Document status")
    chunk_count: int = Field(default=0, description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: dict = Field(default_factory=dict, description="Document metadata")

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Response model for document list."""

    documents: List[DocumentResponse] = Field(..., description="List of documents")
    count: int = Field(..., description="Total number of documents")


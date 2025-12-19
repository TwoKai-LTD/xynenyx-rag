"""Pydantic models for query requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID


class QueryRequest(BaseModel):
    """Request model for vector search query."""

    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filter_document_ids: Optional[List[UUID]] = Field(
        None, description="Optional list of document IDs to filter by"
    )


class QueryResult(BaseModel):
    """Single query result."""

    content: str = Field(..., description="Chunk content")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: dict = Field(..., description="Chunk metadata")
    document_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")


class QueryResponse(BaseModel):
    """Response model for query results."""

    query: str = Field(..., description="Original query")
    results: List[QueryResult] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results returned")


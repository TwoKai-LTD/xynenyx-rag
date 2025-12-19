"""Pydantic models for query requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID


class QueryRequest(BaseModel):
    """Request model for vector search query with advanced options."""

    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filter_document_ids: Optional[List[UUID]] = Field(
        None, description="Optional list of document IDs to filter by"
    )
    use_hybrid_search: bool = Field(
        default=True, description="Use hybrid search (BM25 + vector)"
    )
    use_reranking: bool = Field(default=True, description="Enable reranking")
    rerank_top_n: int = Field(
        default=20, ge=1, le=100, description="Top N results to rerank"
    )
    date_filter: Optional[str | Dict[str, str]] = Field(
        None, description="Temporal filter (preset string or dict with start_date/end_date)"
    )
    company_filter: Optional[List[str]] = Field(
        None, description="Filter by company names"
    )
    investor_filter: Optional[List[str]] = Field(
        None, description="Filter by investor names"
    )
    sector_filter: Optional[List[str]] = Field(
        None, description="Filter by sectors/industries"
    )


class QueryResult(BaseModel):
    """Single query result with enhanced scoring."""

    content: str = Field(..., description="Chunk content")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: dict = Field(..., description="Chunk metadata")
    document_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")
    rerank_score: Optional[float] = Field(None, description="Rerank score (if reranking enabled)")
    bm25_score: Optional[float] = Field(None, description="BM25 score (if hybrid search enabled)")
    vector_score: Optional[float] = Field(None, description="Vector similarity score")
    rrf_score: Optional[float] = Field(None, description="RRF fusion score (if hybrid search enabled)")


class QueryResponse(BaseModel):
    """Response model for query results with search metadata."""

    query: str = Field(..., description="Original query")
    results: List[QueryResult] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results returned")
    search_mode: str = Field(..., description="Search mode used (vector, bm25, hybrid)")
    reranking_enabled: bool = Field(default=False, description="Whether reranking was applied")


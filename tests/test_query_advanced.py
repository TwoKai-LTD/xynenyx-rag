"""Tests for advanced query endpoint with hybrid search and filtering."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI test client."""
    from app.main import app
    return TestClient(app)


def test_query_with_hybrid_search(client):
    """Test query with hybrid search enabled."""
    from app.routers import query
    
    # Create a proper async mock function
    async def mock_retrieve(*args, **kwargs):
        return [
            {
                "content": "Test content",
                "rrf_score": 0.9,
                "bm25_score": 0.8,
                "vector_score": 0.85,
                "metadata": {},
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
            }
        ]
    
    # Mock the hybrid retriever's retrieve method
    original_retrieve = query._hybrid_retriever.retrieve
    query._hybrid_retriever.retrieve = mock_retrieve

    try:
        response = client.post(
            "/query",
            json={
                "query": "AI startup",
                "top_k": 5,
                "use_hybrid_search": True,
                "use_reranking": False,
            },
            headers={"X-User-ID": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "hybrid"
        assert len(data["results"]) > 0
    finally:
        query._hybrid_retriever.retrieve = original_retrieve


def test_query_with_reranking(client):
    """Test query with reranking enabled."""
    from app.routers import query
    
    async def mock_retrieve(*args, **kwargs):
        return [
            {
                "content": "Test content",
                "rrf_score": 0.9,
                "metadata": {},
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
            }
        ]
    
    def mock_rerank(query, documents, top_k=None):
        return [
            {
                "content": "Test content",
                "rerank_score": 0.95,
                "metadata": {},
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
            }
        ]
    
    original_retrieve = query._hybrid_retriever.retrieve
    original_rerank = query._reranker.rerank
    query._hybrid_retriever.retrieve = mock_retrieve
    query._reranker.rerank = mock_rerank

    try:
        response = client.post(
            "/query",
            json={
                "query": "AI startup",
                "top_k": 5,
                "use_hybrid_search": True,
                "use_reranking": True,
                "rerank_top_n": 10,
            },
            headers={"X-User-ID": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["reranking_enabled"] is True
        assert data["results"][0].get("rerank_score") is not None
    finally:
        query._hybrid_retriever.retrieve = original_retrieve
        query._reranker.rerank = original_rerank


def test_query_with_filters(client):
    """Test query with temporal and entity filters."""
    from app.routers import query
    from datetime import datetime
    
    async def mock_retrieve(*args, **kwargs):
        return [
            {
                "content": "Test content",
                "rrf_score": 0.9,
                "metadata": {
                    "published_date": "2024-01-15",
                    "companies": ["Anthropic"],
                },
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
            }
        ]
    
    original_retrieve = query._hybrid_retriever.retrieve
    original_parse = query._temporal_filter.parse_filter
    original_temporal_filter = query._temporal_filter.filter_results
    original_entity_filter = query._entity_filter.filter_results
    
    query._hybrid_retriever.retrieve = mock_retrieve
    
    def mock_parse_filter(date_filter):
        return {
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 31),
        }
    
    def mock_temporal_filter_results(results, date_range):
        return results
    
    def mock_entity_filter_results(results, company_filter=None, investor_filter=None, sector_filter=None):
        return results
    
    query._temporal_filter.parse_filter = mock_parse_filter
    query._temporal_filter.filter_results = mock_temporal_filter_results
    query._entity_filter.filter_results = mock_entity_filter_results

    try:
        response = client.post(
            "/query",
            json={
                "query": "AI startup",
                "top_k": 5,
                "date_filter": "last_week",
                "company_filter": ["Anthropic"],
            },
            headers={"X-User-ID": "test-user"},
        )

        assert response.status_code == 200
    finally:
        query._hybrid_retriever.retrieve = original_retrieve
        query._temporal_filter.parse_filter = original_parse
        query._temporal_filter.filter_results = original_temporal_filter
        query._entity_filter.filter_results = original_entity_filter


def test_query_vector_only(client):
    """Test query with vector-only search (backward compatibility)."""
    from app.routers import query
    
    async def mock_retrieve(*args, **kwargs):
        return [
            {
                "content": "Test content",
                "similarity": 0.9,
                "vector_score": 0.9,
                "metadata": {},
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
            }
        ]
    
    original_retrieve = query._vector_only_retriever.retrieve
    query._vector_only_retriever.retrieve = mock_retrieve

    try:
        response = client.post(
            "/query",
            json={
                "query": "AI startup",
                "top_k": 5,
                "use_hybrid_search": False,
            },
            headers={"X-User-ID": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "vector"
    finally:
        query._vector_only_retriever.retrieve = original_retrieve

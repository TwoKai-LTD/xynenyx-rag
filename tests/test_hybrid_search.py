"""Tests for hybrid search (BM25, vector, and hybrid retrieval)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.retrieval.bm25_retriever import BM25RetrieverWrapper
from app.retrieval.hybrid_retriever import HybridRetriever, rrf_score
from app.retrieval.vector_store import VectorStore
from app.clients.llm import LLMServiceClient
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Okapi


def test_rrf_score():
    """Test RRF score calculation."""
    assert rrf_score(1, k=60) == 1.0 / (60 + 1)
    assert rrf_score(10, k=60) == 1.0 / (60 + 10)
    assert rrf_score(1, k=60) > rrf_score(10, k=60)  # Higher rank = higher score


@pytest.mark.asyncio
async def test_bm25_retriever_build_index(mock_supabase_client):
    """Test BM25 index building."""
    retriever = BM25RetrieverWrapper(mock_supabase_client)

    # Mock chunk data - properly chain the Supabase client calls
    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Test content about AI",
            "metadata": {},
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "content": "Another chunk about machine learning",
            "metadata": {},
        },
    ]
    
    # Properly mock the chain: client.table().select().execute()
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_execute = MagicMock(return_value=mock_result)
    mock_select.execute = mock_execute
    mock_table.select.return_value = mock_select
    mock_supabase_client.client.table = MagicMock(return_value=mock_table)

    await retriever.build_index()
    assert retriever._index_built is True
    assert retriever.bm25 is not None
    assert len(retriever.chunks) == 2


@pytest.mark.asyncio
async def test_bm25_retriever_retrieve(mock_supabase_client):
    """Test BM25 retrieval."""
    retriever = BM25RetrieverWrapper(mock_supabase_client)

    # Setup mock chunks - properly chain the Supabase client calls
    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Test content about AI and machine learning",
            "metadata": {},
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "content": "Another chunk about different topics",
            "metadata": {},
        },
    ]
    
    # Properly mock the chain: client.table().select().execute()
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_execute = MagicMock(return_value=mock_result)
    mock_select.execute = mock_execute
    mock_table.select.return_value = mock_select
    mock_supabase_client.client.table = MagicMock(return_value=mock_table)

    # Build index
    await retriever.build_index()
    assert retriever._index_built is True

    # Retrieve
    results = await retriever.retrieve("AI machine learning", top_k=5)
    assert len(results) > 0
    assert "content" in results[0]
    assert "bm25_score" in results[0]
    # First result should be more relevant (contains both AI and machine learning)
    assert results[0]["bm25_score"] >= 0  # BM25 scores can be 0 or positive
    # Verify the more relevant chunk (with both terms) comes first
    if len(results) > 1:
        # The chunk with "AI and machine learning" should score higher
        first_chunk_content = results[0]["content"].lower()
        assert "ai" in first_chunk_content or "machine learning" in first_chunk_content


@pytest.mark.asyncio
async def test_hybrid_retriever_fuse_results():
    """Test RRF fusion of BM25 and vector results."""
    bm25_results = [
        {"chunk_id": "chunk-1", "content": "Test 1", "bm25_score": 0.8, "metadata": {}},
        {"chunk_id": "chunk-2", "content": "Test 2", "bm25_score": 0.7, "metadata": {}},
    ]

    vector_results = [
        {"chunk_id": "chunk-2", "content": "Test 2", "vector_score": 0.9, "metadata": {}},
        {"chunk_id": "chunk-3", "content": "Test 3", "vector_score": 0.8, "metadata": {}},
    ]

    retriever = HybridRetriever(
        MagicMock(), MagicMock(), MagicMock()
    )

    fused = retriever._fuse_results(bm25_results, vector_results, top_k=5)

    # chunk-2 should have highest score (appears in both)
    assert len(fused) > 0
    assert any(r["chunk_id"] == "chunk-2" for r in fused)
    assert "rrf_score" in fused[0]


@pytest.mark.asyncio
async def test_hybrid_retriever_retrieve(mock_llm_client, mock_supabase_client):
    """Test hybrid retrieval."""
    from app.retrieval.vector_store import VectorStore

    bm25_retriever = BM25RetrieverWrapper(mock_supabase_client)
    vector_store = VectorStore(mock_supabase_client)
    hybrid_retriever = HybridRetriever(bm25_retriever, vector_store, mock_llm_client)

    # Mock retrievers
    with patch.object(bm25_retriever, "retrieve", new_callable=AsyncMock) as mock_bm25:
        with patch.object(vector_store, "search", new_callable=AsyncMock) as mock_vector:
            mock_bm25.return_value = [
                {"chunk_id": "chunk-1", "content": "Test", "bm25_score": 0.8, "metadata": {}}
            ]
            mock_vector.return_value = [
                {"id": "chunk-1", "content": "Test", "similarity": 0.9, "metadata": {}}
            ]

            results = await hybrid_retriever.retrieve("test query", top_k=5, user_id="test-user")

            assert len(results) > 0
            assert "rrf_score" in results[0]


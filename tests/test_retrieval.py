"""Tests for retrieval components."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.clients.supabase import SupabaseClient
from app.clients.llm import LLMServiceClient


@pytest.mark.asyncio
async def test_vector_store_search(mock_supabase_client):
    """Test vector store search."""
    vector_store = VectorStore(mock_supabase_client)

    query_embedding = [0.1] * 1536
    results = await vector_store.search(query_embedding, top_k=5)

    assert isinstance(results, list)
    mock_supabase_client.vector_search.assert_called_once()


@pytest.mark.asyncio
async def test_retriever_retrieve(mock_llm_client, mock_supabase_client):
    """Test retriever."""
    from app.retrieval.vector_store import VectorStore

    vector_store = VectorStore(mock_supabase_client)
    retriever = Retriever(vector_store, mock_llm_client)

    # Mock search results
    mock_supabase_client.vector_search = AsyncMock(
        return_value=[
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "Test content",
                "similarity": 0.9,
                "metadata": {"test": "data"},
            }
        ]
    )

    results = await retriever.retrieve("test query", top_k=5, user_id="test-user")

    assert len(results) > 0
    assert "content" in results[0]
    assert "similarity" in results[0]
    assert "metadata" in results[0]


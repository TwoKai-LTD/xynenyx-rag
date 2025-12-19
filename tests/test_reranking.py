"""Tests for reranker functionality."""
import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.reranker import Reranker


def test_reranker_rerank():
    """Test reranker functionality."""
    reranker = Reranker()

    documents = [
        {"content": "This is about AI and machine learning", "chunk_id": "chunk-1"},
        {"content": "Random content here", "chunk_id": "chunk-2"},
        {"content": "AI startup funding news", "chunk_id": "chunk-3"},
    ]

    query = "AI startup"

    with patch("app.retrieval.reranker.CrossEncoder") as mock_cross_encoder:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.8]  # Scores for each document
        mock_cross_encoder.return_value = mock_model

        reranker.model = mock_model
        reranker._model_loaded = True

        results = reranker.rerank(query, documents, top_k=2)

        assert len(results) == 2
        assert "rerank_score" in results[0]
        assert results[0]["rerank_score"] == 0.9  # Highest score first
        assert results[0]["chunk_id"] == "chunk-1"


def test_reranker_empty_documents():
    """Test reranker with empty documents."""
    reranker = Reranker()
    results = reranker.rerank("test query", [], top_k=5)
    assert results == []


def test_reranker_lazy_loading():
    """Test that reranker loads model lazily."""
    reranker = Reranker()
    assert reranker.model is None
    assert not reranker._model_loaded

    # Model should be loaded on first use
    with patch("app.retrieval.reranker.CrossEncoder") as mock_cross_encoder:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_cross_encoder.return_value = mock_model

        reranker.rerank("test", [{"content": "test", "chunk_id": "1"}], top_k=1)

        assert reranker._model_loaded is True
        assert reranker.model is not None


"""Cross-encoder reranker for search results."""
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for improving search result relevance."""

    def __init__(self):
        """Initialize reranker with lazy model loading."""
        self.model: Optional[CrossEncoder] = None
        self.model_name = settings.reranker_model
        self._model_loaded = False

    def _load_model(self) -> None:
        """Lazy load the cross-encoder model."""
        if self._model_loaded:
            return

        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self._model_loaded = True
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query text
            documents: List of documents to rerank (with content field)
            top_k: Number of top results to return

        Returns:
            Reranked documents with rerank_score added
        """
        if not documents:
            return []

        # Load model if not already loaded
        self._load_model()

        if not self.model:
            logger.warning("Reranker model not available, returning original results")
            return documents

        top_k = top_k or len(documents)

        try:
            # Create query-document pairs
            pairs = [(query, doc.get("content", "")) for doc in documents]

            # Score all pairs
            scores = self.model.predict(pairs)

            # Combine documents with scores
            scored_docs = list(zip(documents, scores))

            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Add rerank_score to results
            results = []
            for doc, score in scored_docs[:top_k]:
                result = {**doc, "rerank_score": float(score)}
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original results on error
            return documents[:top_k]


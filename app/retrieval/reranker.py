"""Cross-encoder reranker for search results."""
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from app.config import settings
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for improving search result relevance."""

    def __init__(self):
        """Initialize reranker with lazy model loading."""
        self.model: Optional[CrossEncoder] = None
        self.model_name = settings.reranker_model
        self._model_loaded = False
        self._model_load_failed = False
        self.cache_dir = self._get_cache_dir()

    def _get_cache_dir(self) -> str:
        """Get cache directory for reranker model."""
        if settings.reranker_cache_dir:
            cache_dir = settings.reranker_cache_dir
        else:
            # Default to user cache directory or /tmp as fallback
            home_dir = os.path.expanduser("~")
            if home_dir and os.access(home_dir, os.W_OK):
                cache_dir = str(Path(home_dir) / ".cache" / "sentence_transformers")
            else:
                cache_dir = "/tmp/sentence_transformers"
        
        # Ensure cache directory exists and is writable
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(cache_dir, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Using reranker cache directory: {cache_dir}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Cache directory {cache_dir} not writable, using /tmp: {e}")
            cache_dir = "/tmp/sentence_transformers"
            os.makedirs(cache_dir, exist_ok=True)
        
        return cache_dir

    def _load_model(self) -> None:
        """Lazy load the cross-encoder model."""
        if self._model_loaded or self._model_load_failed:
            return

        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            # Set cache directory environment variables for sentence_transformers and HuggingFace
            # This ensures all underlying libraries use the writable cache directory
            hf_cache_dir = os.path.join(os.path.dirname(self.cache_dir), "huggingface")
            os.makedirs(hf_cache_dir, exist_ok=True)
            
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.cache_dir
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
            os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
            
            self.model = CrossEncoder(self.model_name, cache_folder=self.cache_dir)
            self._model_loaded = True
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading reranker model: {e}. Reranking will be disabled.")
            self._model_load_failed = True
            self.model = None
            # Don't raise - allow service to continue without reranking

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

        if not self.model or self._model_load_failed:
            logger.debug("Reranker model not available, returning original results")
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


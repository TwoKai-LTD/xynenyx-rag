"""Hybrid retriever combining BM25 and vector search with RRF fusion."""
from typing import List, Dict, Any, Optional
from app.retrieval.bm25_retriever import BM25RetrieverWrapper
from app.retrieval.vector_store import VectorStore
from app.clients.llm import LLMServiceClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def rrf_score(rank: int, k: int = 60) -> float:
    """
    Calculate Reciprocal Rank Fusion score.

    Args:
        rank: Rank of the result (1-indexed)
        k: RRF k parameter (default 60)

    Returns:
        RRF score
    """
    return 1.0 / (k + rank)


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search with RRF fusion."""

    def __init__(
        self,
        bm25_retriever: BM25RetrieverWrapper,
        vector_store: VectorStore,
        llm_client: LLMServiceClient,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_retriever: BM25 retriever instance
            vector_store: Vector store instance
            llm_client: LLM service client for embedding generation
        """
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.rrf_k = settings.rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_bm25: bool = True,
        use_vector: bool = True,
        user_id: str = "rag-service",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search with RRF fusion.

        Args:
            query: Search query text
            top_k: Number of results to return
            use_bm25: Enable BM25 retrieval
            use_vector: Enable vector retrieval
            user_id: User ID for embedding generation

        Returns:
            List of fused retrieval results with:
                - content: Chunk content
                - rrf_score: Combined RRF score
                - bm25_score: BM25 score (if available)
                - vector_score: Vector similarity score (if available)
                - metadata: Chunk metadata
                - document_id: Document ID
                - chunk_id: Chunk ID
        """
        top_k = top_k or settings.default_top_k

        # Run retrievers in parallel
        bm25_results = []
        vector_results = []

        if use_bm25:
            try:
                bm25_results = await self.bm25_retriever.retrieve(query, top_k=top_k * 2)
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}")

        if use_vector:
            try:
                # Generate query embedding
                query_embedding = await self.llm_client.generate_embedding(query, user_id)

                # Perform vector search
                vector_results_raw = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2,
                )

                # Format vector results
                vector_results = [
                    {
                        "content": result.get("content", ""),
                        "vector_score": result.get("similarity", 0.0),
                        "metadata": result.get("metadata", {}),
                        "document_id": result.get("document_id"),
                        "chunk_id": result.get("id"),
                    }
                    for result in vector_results_raw
                ]
            except Exception as e:
                logger.warning(f"Vector retrieval failed: {e}")

        # Fuse results using RRF
        fused_results = self._fuse_results(bm25_results, vector_results, top_k)

        return fused_results

    def _fuse_results(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and vector results using Reciprocal Rank Fusion.

        Args:
            bm25_results: BM25 retrieval results
            vector_results: Vector retrieval results
            top_k: Number of results to return

        Returns:
            Fused and sorted results
        """
        # Create a map of chunk_id -> result data
        result_map: Dict[str, Dict[str, Any]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = str(result.get("chunk_id", ""))
            if not chunk_id:
                continue

            rrf = rrf_score(rank, self.rrf_k)

            if chunk_id not in result_map:
                result_map[chunk_id] = {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "document_id": result.get("document_id"),
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "bm25_score": result.get("bm25_score"),
                    "vector_score": None,
                }

            result_map[chunk_id]["rrf_score"] += rrf

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = str(result.get("chunk_id", ""))
            if not chunk_id:
                continue

            rrf = rrf_score(rank, self.rrf_k)

            if chunk_id not in result_map:
                result_map[chunk_id] = {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "document_id": result.get("document_id"),
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "bm25_score": None,
                    "vector_score": result.get("vector_score"),
                }

            result_map[chunk_id]["rrf_score"] += rrf
            if result_map[chunk_id]["vector_score"] is None:
                result_map[chunk_id]["vector_score"] = result.get("vector_score")

        # Sort by RRF score and return top-k
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return sorted_results[:top_k]


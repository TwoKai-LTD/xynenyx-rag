"""Vector retriever for semantic search (backward compatible)."""
from typing import List, Dict, Any, Optional
from uuid import UUID
from app.retrieval.vector_store import VectorStore
from app.clients.llm import LLMServiceClient
from app.config import settings


class Retriever:
    """
    Retriever for vector similarity search.

    This class maintains backward compatibility for vector-only search.
    For hybrid search, use HybridRetriever directly.
    """

    def __init__(self, vector_store: VectorStore, llm_client: LLMServiceClient):
        """
        Initialize retriever.

        Args:
            vector_store: Vector store instance
            llm_client: LLM service client for embedding generation
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.default_top_k = settings.default_top_k
        self.min_similarity_score = settings.min_similarity_score

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_document_ids: Optional[List[UUID]] = None,
        user_id: str = "rag-service",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query (vector-only).

        Args:
            query: Search query text
            top_k: Number of results to return (default from config)
            filter_document_ids: Optional list of document IDs to filter by
            user_id: User ID for embedding generation

        Returns:
            List of retrieval results with:
                - content: Chunk content
                - similarity: Similarity score
                - metadata: Chunk metadata
                - document_id: Document ID
                - chunk_id: Chunk ID
        """
        # Generate query embedding
        query_embedding = await self.llm_client.generate_embedding(query, user_id)

        # Perform vector search
        top_k = top_k or self.default_top_k
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_document_ids=filter_document_ids,
        )

        # Filter by minimum similarity score
        filtered_results = [
            result
            for result in results
            if result.get("similarity", 0.0) >= self.min_similarity_score
        ]

        # Format results
        formatted_results = []
        for result in filtered_results:
            formatted_results.append(
                {
                    "content": result.get("content", ""),
                    "similarity": result.get("similarity", 0.0),
                    "vector_score": result.get("similarity", 0.0),  # Add for consistency
                    "metadata": result.get("metadata", {}),
                    "document_id": result.get("document_id"),
                    "chunk_id": result.get("id"),
                }
            )

        return formatted_results


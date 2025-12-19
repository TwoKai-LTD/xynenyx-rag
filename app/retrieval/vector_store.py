"""Vector store for Supabase pgvector integration."""
from typing import List, Dict, Any, Optional
from uuid import UUID
from app.clients.supabase import SupabaseClient


class VectorStore:
    """Vector store wrapper for Supabase pgvector."""

    def __init__(self, supabase_client: SupabaseClient):
        """
        Initialize vector store.

        Args:
            supabase_client: Supabase client instance
        """
        self.supabase_client = supabase_client

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_document_ids: Optional[List[UUID]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector (1536 dimensions)
            top_k: Number of results to return
            filter_document_ids: Optional list of document IDs to filter by

        Returns:
            List of search results with:
                - id: Chunk ID
                - document_id: Document ID
                - content: Chunk content
                - metadata: Chunk metadata
                - similarity: Similarity score (0-1)
        """
        results = await self.supabase_client.vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_document_ids=filter_document_ids,
        )

        return results


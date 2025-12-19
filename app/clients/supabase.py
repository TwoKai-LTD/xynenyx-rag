"""Supabase client wrapper for documents and chunks operations."""
from typing import List, Dict, Any, Optional
from uuid import UUID
from supabase import create_client, Client
from app.config import settings


class SupabaseClient:
    """Supabase client wrapper for RAG operations."""

    def __init__(self):
        """Initialize Supabase client."""
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )

    async def create_document(
        self,
        user_id: str,
        name: str,
        s3_key: str,
        content_type: str = "text/html",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a document record.

        Args:
            user_id: User ID
            name: Document name (article title)
            s3_key: S3 key (or placeholder for RSS)
            content_type: Content type
            metadata: Additional metadata

        Returns:
            Created document record
        """
        result = (
            self.client.table("documents")
            .insert(
                {
                    "user_id": user_id,
                    "name": name,
                    "s3_key": s3_key,
                    "content_type": content_type,
                    "status": "pending",
                    "metadata": metadata or {},
                }
            )
            .execute()
        )
        return result.data[0] if result.data else {}

    async def update_document_status(
        self,
        document_id: UUID,
        status: str,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update document status.

        Args:
            document_id: Document ID
            status: New status (pending, processing, ready, error)
            chunk_count: Number of chunks (optional)
            error_message: Error message if status is error
        """
        update_data: Dict[str, Any] = {"status": status}
        if chunk_count is not None:
            update_data["chunk_count"] = chunk_count
        if error_message:
            update_data["error_message"] = error_message

        self.client.table("documents").update(update_data).eq("id", str(document_id)).execute()

    async def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Insert document chunks in batch.

        Args:
            chunks: List of chunk dictionaries with:
                - document_id
                - chunk_index
                - content
                - embedding (list[float])
                - token_count
                - metadata
        """
        if not chunks:
            return

        # Convert embeddings to string format for Supabase
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunk = {
                "document_id": str(chunk["document_id"]),
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "embedding": chunk.get("embedding"),  # Supabase handles vector conversion
                "token_count": chunk.get("token_count"),
                "metadata": chunk.get("metadata", {}),
            }
            formatted_chunks.append(formatted_chunk)

        # Batch insert (Supabase handles batching)
        self.client.table("document_chunks").insert(formatted_chunks).execute()

    async def vector_search(
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
            List of search results with content, similarity, metadata
        """
        # Convert UUIDs to strings if provided
        filter_ids = [str(doc_id) for doc_id in filter_document_ids] if filter_document_ids else None

        result = self.client.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "filter_document_ids": filter_ids,
            },
        ).execute()

        return result.data if result.data else []

    async def get_document(self, document_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document record or None
        """
        result = self.client.table("documents").select("*").eq("id", str(document_id)).execute()
        return result.data[0] if result.data else None

    async def list_documents(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List documents with optional filters.

        Args:
            user_id: Filter by user ID
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of document records
        """
        query = self.client.table("documents").select("*")

        if user_id:
            query = query.eq("user_id", user_id)
        if status:
            query = query.eq("status", status)

        query = query.order("created_at", desc=True).limit(limit).offset(offset)

        result = query.execute()
        return result.data if result.data else []


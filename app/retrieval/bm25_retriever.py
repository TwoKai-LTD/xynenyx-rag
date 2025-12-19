"""BM25 keyword-based retriever using rank-bm25."""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.clients.supabase import SupabaseClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class BM25RetrieverWrapper:
    """BM25 retriever wrapper that builds index from Supabase chunks."""

    def __init__(self, supabase_client: SupabaseClient):
        """
        Initialize BM25 retriever.

        Args:
            supabase_client: Supabase client instance
        """
        self.supabase_client = supabase_client
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        self._index_built = False

    async def build_index(self, user_id: Optional[str] = None) -> None:
        """
        Build BM25 index from document chunks in Supabase.

        Args:
            user_id: Optional user ID to filter chunks
        """
        try:
            logger.info("Building BM25 index from Supabase chunks...")

            # Load all chunks from Supabase
            chunks = await self._load_chunks_from_supabase(user_id)

            if not chunks:
                logger.warning("No chunks found to build BM25 index")
                self._index_built = False
                return

            # Store chunks for retrieval
            self.chunks = chunks

            # Tokenize chunks for BM25
            tokenized_corpus = []
            for chunk in chunks:
                # Simple tokenization (split on whitespace and lowercase)
                tokens = chunk["content"].lower().split()
                tokenized_corpus.append(tokens)

            # Build BM25 index
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._index_built = True
            logger.info(f"BM25 index built with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self._index_built = False
            raise

    async def _load_chunks_from_supabase(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load all chunks from Supabase.

        Args:
            user_id: Optional user ID filter

        Returns:
            List of chunk dictionaries
        """
        # Query all chunks with embeddings
        # Note: This is a simplified approach - for production, consider streaming/pagination
        query = self.supabase_client.client.table("document_chunks").select(
            "id, document_id, content, metadata"
        )

        # Filter by user if provided (via documents table join)
        if user_id:
            # This requires a join - for now, we'll load all chunks
            # In production, add a helper function to filter by user_id
            pass

        result = query.execute()
        return result.data if result.data else []

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 keyword search.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of retrieval results with:
                - content: Chunk content
                - bm25_score: BM25 relevance score
                - metadata: Chunk metadata
                - document_id: Document ID
                - chunk_id: Chunk ID
        """
        if not self._index_built or not self.bm25:
            await self.build_index()

        if not self.bm25 or not self.chunks:
            logger.warning("BM25 retriever not available, returning empty results")
            return []

        top_k = top_k or settings.bm25_top_k

        try:
            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Create list of (chunk, score) tuples
            scored_chunks = list(zip(self.chunks, scores))

            # Sort by score (descending)
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Format results
            results = []
            for chunk, score in scored_chunks[:top_k]:
                results.append(
                    {
                        "content": chunk.get("content", ""),
                        "bm25_score": float(score),
                        "metadata": chunk.get("metadata", {}),
                        "document_id": chunk.get("document_id"),
                        "chunk_id": str(chunk.get("id", "")),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []

    def refresh_index(self) -> None:
        """Mark index as needing refresh."""
        self._index_built = False

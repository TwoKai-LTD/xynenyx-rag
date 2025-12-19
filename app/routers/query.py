"""API routes for vector search queries."""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.queries import QueryRequest, QueryResponse, QueryResult
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.clients.supabase import SupabaseClient
from app.clients.llm import LLMServiceClient

router = APIRouter(prefix="/query", tags=["query"])

# Initialize retriever (singleton pattern)
_supabase_client = SupabaseClient()
_vector_store = VectorStore(_supabase_client)
_llm_client = LLMServiceClient()
_retriever = Retriever(_vector_store, _llm_client)


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Perform vector similarity search.

    Args:
        request: Query request with search text and parameters
        x_user_id: User ID from header

    Returns:
        Query results with content, similarity scores, and metadata
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")

    try:
        # Retrieve relevant chunks
        results = await _retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_document_ids=request.filter_document_ids,
            user_id=x_user_id,
        )

        # Format results
        query_results = [
            QueryResult(
                content=result["content"],
                similarity=result["similarity"],
                metadata=result["metadata"],
                document_id=str(result["document_id"]),
                chunk_id=str(result["chunk_id"]),
            )
            for result in results
        ]

        return QueryResponse(
            query=request.query,
            results=query_results,
            count=len(query_results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}") from e


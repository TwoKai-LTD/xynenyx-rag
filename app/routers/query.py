"""API routes for vector search queries with hybrid search and reranking."""
import logging
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.queries import QueryRequest, QueryResponse, QueryResult
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_retriever import BM25RetrieverWrapper
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.filters import TemporalFilter, EntityFilter
from app.clients.supabase import SupabaseClient
from app.clients.llm import LLMServiceClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

# Initialize components (singleton pattern)
_supabase_client = SupabaseClient()
_vector_store = VectorStore(_supabase_client)
_llm_client = LLMServiceClient()
_bm25_retriever = BM25RetrieverWrapper(_supabase_client)
_hybrid_retriever = HybridRetriever(_bm25_retriever, _vector_store, _llm_client)
_vector_only_retriever = Retriever(_vector_store, _llm_client)
_reranker = Reranker()
_temporal_filter = TemporalFilter()
_entity_filter = EntityFilter()


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Perform hybrid search with optional reranking and filtering.

    Args:
        request: Query request with search text and parameters
        x_user_id: User ID from header

    Returns:
        Query results with content, scores, and metadata
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")

    try:
        search_mode = "vector"
        results = []

        # Determine search mode and retrieve
        if request.use_hybrid_search:
            # Hybrid search (BM25 + Vector with RRF)
            search_mode = "hybrid"
            results = await _hybrid_retriever.retrieve(
                query=request.query,
                top_k=request.top_k * 2,  # Get more results for reranking
                use_bm25=True,
                use_vector=True,
                user_id=x_user_id,
            )
        else:
            # Vector-only search (backward compatibility)
            search_mode = "vector"
            results = await _vector_only_retriever.retrieve(
                query=request.query,
                top_k=request.top_k * 2,  # Get more results for reranking
                filter_document_ids=request.filter_document_ids,
                user_id=x_user_id,
            )

        # Apply temporal filter
        if request.date_filter:
            date_range = _temporal_filter.parse_filter(request.date_filter)
            if date_range:
                results = _temporal_filter.filter_results(results, date_range)

        # Apply entity filters
        results = _entity_filter.filter_results(
            results,
            company_filter=request.company_filter,
            investor_filter=request.investor_filter,
            sector_filter=request.sector_filter,
        )

        # Rerank if enabled
        reranking_enabled = False
        if request.use_reranking and results:
            try:
                # Rerank top N results
                results_to_rerank = results[: request.rerank_top_n]
                reranked = _reranker.rerank(
                    query=request.query,
                    documents=results_to_rerank,
                    top_k=request.top_k,
                )
                # Combine reranked with remaining results
                reranked_ids = {r.get("chunk_id") for r in reranked}
                remaining = [r for r in results[request.rerank_top_n :] if r.get("chunk_id") not in reranked_ids]
                results = reranked + remaining
                reranking_enabled = True
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
                # Continue with original results (reranking is optional)
                reranking_enabled = False

        # Take top-k final results
        results = results[: request.top_k]

        # Format results
        query_results = []
        for result in results:
            # Determine similarity score (prefer normalized scores, clamp to 0-1)
            # Priority: rerank_score (0-1) > vector_score (0-1) > normalized rrf_score > similarity
            raw_similarity = (
                result.get("rerank_score")
                or result.get("vector_score")
                or result.get("similarity")
                or result.get("rrf_score")
                or 0.0
            )
            
            # Normalize similarity to 0-1 range (clamp if > 1, ensure >= 0)
            # RRF and BM25 scores can be > 1, so we normalize them
            if raw_similarity > 1.0:
                # Use sigmoid-like normalization for scores > 1, or simple clamp
                # For now, use simple clamp to 1.0 to maintain relative ordering
                similarity = min(1.0, raw_similarity / (1.0 + raw_similarity)) if raw_similarity > 1.0 else raw_similarity
            else:
                similarity = max(0.0, raw_similarity)

            query_results.append(
                QueryResult(
                    content=result.get("content", ""),
                    similarity=similarity,
                    metadata=result.get("metadata", {}),
                    document_id=str(result.get("document_id", "")),
                    chunk_id=str(result.get("chunk_id", "")),
                    rerank_score=result.get("rerank_score"),
                    bm25_score=result.get("bm25_score"),
                    vector_score=result.get("vector_score") or result.get("similarity"),
                    rrf_score=result.get("rrf_score"),
                )
            )

        return QueryResponse(
            query=request.query,
            results=query_results,
            count=len(query_results),
            search_mode=search_mode,
            reranking_enabled=reranking_enabled,
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}") from e

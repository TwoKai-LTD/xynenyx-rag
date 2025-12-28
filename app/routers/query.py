"""API routes for vector search queries with hybrid search and reranking."""
import logging
import math
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.queries import QueryRequest, QueryResponse, QueryResult
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_retriever import BM25RetrieverWrapper
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.filters import TemporalFilter, EntityFilter
from app.clients.supabase import SupabaseClient
from app.clients.llm import LLMServiceClient
from app.services.cache import QueryCache

logger = logging.getLogger(__name__)


def sanitize_float(value: float) -> float:
    """Sanitize float values to ensure JSON compliance.
    
    Converts inf, -inf, and nan to valid float values.
    """
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    return value

router = APIRouter(prefix="/query", tags=["query"])

# Initialize components (singleton pattern)
_supabase_client = SupabaseClient()
_vector_store = VectorStore(_supabase_client)
_llm_client = LLMServiceClient()
_bm25_retriever = BM25RetrieverWrapper(_supabase_client)
_hybrid_retriever = HybridRetriever(_bm25_retriever, _vector_store, _llm_client)
_multi_query_retriever = MultiQueryRetriever(_hybrid_retriever, _llm_client)
_vector_only_retriever = Retriever(_vector_store, _llm_client)
_reranker = Reranker()
_temporal_filter = TemporalFilter()
_entity_filter = EntityFilter()
_query_cache = QueryCache(ttl_seconds=3600)  # 1 hour cache


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
        # Check cache first
        cache_filters = {}
        if request.date_filter:
            cache_filters["date_filter"] = request.date_filter
        if request.company_filter:
            cache_filters["company_filter"] = request.company_filter
        if request.investor_filter:
            cache_filters["investor_filter"] = request.investor_filter
        if request.sector_filter:
            cache_filters["sector_filter"] = request.sector_filter
        
        cached_results = _query_cache.get(request.query, cache_filters if cache_filters else None)
        if cached_results:
            logger.info("Returning cached query results")
            return QueryResponse(
                query=request.query,
                results=cached_results.get("results", []),
                count=cached_results.get("count", 0),
                search_mode=cached_results.get("search_mode", "vector"),
                reranking_enabled=cached_results.get("reranking_enabled", False),
            )

        search_mode = "vector"
        results = []

        # Determine search mode and retrieve
        if request.use_multi_query:
            # Multi-query retrieval (generates variations and merges results)
            search_mode = "multi_query"
            results = await _multi_query_retriever.retrieve(
                query=request.query,
                top_k=request.top_k * 2,  # Get more results for reranking
                query_variations=request.query_variations,
                user_id=x_user_id,
            )
        elif request.use_hybrid_search:
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
            # Determine similarity score (use highest available)
            # Priority: rerank_score (0-1) > rrf_score (may be > 1) > vector_score (0-1) > similarity (may be negative)
            raw_similarity = (
                result.get("rerank_score")
                or result.get("rrf_score")
                or result.get("vector_score")
                or result.get("similarity")
                or 0.0
            )
            
            # Sanitize and normalize similarity to 0-1 range
            # Handle inf, -inf, nan, negative values, and values > 1
            raw_float = float(raw_similarity)
            sanitized = sanitize_float(raw_float)
            similarity = max(0.0, min(1.0, sanitized))

            # Sanitize all score fields to ensure JSON compliance
            rerank_score = sanitize_float(float(result.get("rerank_score"))) if result.get("rerank_score") is not None else None
            bm25_score = sanitize_float(float(result.get("bm25_score"))) if result.get("bm25_score") is not None else None
            vector_score_raw = result.get("vector_score") or result.get("similarity")
            vector_score = sanitize_float(float(vector_score_raw)) if vector_score_raw is not None else None
            rrf_score = sanitize_float(float(result.get("rrf_score"))) if result.get("rrf_score") is not None else None

            query_results.append(
                QueryResult(
                    content=result.get("content", ""),
                    similarity=similarity,
                    metadata=result.get("metadata", {}),
                    document_id=str(result.get("document_id", "")),
                    chunk_id=str(result.get("chunk_id", "")),
                    rerank_score=rerank_score,
                    bm25_score=bm25_score,
                    vector_score=vector_score,
                    rrf_score=rrf_score,
                )
            )

        response = QueryResponse(
            query=request.query,
            results=query_results,
            count=len(query_results),
            search_mode=search_mode,
            reranking_enabled=reranking_enabled,
        )
        
        # Cache results
        _query_cache.set(
            request.query,
            {
                "results": query_results,
                "count": len(query_results),
                "search_mode": search_mode,
                "reranking_enabled": reranking_enabled,
            },
            cache_filters if cache_filters else None,
        )
        
        return response
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}") from e

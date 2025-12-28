"""Multi-query retrieval for improved recall."""
import logging
import json
from typing import List, Dict, Any, Optional
from app.retrieval.hybrid_retriever import HybridRetriever
from app.clients.llm import LLMServiceClient
from app.config import settings

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """Retriever that uses multiple query variations to improve recall."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        llm_client: LLMServiceClient,
    ):
        """
        Initialize multi-query retriever.

        Args:
            hybrid_retriever: Hybrid retriever for individual queries
            llm_client: LLM client for generating query variations
        """
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_variations: Optional[List[str]] = None,
        user_id: str = "rag-service",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multiple query variations.

        Args:
            query: Original query
            top_k: Number of results to return
            query_variations: Optional list of query variations (if not provided, will generate)
            user_id: User ID for embedding generation

        Returns:
            List of merged retrieval results with deduplication
        """
        top_k = top_k or settings.default_top_k

        # Generate query variations if not provided
        if query_variations is None:
            query_variations = await self._generate_query_variations(query, user_id)
        
        # Include original query
        if query not in query_variations:
            query_variations.insert(0, query)
        else:
            # Move original to front if it exists
            query_variations.remove(query)
            query_variations.insert(0, query)

        logger.info(f"Retrieving with {len(query_variations)} query variations: {query_variations}")

        # Retrieve for each query variation
        all_results = []
        seen_chunk_ids = set()

        for q in query_variations:
            try:
                # Retrieve with hybrid search for each variation
                # Use top_k * 2 to get more candidates for merging
                results = await self.hybrid_retriever.retrieve(
                    query=q,
                    top_k=top_k * 2,
                    use_bm25=True,
                    use_vector=True,
                    user_id=user_id,
                )

                # Deduplicate by chunk_id and add to all_results
                for result in results:
                    chunk_id = result.get("chunk_id")
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        # Add query variation info to metadata
                        result["query_variation"] = q
                        all_results.append(result)

            except Exception as e:
                logger.warning(f"Retrieval failed for query variation '{q}': {e}")
                continue

        # Re-rank merged results by RRF score (if available) or similarity
        # This helps prioritize results that appeared in multiple query variations
        all_results.sort(
            key=lambda x: (
                x.get("rrf_score", 0) or x.get("vector_score", 0) or x.get("similarity", 0)
            ),
            reverse=True,
        )

        # Take top-k final results
        final_results = all_results[:top_k]

        logger.info(f"Multi-query retrieval returned {len(final_results)} results from {len(query_variations)} query variations")

        return final_results

    async def _generate_query_variations(
        self,
        query: str,
        user_id: str,
    ) -> List[str]:
        """
        Generate query variations using LLM.

        Args:
            query: Original query
            user_id: User ID for usage tracking

        Returns:
            List of query variations
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a query rewriting assistant for a startup/VC research system.

TASK: Generate 3-5 search query variations that would help find relevant articles about startups, funding, companies, and investors.

GUIDELINES:
- Keep the core intent of the original query
- Expand with synonyms and related terms (e.g., "AI" → "artificial intelligence", "startup" → "company" or "venture")
- Add domain-specific terms when relevant (e.g., "funding" → "funding round" or "venture capital")
- Include variations that might appear in article titles or content
- Make queries more specific when the original is vague

OUTPUT FORMAT:
Return a JSON object with a "queries" array containing 3-5 query strings.
Example: {"queries": ["query 1", "query 2", "query 3"]}""",
                },
                {
                    "role": "user",
                    "content": f"Original query: {query}\n\nGenerate 3-5 query variations that would find relevant startup/VC articles.",
                },
            ]

            # Use JSON mode for structured output
            response = await self.llm_client.complete(
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                user_id=user_id,
            )

            import json
            content = response.get("content", "").strip()
            
            # Parse JSON response
            try:
                parsed = json.loads(content)
                queries = parsed.get("queries", [])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse query variations JSON: {e}. Content: {content}")
                queries = []

            # Validate queries
            if not queries or len(queries) == 0:
                logger.warning("No queries generated, using original query")
                return [query]

            # Limit to 5 queries
            return queries[:5]

        except Exception as e:
            logger.error(f"Query variation generation failed: {e}", exc_info=True)
            return [query]


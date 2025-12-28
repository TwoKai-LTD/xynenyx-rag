"""Caching service for RAG embeddings and queries."""
import hashlib
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, ttl_seconds: int = 86400 * 7):  # 7 days default
        """
        Initialize embedding cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            text: Text to get embedding for

        Returns:
            Embedding vector or None if not cached or expired
        """
        cache_key = self._get_cache_key(text)
        entry = self.cache.get(cache_key)

        if not entry:
            return None

        # Check if expired
        if datetime.now() - entry["timestamp"] > timedelta(seconds=self.ttl_seconds):
            del self.cache[cache_key]
            return None

        return entry["embedding"]

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        cache_key = self._get_cache_key(text)
        self.cache[cache_key] = {
            "embedding": embedding,
            "timestamp": datetime.now(),
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)


class QueryCache:
    """Simple in-memory cache for query results."""

    def __init__(self, ttl_seconds: int = 3600):  # 1 hour default
        """
        Initialize query cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def _get_cache_key(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate cache key from query and filters."""
        cache_data = {"query": query}
        if filters:
            cache_data.update(filters)
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get query results from cache.

        Args:
            query: Search query
            filters: Optional filters (date, company, etc.)

        Returns:
            Cached results or None if not cached or expired
        """
        cache_key = self._get_cache_key(query, filters)
        entry = self.cache.get(cache_key)

        if not entry:
            return None

        # Check if expired
        if datetime.now() - entry["timestamp"] > timedelta(seconds=self.ttl_seconds):
            del self.cache[cache_key]
            return None

        return entry["results"]

    def set(
        self,
        query: str,
        results: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store query results in cache.

        Args:
            query: Search query
            results: Query results to cache
            filters: Optional filters used
        """
        cache_key = self._get_cache_key(query, filters)
        self.cache[cache_key] = {
            "results": results,
            "timestamp": datetime.now(),
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Query cache cleared")

    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)


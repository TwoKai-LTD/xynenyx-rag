"""LLM service client for embedding generation."""
import asyncio
from typing import List, Optional
import httpx
from app.config import settings


class LLMServiceClient:
    """HTTP client for LLM service embedding generation."""

    def __init__(self):
        """Initialize LLM service client."""
        self.base_url = settings.llm_service_url
        self.timeout = settings.llm_service_timeout
        self.batch_size = settings.embedding_batch_size
        self.max_retries = settings.embedding_max_retries
        self.retry_delay = settings.embedding_retry_delay

    async def generate_embedding(
        self,
        text: str,
        user_id: str = "rag-service",
        retry_count: int = 0,
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            user_id: User ID for usage tracking
            retry_count: Current retry attempt

        Returns:
            Embedding vector (1536 dimensions)

        Raises:
            Exception: If embedding generation fails after retries
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"text": text, "provider": "openai"},
                    headers={"X-User-ID": user_id},
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
            except Exception as e:
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))
                    return await self.generate_embedding(text, user_id, retry_count + 1)
                raise ValueError(f"Failed to generate embedding after {self.max_retries} retries: {str(e)}") from e

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        user_id: str = "rag-service",
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed
            user_id: User ID for usage tracking

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Generate embeddings concurrently for the batch
            tasks = [self.generate_embedding(text, user_id) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any failures in the batch
            embeddings = []
            for emb in batch_embeddings:
                if isinstance(emb, Exception):
                    # Log error but continue with other embeddings
                    print(f"Error generating embedding: {emb}")
                    # Use zero vector as fallback (or skip)
                    embeddings.append([0.0] * 1536)
                else:
                    embeddings.append(emb)

            all_embeddings.extend(embeddings)

            # Small delay between batches to avoid rate limits
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)

        return all_embeddings


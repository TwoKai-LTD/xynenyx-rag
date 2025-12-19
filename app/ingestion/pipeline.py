"""RSS ingestion pipeline orchestrator."""
from typing import List, Dict, Any, Optional
from uuid import UUID
from app.ingestion.rss_parser import RSSParser
from app.ingestion.html_parser import HTMLParser
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.chunkers import Chunker
from app.clients.llm import LLMServiceClient
from app.clients.supabase import SupabaseClient


class IngestionPipeline:
    """Orchestrates the full RSS ingestion workflow."""

    def __init__(self):
        """Initialize ingestion pipeline components."""
        self.rss_parser = RSSParser()
        self.html_parser = HTMLParser()
        self.metadata_extractor = MetadataExtractor()
        self.chunker = Chunker()
        self.llm_client = LLMServiceClient()
        self.supabase_client = SupabaseClient()

    async def ingest_feed(
        self,
        feed_url: str,
        feed_name: str,
        user_id: str,
        feed_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest an RSS feed: parse, extract, chunk, embed, and store.

        Args:
            feed_url: RSS feed URL
            feed_name: Feed name
            user_id: User ID for document ownership
            feed_id: Optional feed ID for S3 key generation

        Returns:
            Dictionary with ingestion results
        """
        articles_ingested = 0
        articles_failed = 0
        errors = []

        try:
            # 1. Parse RSS feed
            feed_data = self.rss_parser.parse_feed(feed_url)
            entries = feed_data.get("entries", [])

            if not entries:
                return {
                    "feed_url": feed_url,
                    "feed_name": feed_name,
                    "articles_ingested": 0,
                    "articles_failed": 0,
                    "status": "completed",
                    "message": "No new articles found",
                }

            # 2. Process each article
            for entry in entries:
                try:
                    await self._process_article(entry, feed_name, feed_url, user_id, feed_id)
                    articles_ingested += 1
                except Exception as e:
                    articles_failed += 1
                    error_msg = f"Failed to process article {entry.get('link', 'unknown')}: {str(e)}"
                    errors.append(error_msg)
                    print(error_msg)
                    continue

            return {
                "feed_url": feed_url,
                "feed_name": feed_name,
                "articles_ingested": articles_ingested,
                "articles_failed": articles_failed,
                "status": "completed",
                "errors": errors[:10],  # Limit error list
            }

        except Exception as e:
            return {
                "feed_url": feed_url,
                "feed_name": feed_name,
                "articles_ingested": articles_ingested,
                "articles_failed": articles_failed,
                "status": "error",
                "error": str(e),
                "errors": errors,
            }

    async def _process_article(
        self,
        entry: Dict[str, Any],
        feed_name: str,
        feed_url: str,
        user_id: str,
        feed_id: Optional[str],
    ) -> None:
        """
        Process a single article through the full pipeline.

        Args:
            entry: RSS entry dictionary
            feed_name: Feed name
            feed_url: Feed URL
            user_id: User ID
            feed_id: Optional feed ID
        """
        article_url = entry.get("link", "")
        article_title = entry.get("title", "Untitled")

        # Generate S3 key placeholder for RSS feeds
        article_id = entry.get("id", article_url.split("/")[-1])
        s3_key = f"rss://{feed_id or 'default'}/{article_id}"

        # 1. Create document record
        document = await self.supabase_client.create_document(
            user_id=user_id,
            name=article_title,
            s3_key=s3_key,
            content_type="text/html",
            metadata={
                "feed_name": feed_name,
                "feed_url": feed_url,
                "article_url": article_url,
                "published_date": entry.get("published_date"),
            },
        )
        document_id = UUID(document["id"])

        try:
            # Update status to processing
            await self.supabase_client.update_document_status(document_id, "processing")

            # 2. Fetch and extract HTML content
            content = await self.html_parser.extract_content(article_url)
            if not content:
                # Fallback to description if HTML extraction fails
                content = entry.get("description", "")
                if not content:
                    raise ValueError("No content available for article")

            # 3. Extract metadata
            article_metadata = {
                "feed_name": feed_name,
                "feed_url": feed_url,
                "article_url": article_url,
                "title": article_title,
                "published_date": entry.get("published_date"),
            }
            metadata = self.metadata_extractor.extract(content, article_metadata)

            # 4. Chunk content
            chunks = self.chunker.chunk_document(content, metadata)

            if not chunks:
                raise ValueError("No chunks generated from content")

            # 5. Generate embeddings
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.llm_client.generate_embeddings_batch(chunk_texts, user_id)

            if len(embeddings) != len(chunks):
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")

            # 6. Prepare chunks for storage
            chunks_to_store = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunks_to_store.append(
                    {
                        "document_id": document_id,
                        "chunk_index": idx,
                        "content": chunk["content"],
                        "embedding": embedding,
                        "token_count": chunk["token_count"],
                        "metadata": chunk["metadata"],
                    }
                )

            # 7. Store chunks
            await self.supabase_client.insert_chunks(chunks_to_store)

            # 8. Update document status
            await self.supabase_client.update_document_status(
                document_id, "ready", chunk_count=len(chunks)
            )

        except Exception as e:
            # Update document status to error
            await self.supabase_client.update_document_status(
                document_id, "error", error_message=str(e)
            )
            raise


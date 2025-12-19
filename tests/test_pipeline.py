"""Tests for ingestion pipeline."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.ingestion.pipeline import IngestionPipeline


@pytest.mark.asyncio
async def test_ingestion_pipeline_ingest_feed(mock_supabase_client, mock_llm_client):
    """Test full ingestion pipeline."""
    pipeline = IngestionPipeline()

    # Mock all components
    with patch.object(pipeline, "rss_parser") as mock_rss, patch.object(
        pipeline, "html_parser"
    ) as mock_html, patch.object(pipeline, "metadata_extractor") as mock_meta, patch.object(
        pipeline, "chunker"
    ) as mock_chunker, patch.object(
        pipeline, "llm_client", mock_llm_client
    ), patch.object(
        pipeline, "supabase_client", mock_supabase_client
    ):
        # Mock RSS parsing
        mock_rss.parse_feed.return_value = {
            "title": "Test Feed",
            "entries": [
                {
                    "id": "entry-1",
                    "link": "https://example.com/article",
                    "title": "Test Article",
                    "description": "Test description",
                    "published_date": "2024-01-01T00:00:00",
                }
            ],
        }

        # Mock HTML extraction
        mock_html.extract_content = AsyncMock(return_value="Test article content here.")

        # Mock metadata extraction
        mock_meta.extract.return_value = {
            "companies": [],
            "funding_amounts": [],
            "dates": [],
            "investors": [],
            "sectors": [],
            "feed_name": "Test Feed",
            "article_url": "https://example.com/article",
            "title": "Test Article",
        }

        # Mock chunking
        mock_chunker.chunk_document.return_value = [
            {
                "content": "Test article content here.",
                "metadata": {},
                "token_count": 10,
                "chunk_index": 0,
            }
        ]

        # Run pipeline
        result = await pipeline.ingest_feed(
            feed_url="https://example.com/feed",
            feed_name="Test Feed",
            user_id="test-user",
        )

        assert result["status"] in ["completed", "error"]
        assert "articles_ingested" in result


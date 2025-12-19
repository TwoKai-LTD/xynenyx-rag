"""Tests for ingestion components."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.ingestion.rss_parser import RSSParser
from app.ingestion.html_parser import HTMLParser
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.chunkers import Chunker


def test_rss_parser_parse_feed(sample_rss_feed):
    """Test RSS feed parsing."""
    parser = RSSParser()

    with patch("app.ingestion.rss_parser.feedparser") as mock_feedparser:
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.feed = {
            "title": sample_rss_feed["title"],
            "link": sample_rss_feed["link"],
            "description": sample_rss_feed["description"],
        }
        mock_entry = MagicMock()
        mock_entry.get = lambda key, default=None: sample_rss_feed["entries"][0].get(key, default)
        mock_entry.id = sample_rss_feed["entries"][0]["id"]
        mock_entry.link = sample_rss_feed["entries"][0]["link"]
        mock_entry.title = sample_rss_feed["entries"][0]["title"]
        mock_entry.description = sample_rss_feed["entries"][0]["description"]
        mock_entry.published_parsed = None
        mock_feed.entries = [mock_entry]
        mock_feedparser.parse.return_value = mock_feed

        result = parser.parse_feed("https://example.com/feed")
        assert result["title"] == sample_rss_feed["title"]
        assert len(result["entries"]) == 1


@pytest.mark.asyncio
async def test_html_parser_extract_content(sample_html_content):
    """Test HTML content extraction."""
    parser = HTMLParser()

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.text = sample_html_content
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        content = await parser.extract_content("https://example.com/article")
        assert content is not None
        assert "Test Article" in content
        assert "test content" in content.lower()


def test_metadata_extractor_extract(sample_metadata):
    """Test metadata extraction."""
    extractor = MetadataExtractor()
    content = """
    Test Company announced a $10 million funding round led by Test Investor.
    The company operates in the AI sector.
    """
    article_metadata = {
        "feed_name": "Test Feed",
        "article_url": "https://example.com/article",
        "title": "Test Article",
    }

    metadata = extractor.extract(content, article_metadata)
    assert "companies" in metadata
    assert "funding_amounts" in metadata
    assert "investors" in metadata
    assert "sectors" in metadata
    assert metadata["feed_name"] == "Test Feed"


def test_chunker_chunk_document():
    """Test document chunking."""
    chunker = Chunker()
    text = "This is a test document. " * 100  # Create longer text
    metadata = {"test": "metadata"}

    chunks = chunker.chunk_document(text, metadata)
    assert len(chunks) > 0
    assert all("content" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
    assert all("token_count" in chunk for chunk in chunks)
    assert all("chunk_index" in chunk for chunk in chunks)


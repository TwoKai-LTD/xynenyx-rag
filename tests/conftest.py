"""Pytest configuration and fixtures."""
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client."""
    mock = AsyncMock()
    mock.create_document = AsyncMock(
        return_value={
            "id": "test-doc-id",
            "user_id": "test-user-id",
            "name": "Test Document",
            "status": "pending",
        }
    )
    mock.update_document_status = AsyncMock()
    mock.insert_chunks = AsyncMock()
    mock.vector_search = AsyncMock(return_value=[])
    mock.get_document = AsyncMock(return_value=None)
    mock.list_documents = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_llm_client():
    """Mock LLM service client."""
    mock = AsyncMock()
    # Return a 1536-dimensional embedding vector
    mock_embedding = [0.1] * 1536
    mock.generate_embedding = AsyncMock(return_value=mock_embedding)
    mock.generate_embeddings_batch = AsyncMock(
        return_value=[mock_embedding, mock_embedding]
    )
    return mock


@pytest.fixture
def sample_rss_feed():
    """Sample RSS feed data."""
    return {
        "title": "Test Feed",
        "link": "https://example.com",
        "description": "Test feed description",
        "entries": [
            {
                "id": "entry-1",
                "link": "https://example.com/article-1",
                "title": "Test Article 1",
                "description": "Article 1 description",
                "published_date": "2024-01-01T00:00:00",
            }
        ],
    }


@pytest.fixture
def sample_html_content():
    """Sample HTML content."""
    return """
    <html>
        <body>
            <article>
                <h1>Test Article</h1>
                <p>This is test content for the article.</p>
                <p>It contains multiple paragraphs.</p>
            </article>
        </body>
    </html>
    """


@pytest.fixture
def sample_metadata():
    """Sample extracted metadata."""
    return {
        "companies": ["Test Company"],
        "funding_amounts": [{"amount_millions": 10.0, "currency": "USD"}],
        "dates": ["2024-01-01"],
        "investors": ["Test Investor"],
        "sectors": ["AI"],
        "feed_name": "Test Feed",
        "article_url": "https://example.com/article",
        "title": "Test Article",
    }

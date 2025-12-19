"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_endpoint(client):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_create_feed_missing_user_id(client):
    """Test feed creation without user ID."""
    response = client.post("/feeds", json={"name": "Test", "url": "https://example.com/feed"})
    assert response.status_code == 401


def test_list_feeds(client):
    """Test listing feeds."""
    response = client.get("/feeds")
    assert response.status_code == 200
    data = response.json()
    assert "feeds" in data


def test_query_missing_user_id(client):
    """Test query without user ID."""
    response = client.post("/query", json={"query": "test query"})
    assert response.status_code == 401


def test_list_documents_missing_user_id(client):
    """Test listing documents without user ID."""
    response = client.get("/documents")
    assert response.status_code == 401


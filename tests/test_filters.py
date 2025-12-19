"""Tests for temporal and entity filtering."""
import pytest
from datetime import datetime, timedelta
from app.retrieval.filters import TemporalFilter, EntityFilter


def test_temporal_filter_parse_preset():
    """Test parsing preset date filters."""
    filter_obj = TemporalFilter()

    # Test preset
    date_range = filter_obj.parse_filter("last_week")
    assert date_range is not None
    assert "start_date" in date_range
    assert "end_date" in date_range
    assert date_range["start_date"] < date_range["end_date"]


def test_temporal_filter_parse_dict():
    """Test parsing dict date filters."""
    filter_obj = TemporalFilter()

    date_range = filter_obj.parse_filter(
        {"start_date": "2024-01-01", "end_date": "2024-01-31"}
    )
    assert date_range is not None
    assert date_range["start_date"] < date_range["end_date"]


def test_temporal_filter_filter_results():
    """Test filtering results by date."""
    filter_obj = TemporalFilter()

    results = [
        {
            "content": "Test 1",
            "metadata": {"published_date": "2024-01-15T00:00:00"},
        },
        {
            "content": "Test 2",
            "metadata": {"published_date": "2023-12-01T00:00:00"},
        },
        {
            "content": "Test 3",
            "metadata": {},  # No date
        },
    ]

    date_range = {
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 1, 31),
    }

    filtered = filter_obj.filter_results(results, date_range)
    # Should include Test 1 (in range) and Test 3 (no date)
    assert len(filtered) >= 1
    assert any(r["content"] == "Test 1" for r in filtered)


def test_entity_filter_company():
    """Test filtering by company name."""
    filter_obj = EntityFilter()

    results = [
        {
            "content": "Test 1",
            "metadata": {"companies": ["Anthropic", "OpenAI"]},
        },
        {
            "content": "Test 2",
            "metadata": {"companies": ["Google"]},
        },
    ]

    filtered = filter_obj.filter_results(results, company_filter=["Anthropic"])
    assert len(filtered) == 1
    assert filtered[0]["content"] == "Test 1"


def test_entity_filter_investor():
    """Test filtering by investor name."""
    filter_obj = EntityFilter()

    results = [
        {
            "content": "Test 1",
            "metadata": {
                "investors": [
                    {"name": "Andreessen Horowitz", "role": "lead"},
                    {"name": "Sequoia", "role": "participant"},
                ]
            },
        },
        {
            "content": "Test 2",
            "metadata": {"investors": [{"name": "Y Combinator", "role": "lead"}]},
        },
    ]

    filtered = filter_obj.filter_results(results, investor_filter=["Andreessen"])
    assert len(filtered) == 1
    assert filtered[0]["content"] == "Test 1"


def test_entity_filter_sector():
    """Test filtering by sector."""
    filter_obj = EntityFilter()

    results = [
        {
            "content": "Test 1",
            "metadata": {
                "sectors": [
                    {"sector": "AI", "confidence": 0.9},
                    {"sector": "FinTech", "confidence": 0.7},
                ]
            },
        },
        {
            "content": "Test 2",
            "metadata": {"sectors": [{"sector": "HealthTech", "confidence": 0.8}]},
        },
    ]

    filtered = filter_obj.filter_results(results, sector_filter=["AI"])
    assert len(filtered) == 1
    assert filtered[0]["content"] == "Test 1"


def test_entity_filter_multiple():
    """Test filtering with multiple entity filters."""
    filter_obj = EntityFilter()

    results = [
        {
            "content": "Test 1",
            "metadata": {
                "companies": ["Anthropic"],
                "sectors": [{"sector": "AI", "confidence": 0.9}],
            },
        },
        {
            "content": "Test 2",
            "metadata": {
                "companies": ["Anthropic"],
                "sectors": [{"sector": "FinTech", "confidence": 0.8}],
            },
        },
    ]

    # Should only return results matching both filters
    filtered = filter_obj.filter_results(
        results, company_filter=["Anthropic"], sector_filter=["AI"]
    )
    assert len(filtered) == 1
    assert filtered[0]["content"] == "Test 1"


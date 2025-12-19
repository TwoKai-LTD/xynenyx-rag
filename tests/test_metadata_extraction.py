"""Tests for enhanced metadata extraction."""
import pytest
from app.ingestion.metadata_extractor import MetadataExtractor


def test_enhanced_company_extraction():
    """Test enhanced company name extraction."""
    extractor = MetadataExtractor()

    content = """
    Anthropic announced a new AI model. OpenAI revealed GPT-5.
    TechCorp Labs secured funding. AI Systems Inc launched a product.
    """
    companies = extractor._extract_companies(content)

    assert len(companies) > 0
    assert any("Anthropic" in c or "OpenAI" in c for c in companies)


def test_enhanced_funding_extraction():
    """Test enhanced funding amount extraction with rounds."""
    extractor = MetadataExtractor()

    content = """
    The company raised $50 million in Series A funding.
    Another startup secured €30 million. A third raised $2 billion.
    """
    funding = extractor._extract_funding_amounts(content)

    assert len(funding) > 0
    assert any(f["amount_millions"] == 50.0 for f in funding)
    assert any(f.get("round") is not None for f in funding)


def test_enhanced_date_extraction():
    """Test enhanced date extraction with dateparser."""
    extractor = MetadataExtractor()

    content = """
    Published on January 15, 2024. Last week, the company announced.
    The funding round closed yesterday.
    """
    dates = extractor._extract_dates(content)

    assert len(dates) > 0
    # Should include parsed dates
    assert all("T" in d or "-" in d for d in dates)  # ISO format


def test_enhanced_investor_extraction():
    """Test enhanced investor extraction with roles."""
    extractor = MetadataExtractor()

    content = """
    The round was led by Andreessen Horowitz, with participation from
    Sequoia Capital and Y Combinator. Investors include Accel Partners.
    """
    investors = extractor._extract_investors(content)

    assert len(investors) > 0
    # Should have structured format with roles
    assert isinstance(investors[0], dict)
    assert "name" in investors[0]
    assert "role" in investors[0]


def test_enhanced_sector_extraction():
    """Test enhanced sector extraction with confidence scores."""
    extractor = MetadataExtractor()

    content = """
    The AI startup operates in the FinTech space. They also work
    with machine learning and cybersecurity technologies.
    """
    sectors = extractor._extract_sectors(content)

    assert len(sectors) > 0
    # Should have structured format with confidence
    assert isinstance(sectors[0], dict)
    assert "sector" in sectors[0]
    assert "confidence" in sectors[0]
    assert sectors[0]["confidence"] > 0


def test_metadata_extraction_integration():
    """Test full metadata extraction integration."""
    extractor = MetadataExtractor()

    content = """
    Anthropic, an AI startup, announced a $50 million Series A funding round
    led by Andreessen Horowitz. The round closed last week. The company
    operates in the AI and machine learning sectors.
    """
    article_metadata = {
        "title": "Anthropic Funding",
        "article_url": "https://example.com/article",
        "feed_name": "Test Feed",
    }

    metadata = extractor.extract(content, article_metadata)

    assert "companies" in metadata
    assert "funding_amounts" in metadata
    assert "investors" in metadata
    assert "sectors" in metadata
    assert metadata["title"] == "Anthropic Funding"


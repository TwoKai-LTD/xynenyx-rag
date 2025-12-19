"""Metadata extraction from content (companies, funding, dates, investors)."""
import re
from typing import Dict, Any, List
from datetime import datetime


class MetadataExtractor:
    """Extract structured metadata from startup/VC content."""

    def __init__(self):
        """Initialize metadata extractor."""
        # Funding amount patterns
        self.funding_patterns = [
            r"\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)",
            r"€(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)",
            r"£(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)",
            r"raised\s+\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)?",
            r"funding\s+of\s+\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)?",
        ]

        # Investor patterns
        self.investor_patterns = [
            r"led\s+by\s+([A-Z][a-zA-Z\s&,]+)",
            r"investors\s+include\s+([A-Z][a-zA-Z\s&,]+)",
            r"backed\s+by\s+([A-Z][a-zA-Z\s&,]+)",
            r"invested\s+by\s+([A-Z][a-zA-Z\s&,]+)",
        ]

        # Date patterns
        self.date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # ISO format
            r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}",  # "January 15, 2024"
            r"\d{1,2}\s+[A-Z][a-z]+\s+\d{4}",  # "15 January 2024"
        ]

    def extract(self, content: str, article_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured metadata from content.

        Args:
            content: Article content text
            article_metadata: Original article metadata (title, URL, etc.)

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "companies": self._extract_companies(content),
            "funding_amounts": self._extract_funding_amounts(content),
            "dates": self._extract_dates(content),
            "investors": self._extract_investors(content),
            "sectors": self._extract_sectors(content),
        }

        # Merge with original article metadata
        metadata.update(article_metadata)

        return metadata

    def _extract_companies(self, content: str) -> List[str]:
        """Extract company names (basic pattern matching)."""
        # Look for capitalized phrases that might be company names
        # This is basic - could be enhanced with NLP
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:announced|raised|launched|secured)"
        matches = re.findall(pattern, content)
        return list(set(matches[:10]))  # Limit to 10 unique companies

    def _extract_funding_amounts(self, content: str) -> List[Dict[str, Any]]:
        """Extract funding amounts."""
        amounts = []
        for pattern in self.funding_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                # Determine unit (million or billion)
                unit_text = match.group(0).lower()
                if "billion" in unit_text or "B" in unit_text:
                    value = value * 1000  # Convert to millions
                amounts.append({"amount_millions": value, "currency": "USD"})
        return amounts[:5]  # Limit to 5 amounts

    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates from content."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, content)
            dates.extend(matches)
        return list(set(dates[:10]))  # Limit to 10 unique dates

    def _extract_investors(self, content: str) -> List[str]:
        """Extract investor names."""
        investors = []
        for pattern in self.investor_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                investor_text = match.group(1).strip()
                # Split by common separators
                investor_list = re.split(r"[,\s]+and\s+|\s*,\s*", investor_text)
                investors.extend([inv.strip() for inv in investor_list if inv.strip()])
        return list(set(investors[:20]))  # Limit to 20 unique investors

    def _extract_sectors(self, content: str) -> List[str]:
        """Extract sectors/industries (keyword matching)."""
        sectors = [
            "AI",
            "Machine Learning",
            "FinTech",
            "HealthTech",
            "SaaS",
            "E-commerce",
            "Cybersecurity",
            "EdTech",
            "Climate Tech",
            "Biotech",
            "Enterprise Software",
            "Consumer",
        ]

        found_sectors = []
        content_lower = content.lower()
        for sector in sectors:
            if sector.lower() in content_lower:
                found_sectors.append(sector)

        return found_sectors


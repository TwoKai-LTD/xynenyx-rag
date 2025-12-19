"""Metadata extraction from content (companies, funding, dates, investors)."""
import re
from typing import Dict, Any, List
from datetime import datetime, timedelta
import dateparser


class MetadataExtractor:
    """Extract structured metadata from startup/VC content."""

    def __init__(self):
        """Initialize metadata extractor."""
        # Funding amount patterns (enhanced)
        self.funding_patterns = [
            r"\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B|k|K)",
            r"€(\d+(?:\.\d+)?)\s*(?:million|M|billion|B|k|K)",
            r"£(\d+(?:\.\d+)?)\s*(?:million|M|billion|B|k|K)",
            r"¥(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)",
            r"raised\s+\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B|k|K)?",
            r"funding\s+of\s+\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B|k|K)?",
            r"secured\s+\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B|k|K)?",
            r"closed\s+(?:a\s+)?\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B|k|K)?",
        ]

        # Funding round patterns
        self.round_patterns = [
            r"(?:Seed|seed)\s+round",
            r"Series\s+([A-Z])\s+round",
            r"Series\s+([A-Z])\s+funding",
            r"([A-Z])\s+round",
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
        # Extract metadata with enhanced methods
        companies = self._extract_companies(content)
        funding_amounts = self._extract_funding_amounts(content)
        dates = self._extract_dates(content)
        investors = self._extract_investors(content)
        sectors = self._extract_sectors(content)

        metadata = {
            "companies": companies,
            "funding_amounts": funding_amounts,
            "dates": dates,
            "investors": investors,
            "sectors": sectors,
        }

        # Merge with original article metadata
        metadata.update(article_metadata)

        return metadata

    def _extract_companies(self, content: str) -> List[str]:
        """Extract company names (enhanced pattern matching)."""
        companies = set()

        # Pattern 1: Company name followed by action verbs
        pattern1 = r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:announced|raised|launched|secured|closed|revealed)"
        matches1 = re.findall(pattern1, content)
        companies.update(matches1)

        # Pattern 2: "Company X" format
        pattern2 = r"(?:company|startup|firm)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"
        matches2 = re.findall(pattern2, content, re.IGNORECASE)
        companies.update(matches2)

        # Pattern 3: Common company name patterns (e.g., "TechCorp", "AI Labs")
        pattern3 = r"\b([A-Z][a-z]+(?:Corp|Labs|Tech|AI|Systems|Solutions|Inc|LLC))\b"
        matches3 = re.findall(pattern3, content)
        companies.update(matches3)

        # Filter out common false positives
        false_positives = {
            "The", "This", "That", "These", "Those", "Today", "Yesterday",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        }
        companies = {c for c in companies if c not in false_positives and len(c) > 2}

        return list(companies)[:15]  # Limit to 15 unique companies

    def _extract_funding_amounts(self, content: str) -> List[Dict[str, Any]]:
        """Extract funding amounts with round information."""
        amounts = []
        seen = set()

        for pattern in self.funding_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                unit_text = match.group(0).lower()

                # Determine unit and convert to millions
                if "billion" in unit_text or "B" in unit_text:
                    value = value * 1000
                elif "k" in unit_text or "K" in unit_text:
                    value = value / 1000

                # Determine currency
                currency = "USD"
                if "€" in match.group(0):
                    currency = "EUR"
                elif "£" in match.group(0):
                    currency = "GBP"
                elif "¥" in match.group(0):
                    currency = "JPY"

                # Extract funding round if nearby
                round_info = self._extract_round_nearby(content, match.start(), match.end())

                amount_key = (value, currency, round_info)
                if amount_key not in seen:
                    seen.add(amount_key)
                    amounts.append({
                        "amount_millions": value,
                        "currency": currency,
                        "round": round_info,
                    })

        return amounts[:5]  # Limit to 5 amounts

    def _extract_round_nearby(self, content: str, start: int, end: int) -> str | None:
        """Extract funding round information near a funding amount."""
        # Look in a 100-character window around the amount
        window_start = max(0, start - 50)
        window_end = min(len(content), end + 50)
        window = content[window_start:window_end]

        for pattern in self.round_patterns:
            match = re.search(pattern, window, re.IGNORECASE)
            if match:
                if match.group(0).startswith("Series"):
                    return f"Series {match.group(1)}"
                elif "Seed" in match.group(0):
                    return "Seed"
        return None

    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates from content using dateparser."""
        dates = []
        seen = set()

        # Extract using regex patterns first
        for pattern in self.date_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                parsed = dateparser.parse(match)
                if parsed:
                    iso_date = parsed.isoformat()
                    if iso_date not in seen:
                        seen.add(iso_date)
                        dates.append(iso_date)

        # Also try parsing relative dates and common phrases
        relative_phrases = [
            "today", "yesterday", "last week", "last month",
            "this week", "this month", "this year",
        ]
        for phrase in relative_phrases:
            if phrase.lower() in content.lower():
                parsed = dateparser.parse(phrase)
                if parsed:
                    iso_date = parsed.isoformat()
                    if iso_date not in seen:
                        seen.add(iso_date)
                        dates.append(iso_date)

        return dates[:10]  # Limit to 10 unique dates

    def _extract_investors(self, content: str) -> List[Dict[str, Any]]:
        """Extract investor names with role identification."""
        investors = []
        seen = set()

        for pattern in self.investor_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                investor_text = match.group(1).strip()
                # Split by common separators
                investor_list = re.split(r"[,\s]+and\s+|\s*,\s*", investor_text)

                # Determine role (lead vs participant)
                is_lead = "led by" in match.group(0).lower()

                for inv in investor_list:
                    inv_clean = inv.strip()
                    if inv_clean and inv_clean not in seen:
                        seen.add(inv_clean)
                        investors.append({
                            "name": inv_clean,
                            "role": "lead" if is_lead and len(investors) == 0 else "participant",
                        })

        return investors[:20]  # Limit to 20 unique investors

    def _extract_sectors(self, content: str) -> List[Dict[str, Any]]:
        """Extract sectors/industries with confidence scores."""
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
            "Blockchain",
            "Web3",
            "Gaming",
            "Media",
            "Transportation",
            "Real Estate",
            "Food & Beverage",
            "Fashion",
        ]

        found_sectors = []
        content_lower = content.lower()

        for sector in sectors:
            sector_lower = sector.lower()
            # Count occurrences for confidence
            count = content_lower.count(sector_lower)
            if count > 0:
                # Simple confidence based on frequency
                confidence = min(1.0, 0.5 + (count * 0.1))
                found_sectors.append({
                    "sector": sector,
                    "confidence": confidence,
                })

        # Sort by confidence
        found_sectors.sort(key=lambda x: x["confidence"], reverse=True)
        return found_sectors[:10]  # Limit to top 10 sectors


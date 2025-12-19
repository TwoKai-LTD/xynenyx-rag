"""Temporal and entity filtering for search results."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import dateparser
from app.config import settings


class TemporalFilter:
    """Filter results by date ranges."""

    def __init__(self):
        """Initialize temporal filter."""
        self.presets = settings.temporal_filter_presets

    def parse_filter(
        self,
        date_filter: str | Dict[str, str] | None,
    ) -> Optional[Dict[str, datetime]]:
        """
        Parse date filter into start and end dates.

        Args:
            date_filter: Filter string (preset) or dict with start_date/end_date

        Returns:
            Dictionary with start_date and end_date, or None if invalid
        """
        if not date_filter:
            return None

        # Handle preset strings
        if isinstance(date_filter, str):
            if date_filter in self.presets:
                days = int(self.presets[date_filter].split()[0])
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                return {"start_date": start_date, "end_date": end_date}
            else:
                # Try parsing as relative date
                parsed = dateparser.parse(date_filter)
                if parsed:
                    end_date = datetime.now()
                    start_date = parsed
                    return {"start_date": start_date, "end_date": end_date}

        # Handle dict with start_date/end_date
        if isinstance(date_filter, dict):
            start_date = None
            end_date = None

            if "start_date" in date_filter:
                start_date = dateparser.parse(date_filter["start_date"])
            if "end_date" in date_filter:
                end_date = dateparser.parse(date_filter["end_date"])

            if start_date or end_date:
                return {
                    "start_date": start_date or datetime.min,
                    "end_date": end_date or datetime.now(),
                }

        return None

    def filter_results(
        self,
        results: List[Dict[str, Any]],
        date_range: Dict[str, datetime],
    ) -> List[Dict[str, Any]]:
        """
        Filter results by date range.

        Args:
            results: List of search results
            date_range: Dictionary with start_date and end_date

        Returns:
            Filtered results
        """
        if not date_range:
            return results

        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")

        filtered = []
        for result in results:
            # Try to get date from metadata
            metadata = result.get("metadata", {})
            date_str = metadata.get("published_date") or metadata.get("date")

            if date_str:
                try:
                    # Parse date string
                    if isinstance(date_str, str):
                        result_date = dateparser.parse(date_str)
                    else:
                        result_date = date_str

                    if result_date:
                        # Check if date is in range
                        if start_date and result_date < start_date:
                            continue
                        if end_date and result_date > end_date:
                            continue

                        filtered.append(result)
                except Exception:
                    # If date parsing fails, include the result
                    filtered.append(result)
            else:
                # If no date in metadata, include the result
                filtered.append(result)

        return filtered


class EntityFilter:
    """Filter results by entity (company, investor, sector)."""

    def filter_results(
        self,
        results: List[Dict[str, Any]],
        company_filter: Optional[List[str]] = None,
        investor_filter: Optional[List[str]] = None,
        sector_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter results by entity criteria.

        Args:
            results: List of search results
            company_filter: List of company names to filter by
            investor_filter: List of investor names to filter by
            sector_filter: List of sectors to filter by

        Returns:
            Filtered results
        """
        if not any([company_filter, investor_filter, sector_filter]):
            return results

        filtered = []
        for result in results:
            metadata = result.get("metadata", {})

            # Check company filter
            if company_filter:
                companies = metadata.get("companies", [])
                if isinstance(companies, list):
                    # Handle both string and dict formats
                    company_names = [
                        c if isinstance(c, str) else c.get("name", "")
                        for c in companies
                    ]
                    company_names_lower = [c.lower() for c in company_names]
                    if not any(
                        filter_name.lower() in company_name
                        for filter_name in company_filter
                        for company_name in company_names_lower
                    ):
                        continue

            # Check investor filter
            if investor_filter:
                investors = metadata.get("investors", [])
                if isinstance(investors, list):
                    # Handle both string and dict formats
                    investor_names = [
                        inv if isinstance(inv, str) else inv.get("name", "")
                        for inv in investors
                    ]
                    investor_names_lower = [inv.lower() for inv in investor_names]
                    if not any(
                        filter_name.lower() in investor_name
                        for filter_name in investor_filter
                        for investor_name in investor_names_lower
                    ):
                        continue

            # Check sector filter
            if sector_filter:
                sectors = metadata.get("sectors", [])
                if isinstance(sectors, list):
                    # Handle both string and dict formats
                    sector_names = [
                        s if isinstance(s, str) else s.get("sector", "")
                        for s in sectors
                    ]
                    sector_names_lower = [s.lower() for s in sector_names]
                    if not any(
                        filter_name.lower() in sector_name
                        for filter_name in sector_filter
                        for sector_name in sector_names_lower
                    ):
                        continue

            filtered.append(result)

        return filtered


"""RSS feed parsing using feedparser."""
import feedparser
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
from app.config import settings


class RSSParser:
    """Parser for RSS and Atom feeds."""

    def __init__(self):
        """Initialize RSS parser."""
        self.timeout = settings.rss_request_timeout
        self.max_retries = settings.rss_max_retries

    def parse_feed(self, feed_url: str) -> Dict[str, Any]:
        """
        Parse an RSS or Atom feed.

        Args:
            feed_url: URL of the RSS/Atom feed

        Returns:
            Dictionary with feed metadata and entries

        Raises:
            ValueError: If feed cannot be parsed
        """
        try:
            # feedparser can handle URLs directly
            feed = feedparser.parse(feed_url)

            if feed.bozo and feed.bozo_exception:
                raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")

            return {
                "title": feed.feed.get("title", ""),
                "link": feed.feed.get("link", ""),
                "description": feed.feed.get("description", ""),
                "entries": self._parse_entries(feed.entries),
            }
        except Exception as e:
            raise ValueError(f"Error parsing RSS feed {feed_url}: {str(e)}") from e

    def _parse_entries(self, entries: List) -> List[Dict[str, Any]]:
        """
        Parse feed entries.

        Args:
            entries: List of feed entries

        Returns:
            List of parsed entry dictionaries
        """
        parsed_entries = []
        seen_urls = set()

        for entry in entries:
            # Extract entry data
            entry_id = entry.get("id", "")
            link = entry.get("link", "")
            title = entry.get("title", "")
            description = entry.get("description", "")

            # Parse published date
            published_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_date = datetime(*entry.published_parsed[:6]).isoformat()
                except Exception:
                    pass

            # Use link or ID for deduplication
            unique_id = link or entry_id
            if not unique_id or unique_id in seen_urls:
                continue

            seen_urls.add(unique_id)

            parsed_entries.append(
                {
                    "id": entry_id,
                    "link": link,
                    "title": title,
                    "description": description,
                    "published_date": published_date,
                    "published_parsed": entry.get("published_parsed"),
                }
            )

        return parsed_entries

    def validate_feed_url(self, feed_url: str) -> bool:
        """
        Validate that a feed URL is accessible and parseable.

        Args:
            feed_url: URL to validate

        Returns:
            True if feed is valid, False otherwise
        """
        try:
            feed = feedparser.parse(feed_url)
            return not feed.bozo and len(feed.entries) > 0
        except Exception:
            return False


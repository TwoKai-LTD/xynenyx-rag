"""HTML content extraction using BeautifulSoup."""
import httpx
from bs4 import BeautifulSoup
from typing import Optional
from app.config import settings


class HTMLParser:
    """Parser for extracting main content from HTML pages."""

    def __init__(self):
        """Initialize HTML parser."""
        self.timeout = settings.html_request_timeout
        self.max_retries = settings.html_max_retries
        self.user_agent = settings.html_user_agent

    async def extract_content(self, url: str) -> Optional[str]:
        """
        Extract main content from an HTML page.

        Args:
            url: URL of the page to extract

        Returns:
            Extracted text content or None if extraction fails
        """
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(
                        url,
                        headers={"User-Agent": self.user_agent},
                    )
                    response.raise_for_status()

                    # Parse HTML
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        script.decompose()

                    # Try to find main content
                    # Common patterns: article, main, .content, .post, .entry
                    content_selectors = [
                        "article",
                        "main",
                        ".content",
                        ".post",
                        ".entry",
                        ".article-content",
                        "[role='main']",
                    ]

                    content = None
                    for selector in content_selectors:
                        elements = soup.select(selector)
                        if elements:
                            content = elements[0]
                            break

                    # Fallback to body if no main content found
                    if not content:
                        content = soup.find("body")

                    if not content:
                        return None

                    # Extract text
                    text = content.get_text(separator="\n", strip=True)

                    # Clean up whitespace
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    return "\n".join(lines)

                except httpx.TimeoutException:
                    if attempt < self.max_retries - 1:
                        continue
                    return None
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        continue
                    print(f"Error extracting content from {url}: {e}")
                    return None

        return None


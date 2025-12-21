from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class UrlExtractResult:
    url: str
    title: Optional[str]
    h1: Optional[str]
    text_snippet: str


def fetch_and_extract(url: str, *, timeout_sec: int = 15) -> UrlExtractResult:
    """
    Minimal extractor for MVP.
    - Fetch HTML
    - Extract <title>, first <h1>, and a short text snippet
    """
    headers = {
        "User-Agent": "SpecScopeXBot/0.1 (+https://example.local) Python requests",
        "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
    }
    r = requests.get(url, headers=headers, timeout=timeout_sec)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(" ", strip=True) if h1_tag else None

    text = soup.get_text(" ", strip=True)
    text = " ".join(text.split())
    snippet = text[:1200]

    return UrlExtractResult(
        url=url,
        title=title,
        h1=h1,
        text_snippet=snippet,
    )

from __future__ import annotations

from typing import Any
import requests

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_newsapi(
    query: str,
    api_key: str,
    limit: int = 10,
    domains: list[str] | None = None,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """
    NewsAPI. Returns normalized articles.
    """
    params = {
        "q": query,
        "language": "en",
        "pageSize": limit,
        "sortBy": "publishedAt",
        "searchIn": "title,description",
        "apiKey": api_key,
    }
    if domains:
        params["domains"] = ",".join(domains)

    response = requests.get(NEWSAPI_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    items: list[dict[str, Any]] = []
    for article in (payload.get("articles") or [])[:limit]:
        items.append(
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "source": (article.get("source") or {}).get("name"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "tickers": [],
                "provider": "newsapi",
            }
        )
    return items

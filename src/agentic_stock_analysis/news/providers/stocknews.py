from __future__ import annotations

from typing import Any
import requests

BASE_URL = "https://stocknewsapi.com/api/v1"


def fetch_stocknews(
    ticker: str,
    api_key: str,
    date: str = "today",
    items: int = 20,
    page: int = 1,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """
    StocknewsAPI (single ticker). Returns normalized articles.
    """
    t = ticker.replace(".", "-").upper()

    params = {
        "tickers": t,
        "date": date,
        "items": items,
        "page": page,
        "token": api_key,
    }

    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    data = payload.get("data") or []
    items: list[dict[str, Any]] = []

    for article in data:
        items.append(
            {
                "title": article.get("title"),
                "description": article.get("content"),
                "source": article.get("source"),
                "url": article.get("url"),
                "published_at": article.get("date"),
                "tickers": article.get("tickers") or [],
                "provider": "stocknews",
            }
        )

    return items

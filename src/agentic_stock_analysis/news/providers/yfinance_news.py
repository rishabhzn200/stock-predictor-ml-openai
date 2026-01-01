from __future__ import annotations

from typing import Any
import yfinance as yf


def fetch_yfinance_news(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    yfinance fallback.
    """
    t = ticker.replace(".", "-").upper()
    yf_ticker = yf.Ticker(t)
    data = (getattr(yf_ticker, "news", None) or [])[:limit]

    items: list[dict[str, Any]] = []
    for article in data:
        items.append(
            {
                "title": article.get("title"),
                "description": None,
                "source": article.get("publisher"),
                "url": article.get("link"),
                "published_at": article.get("providerPublishTime"),
                "tickers": [],
                "provider": "yfinance",
            }
        )
    return items

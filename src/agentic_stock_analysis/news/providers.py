import logging
from typing import Any, Dict, List, Optional

import requests
import yfinance as yf

from agentic_stock_analysis.core.config import get_news_config

logger = logging.getLogger(__name__)


def fetch_news_from_newsapi(
    query: str, limit: int = 5, domains: Optional[list[str]] = None
) -> List[Dict[str, Any]]:
    config = get_news_config()
    if not config.newsapi_key:
        raise ValueError("NEWSAPI_API_KEY is not set")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": limit,
        "sortBy": "publishedAt",
        "searchIn": "title,description",
        "apiKey": config.newsapi_key,
    }
    if domains:
        params["domains"] = ",".join(domains)

    response = requests.get(url, params=params, timeout=10)
    logger.info(f"[news] newsapi response_status={response.status_code}")
    response.raise_for_status()
    data = response.json()

    items: List[Dict[str, Any]] = []
    for article in data.get("articles", [])[:limit]:
        items.append(
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "source": (article.get("source") or {}).get("name"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
            }
        )
    return items


def fetch_news_from_yfinance(ticker_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    yf_ticker = yf.Ticker(ticker_name)
    yf_news = getattr(yf_ticker, "news", []) or []
    yf_news = yf_news[:limit]

    items: List[Dict[str, Any]] = []
    for news in yf_news:
        items.append(
            {
                "title": news.get("title"),
                "description": None,
                "source": news.get("publisher"),
                "url": news.get("link"),
                "published_at": news.get("providerPublishTime"),
            }
        )
    return items

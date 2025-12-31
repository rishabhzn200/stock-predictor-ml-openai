import logging
from typing import Any, Dict, List, Tuple

from agentic_stock_analysis.core.config import get_news_config
from agentic_stock_analysis.news.constants import ALLOWED_NEWS_DOMAINS
from agentic_stock_analysis.news.providers import (
    fetch_news_from_newsapi,
    fetch_news_from_yfinance,
)

logger = logging.getLogger(__name__)


def get_news_items(
    ticker: str, terms: List[str], limit: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (provider_used, items).
    Provider is chosen from env config (NEWS_PROVIDER). Falls back gracefully.
    """
    config = get_news_config()
    provider = (config.provider or "yfinance").lower()

    # Build a NewsAPI query from terms
    query = " OR ".join(f'"{t}"' for t in terms if t)

    items: List[Dict[str, Any]] = []
    try:
        if provider == "newsapi":
            items = fetch_news_from_newsapi(
                query, limit=limit, domains=ALLOWED_NEWS_DOMAINS
            )

            # if domain is too restrictive
            if not items:
                logger.info(
                    "[news] No trusted-domain results; retrying without domain restriction"
                )
                items = fetch_news_from_newsapi(query, limit=limit)

        else:
            items = fetch_news_from_yfinance(ticker, limit=limit)

    except Exception as e:
        logger.exception(
            f"[news] provider={provider} failed; falling back to yfinance: {e}"
        )
        provider = "yfinance"
        items = fetch_news_from_yfinance(ticker, limit=limit)

    return provider, items

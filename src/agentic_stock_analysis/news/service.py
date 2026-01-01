import logging
from typing import Any, Dict, List, Tuple


from agentic_stock_analysis.core.config import get_news_config
from agentic_stock_analysis.news.constants import ALLOWED_NEWS_DOMAINS
from agentic_stock_analysis.news.dedupe import merge_dedupe_and_cap
from agentic_stock_analysis.news.news_sorter import sort_by_latest_timestamp_first
from agentic_stock_analysis.news.providers.stocknews import fetch_stocknews
from agentic_stock_analysis.news.providers.newsapi import fetch_newsapi
from agentic_stock_analysis.news.providers.yfinance_news import fetch_yfinance_news

logger = logging.getLogger(__name__)


def get_news_items(
    ticker: str, terms: List[str], limit: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (provider_used, items) using provider priority list.

    Config examples:
      NEWS_PROVIDERS="stocknews,newsapi"
      NEWS_PROVIDERS="stocknews"
      NEWS_PROVIDERS="newsapi"

    Always falls back to yfinance if needed.

    provider_used values:
      "stocknews"
      "stocknews+newsapi"
      "newsapi"
      "yfinance"
      "none"
    """
    config = get_news_config()
    ticker = (ticker or "").strip().upper()

    terms = [t for t in (terms or []) if t]
    if not terms:
        terms = [ticker]

    # Build NewsAPI query from terms
    query = " OR ".join(f'"{t}"' for t in terms if t)

    # Provider list (preferred)
    if not config.providers:
        raise ValueError("NEWS_PROVIDERS is empty or not set")

    providers = [p.strip().lower() for p in config.providers if p and p.strip()]

    # thresholds
    max_items_total = int(config.max_items_total)
    augment_threshold = int(config.augment_threshold)

    if augment_threshold > max_items_total:
        raise ValueError(
            f"augment_threshold ({augment_threshold}) "
            f"cannot be greater than max_items_total ({max_items_total})"
        )

    collected_data: List[Dict[str, Any]] = []
    used_providers: List[str] = []

    for provider in providers:
        try:
            response_data: List[Dict[str, Any]] = []
            if provider == "stocknews":
                # Stocknews is ticker-based, so terms is not required
                response_data = fetch_stocknews(
                    ticker=ticker,
                    api_key=config.stocknews_api_key,
                    items=config.stocknews_items,
                    page=1,
                    date="today",
                )
            elif provider == "newsapi":
                response_data = fetch_newsapi(
                    query=query,
                    api_key=config.newsapi_api_key,
                    limit=config.newsapi_items,
                    domains=ALLOWED_NEWS_DOMAINS,
                )
                if not response_data:
                    logger.info(
                        "[news] No trusted-domain results; retrying without domain restriction"
                    )
                    response_data = fetch_newsapi(
                        query=query,
                        api_key=config.newsapi_api_key,
                        limit=config.newsapi_items,
                        domains=None,
                    )
                response_data
            elif provider == "yfinance":
                response_data = fetch_yfinance_news(ticker=ticker, limit=limit)
            else:
                logger.info(f"[news] Unknown provider '{provider}', skipping")
                continue

            if response_data:
                # Sort the articles by latest timestamp first
                response_data = sort_by_latest_timestamp_first(response_data)

                # Merge the articles in case of duplicates
                collected_data = merge_dedupe_and_cap(
                    base_items=collected_data,
                    extra_items=response_data,
                    max_items_total=max_items_total,
                )
                used_providers.append(provider)

            # Stop early once we have enough data
            if len(collected_data) >= augment_threshold:
                break
            pass
        except Exception as e:
            logger.exception(f"[news] provider={provider} failed ticker={ticker}: {e}")

    # Final fallback if nothing found
    if not collected_data:
        try:
            collected_data = fetch_yfinance_news(ticker=ticker, limit=limit) or []
            if collected_data:
                used_providers = ["yfinance"]
        except Exception as e:
            logger.exception(f"[news] yfinance fallback failed ticker={ticker}: {e}")

    provider_used = "+".join(used_providers) if used_providers else "none"
    return provider_used, collected_data[:limit]

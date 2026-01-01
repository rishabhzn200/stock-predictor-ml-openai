from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class NewsConfig:
    providers: list[str]  # e.g. ["stocknews", "newsapi"]
    stocknews_api_key: str | None
    newsapi_api_key: str | None

    augment_threshold: int  # if we have fewer than this, try next provider
    max_items_total: int  # cap after dedupe
    stocknews_items: int  # items per Stocknews call (page size)
    newsapi_items: int  # items per NewsAPI call


def get_news_config() -> NewsConfig:
    providers_raw = os.getenv("NEWS_PROVIDERS", "stocknews,newsapi")
    providers = [p.strip().lower() for p in providers_raw.split(",") if p.strip()]
    return NewsConfig(
        providers=providers,
        stocknews_api_key=os.getenv("STOCKNEWS_API_KEY"),
        newsapi_api_key=os.getenv("NEWSAPI_API_KEY"),
        augment_threshold=int(os.getenv("NEWS_AUGMENT_THRESHOLD", "8")),
        max_items_total=int(os.getenv("NEWS_MAX_ITEMS_TOTAL", "15")),
        stocknews_items=int(os.getenv("STOCKNEWS_ITEMS", "20")),
        newsapi_items=int(os.getenv("NEWSAPI_ITEMS", "10")),
    )

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Any


def _norm_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\w\s]", "", t)
    return t


def _norm_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        # drop query + fragment to catch near-duplicates
        return f"{p.scheme}://{p.netloc}{p.path}"
    except Exception:
        return u


def merge_dedupe_and_cap(
    base_items: list[dict[str, Any]],
    extra_items: list[dict[str, Any]],
    max_items_total: int,
) -> list[dict[str, Any]]:
    """
    Merges base + extra items while removing duplicates by URL then title.
    Preserves order (base first).
    """
    seen_url: set[str] = set()
    seen_title: set[str] = set()
    merged_items: list[dict[str, Any]] = []

    def add(items: list[dict[str, Any]]):
        nonlocal merged_items
        for a in items:
            url_key = _norm_url(a.get("url") or "")
            title_key = _norm_title(a.get("title") or "")

            if url_key and url_key in seen_url:
                continue
            if title_key and title_key in seen_title:
                continue

            if url_key:
                seen_url.add(url_key)
            if title_key:
                seen_title.add(title_key)

            merged_items.append(a)
            if len(merged_items) >= max_items_total:
                return

    add(base_items)
    if len(merged_items) < max_items_total:
        add(extra_items)

    return merged_items

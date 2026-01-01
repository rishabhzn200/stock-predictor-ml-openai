from datetime import datetime
from typing import Any, Dict, List


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        # handles ISO + Z
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def sort_by_latest_timestamp_first(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort articles newest → oldest using published_at.
    Safe even if some items have missing/invalid timestamps.
    """
    return sorted(
        items,
        key=lambda x: _parse_dt(x.get("published_at")),
        reverse=True,
    )

import logging
from typing import Any, Dict

import yfinance as yf

from agentic_stock_analysis.agent.state import AgentState

logger = logging.getLogger(__name__)


def fetch_ticker_metadata_node(state: AgentState) -> AgentState:
    ticker_name = state.get("ticker")
    if not ticker_name:
        raise ValueError("Missing 'ticker' in agent state")

    ticker = yf.Ticker(ticker_name)
    info = ticker.info or {}

    metadata: Dict[str, Any] = {
        "symbol": ticker_name,
        "shortName": info.get("shortName"),
        "longName": info.get("longName"),
        "quoteType": info.get("quoteType"),
        "category": info.get("category"),
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
    }

    logger.info(f"[agent] metadata_node ticker={ticker_name} metadata={metadata}")
    return {"ticker_metadata": metadata}

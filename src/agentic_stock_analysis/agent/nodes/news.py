import logging

from agentic_stock_analysis.agent.state import AgentState
from agentic_stock_analysis.news.service import get_news_items

logger = logging.getLogger(__name__)


def news_node(state: AgentState) -> AgentState:
    ticker = state.get("ticker")
    terms = state.get("news_search_terms") or [ticker]

    logger.info(f"[agent] news_node ticker={ticker}")
    provider_used, items = get_news_items(ticker=ticker, terms=terms, limit=10)

    return {"news_provider": provider_used, "news_items": items}

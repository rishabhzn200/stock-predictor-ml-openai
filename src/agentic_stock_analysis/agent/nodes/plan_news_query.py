import json
import logging

from langchain_openai import ChatOpenAI

from agentic_stock_analysis.agent.state import AgentState, NewsQuery

logger = logging.getLogger(__name__)


def plan_news_query_node(state: AgentState) -> AgentState:
    ticker = state.get("ticker")
    if not ticker:
        raise ValueError("Missing 'ticker' in agent state")

    ticker_metadata = state.get("ticker_metadata", {}) or {}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(NewsQuery)

    prompt = f"""
    Generate precise search terms for a News API query, using provided ticker metadata.

    Ticker: {ticker}
    Metadata (JSON):
    {json.dumps(ticker_metadata, indent=2)}

    Return JSON with:
    {{"terms": ["..."]}}

    Rules:
    - Prefer official names from metadata (shortName/longName) over generic terms.
    - Include 3 to 6 terms max.
    - Avoid overly generic terms like just "ETF" or "fund" or "stock".
    - For ETFs/commodities, include 1-2 asset-specific phrases if clearly implied.
    - Only include the raw ticker symbol if it is likely unambiguous (>=4 chars) OR metadata indicates it is commonly referenced.
    """.strip()

    result: NewsQuery = structured_llm.invoke(prompt)
    logger.info(f"[agent] plan_news_query_node response={result}")

    terms = [term.strip() for term in result.terms if term and term.strip()]
    if not terms:
        fallback = [
            ticker_metadata.get("shortName"),
            ticker_metadata.get("longName"),
            ticker,
        ]
        terms = [x for x in fallback if x]

    logger.info(
        f"[agent] plan_news_query_node completed: ticker={ticker}, terms={terms}"
    )

    return {"news_search_terms": terms}

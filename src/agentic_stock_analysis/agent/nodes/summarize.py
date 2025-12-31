import json
import logging

from langchain_openai import ChatOpenAI

from agentic_stock_analysis.agent.state import AgentState

logger = logging.getLogger(__name__)


def summarize_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    question = state["question"]
    prediction = state.get("prediction")
    indicators = state.get("indicators", {})
    provider = state.get("news_provider", "unknown")
    news_items = state.get("news_items", [])
    news_sent_label = state.get("news_sentiment_label", "NO_NEWS")
    news_sent_score = state.get("news_sentiment_score", 0.0)
    alignment = state.get("alignment", "UNKNOWN")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    news_compact = [
        {
            "title": x.get("title"),
            "source": x.get("source"),
            "published_at": x.get("published_at"),
        }
        for x in news_items
        if x.get("title")
    ]

    prompt = f"""
    You are an AI assistant that summarizes short-term stock signals for educational purposes (not financial advice).

    Ticker: {ticker}
    User question: {question}

    Model prediction horizon: next trading day (directional UP/DOWN).
    Model prediction for tomorrow: {prediction}

    News sentiment horizon: next 1–3 trading days.
    News sentiment: {news_sent_label} (score={news_sent_score})
    Model vs news alignment: {alignment}

    Latest indicators (JSON):
    {json.dumps(indicators, indent=2)}

    Recent news headlines (provider={provider}) (JSON):
    {json.dumps(news_compact, indent=2)}

    Write a concise report with:
    1) One-line summary (include model prediction + whether news aligns/conflicts)
    2) Indicators interpretation in simple terms
    3) News context (ONLY headlines provided; no hallucinations)
    4) What to watch next day (2-3 bullets)
    5) Headlines used (max 5)

    Include a short disclaimer that this is not financial advice.
    """.strip()

    result = llm.invoke(prompt)
    report = result.content if hasattr(result, "content") else str(result)
    return {"report": report}

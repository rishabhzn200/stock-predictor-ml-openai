import json

from langchain_openai import ChatOpenAI

from agentic_stock_analysis.agent.state import AgentState, NewsSentiment


def news_sentiment_node(state: AgentState) -> AgentState:
    headlines = [
        n.get("title") for n in (state.get("news_items") or []) if n.get("title")
    ][:5]

    if not headlines:
        return {
            "news_sentiment_label": "NO_NEWS",
            "news_sentiment_score": 0.0,
            "news_headlines_used": [],
        }

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured = llm.with_structured_output(NewsSentiment)

    prompt = f"""
    You are scoring SHORT-TERM news tone for the next 1-3 trading days for the given asset.
    Use ONLY the headlines; do not invent details.

    Headlines:
    {json.dumps(headlines, indent=2)}

    Return:
    - label: POSITIVE, NEGATIVE, NEUTRAL, MIXED
    - score: float in [-1,1]
    """.strip()

    result: NewsSentiment = structured.invoke(prompt)

    label = (result.label or "").strip().upper()
    if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}:
        label = "NEUTRAL"

    return {
        "news_sentiment_label": label,
        "news_sentiment_score": float(result.score),
        "news_headlines_used": headlines,
    }

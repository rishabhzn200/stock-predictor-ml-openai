from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    ticker: str
    ticker_metadata: Dict[str, Any]
    question: str
    prediction: str
    indicators: Dict[str, Any]
    news_sentiment_label: str
    news_sentiment_score: float
    alignment: str
    news_headlines_used: List[str]
    news_search_terms: List[str]
    news_provider: str
    news_items: List[Dict[str, Any]]
    report: str
    error: str


class NewsQuery(BaseModel):
    terms: List[str] = Field(
        ..., description="Search terms for querying financial news APIs."
    )


class NewsSentiment(BaseModel):
    label: str = Field(..., description="POSITIVE, NEGATIVE, NEUTRAL, MIXED, NO_NEWS")
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score in [-1, 1]")

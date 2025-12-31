from pydantic import BaseModel
from typing import Optional


class AnalyzeRequest(BaseModel):
    ticker: str
    explain: bool = True


class AnalyzeResponse(BaseModel):
    ticker: str
    model_prediction: str
    indicators: dict
    explanation: Optional[str] | None


class AgentAnalyzeRequest(BaseModel):
    ticker: str
    question: str


class AgentAnalyzeResponse(BaseModel):
    ticker: str
    question: str
    model_prediction: str | None = None
    news_sentiment_label: str | None = None
    news_sentiment_score: float | None = None
    alignment: str | None = None
    news_headlines_used: list[str] | None = None
    report: str

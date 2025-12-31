import logging
from fastapi import APIRouter, HTTPException

from agentic_stock_analysis.agent.graph import build_agent_graph
from agentic_stock_analysis.ml.predictor import predict_stock
from agentic_stock_analysis.llm.explainer import explain_trend
from agentic_stock_analysis.api_models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-init so reload/startup is fast and reliable
_AGENT_GRAPH = None


def get_agent_graph():
    global _AGENT_GRAPH
    if _AGENT_GRAPH is None:
        _AGENT_GRAPH = build_agent_graph()
    return _AGENT_GRAPH


@router.get("/health_check")
def health_check():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    ticker = request.ticker.strip().upper()
    logger.info(f"/analyze called for ticker={ticker}, explain={request.explain}")

    # Run the prediction pipeline
    try:
        pred, indicators = predict_stock(ticker)
    except Exception as e:
        logger.exception(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    prediction_str = "UP" if pred == 1 else "DOWN"

    # AI explanation
    explanation = None
    if request.explain:
        try:
            explanation = explain_trend(ticker, pred, indicators)
        except Exception as e:
            logger.exception(f"Explanation failed for {ticker}: {e}")
            explanation = None

    return AnalyzeResponse(
        ticker=ticker,
        model_prediction=prediction_str,
        indicators=indicators,
        explanation=explanation,
    )


@router.post("/analyze_agent", response_model=AgentAnalyzeResponse)
def analyze_agent(request: AgentAnalyzeRequest):
    ticker = request.ticker.strip().upper()
    question = request.question.strip()
    logger.info(f"/analyze_agent called ticker={ticker}")

    try:
        agent_graph = get_agent_graph()
        final_state = agent_graph.invoke({"ticker": ticker, "question": question})
    except Exception as e:
        logger.exception(f"Agent graph failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return AgentAnalyzeResponse(
        ticker=ticker,
        question=question,
        model_prediction=final_state.get("prediction"),
        news_sentiment_label=final_state.get("news_sentiment_label"),
        news_sentiment_score=final_state.get("news_sentiment_score"),
        alignment=final_state.get("alignment"),
        news_headlines_used=final_state.get("news_headlines_used"),
        report=final_state.get("report", ""),
    )

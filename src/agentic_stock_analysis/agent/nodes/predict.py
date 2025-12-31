import logging

from agentic_stock_analysis.agent.state import AgentState
from agentic_stock_analysis.ml.predictor import predict_stock

logger = logging.getLogger(__name__)


def predict_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    logger.info(f"[agent] predict_node ticker={ticker}")

    pred_int, indicators = predict_stock(ticker)
    prediction = "UP" if pred_int == 1 else "DOWN"

    return {"prediction": prediction, "indicators": indicators}

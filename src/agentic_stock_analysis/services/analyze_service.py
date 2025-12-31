from agentic_stock_analysis.ml.predictor import predict_stock
from agentic_stock_analysis.llm.explainer import explain_trend


def analyze_ticker(ticker: str, explain: bool = True):
    pred, indicators = predict_stock(ticker)
    explanation = explain_trend(ticker, pred, indicators) if explain else None
    return pred, indicators, explanation

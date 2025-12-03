import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stock_predictor.log_config import setup_logging
from stock_predictor.predict_stock import predict_stock
from stock_predictor.ai_explainer import explain_trend

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Predictor API",
    description="Predicts next day stock movement and gets an AI explanation",
    version="1.0",
)


class AnalyzeRequest(BaseModel):
    ticker: str
    explain: bool = True


class AnalyzeResponse(BaseModel):
    ticker: str
    prediction: str
    indicators: dict
    explanation: str | None


@app.get("/health_check")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    ticker = req.ticker.strip().upper()
    logger.info(f"/analyze called for ticker={ticker}, explain={req.explain}")

    # Run the prediction pipeline
    try:
        pred, indicators = predict_stock(ticker)
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    prediction_str = "UP" if pred == 1 else "DOWN"

    # AI explanation
    explanation = None
    if req.explain:
        try:
            explanation = explain_trend(ticker, pred, indicators)
        except Exception as e:
            # logger.error(f"Explanation failed for {ticker}: {e}")
            logger.exception(f"Explanation failed for {ticker}: {e}")
            # Do not fail the whole request if explanation breaks
            explanation = None

    return AnalyzeResponse(
        ticker=ticker,
        prediction=prediction_str,
        indicators=indicators,
        explanation=explanation,
    )

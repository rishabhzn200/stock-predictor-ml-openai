import logging
from pathlib import Path

from ..services.fetch_data import get_stock_data
from .features import compute_features
from .model import get_model

logger = logging.getLogger(__name__)

FEATURES = ["RSI", "EMA_10", "EMA_50", "MACD"]


def predict_stock(ticker):
    """
    Fetch data, compute features, load (or train) model, and predict
    whether the ticker goes up (1) or down (0) tomorrow.

    Returns:
        pred (int): 1 for UP, 0 for DOWN
        latest_features (dict): the latest row's feature values
    """
    # Use longer period for more training data
    logger.info(f"Running prediction for {ticker}...")
    df = get_stock_data(ticker, period="2y")
    df = compute_features(df)

    # Try to load an existing model
    model = get_model()

    latest = df[FEATURES].iloc[[-1]]
    pred = model.predict(latest)[0]

    # Convert indicators to a JSON-friendly dict (string keys, float values)
    latest_raw = latest.to_dict(orient="records")[0]
    indicators = {str(k): float(v) for k, v in latest_raw.items()}

    logger.info(f"Prediction complete for {ticker}.")
    return pred, indicators

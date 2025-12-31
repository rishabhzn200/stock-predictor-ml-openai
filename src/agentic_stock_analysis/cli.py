import argparse
import logging
import sys
import yfinance as yf

from agentic_stock_analysis.core.log_config import setup_logging
from agentic_stock_analysis.ml.training import ensure_model_trained
from agentic_stock_analysis.ml.model import MODEL_PATH
from agentic_stock_analysis.services.analyze_service import analyze_ticker

setup_logging()
logger = logging.getLogger(__name__)


def is_valid_ticker(ticker: str) -> bool:
    """
    Validate the ticker symbol using yfinance by checking
    if there is any recent historical data.
    """
    try:
        # Try to fetch 1 month of data
        data = yf.Ticker(ticker).history(period="1mo")
        return data is not None and not data.empty
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Run stock prediction for a given ticker.")
    p.add_argument("ticker", type=str, help="Stock ticker symbol, e.g. AAPL")
    p.add_argument("--no-explain", action="store_true", help="Skip OpenAI explanation")
    p.add_argument(
        "--train-if-missing",
        action="store_true",
        help="If model is missing, train it using multi-ticker dataset",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.strip().upper()
    explain = not args.no_explain

    # Ensure model exists (optional, controlled)
    if not MODEL_PATH.exists():
        if args.train_if_missing:
            logger.warning("Model missing. Training model (this may take a while)...")
            ensure_model_trained(years=5, max_tickers=200, min_tickers=50)
        else:
            logger.error(
                f"Model is missing at {MODEL_PATH}. "
                "Run the API once (it trains on startup) or run CLI with --train-if-missing."
            )
            sys.exit(1)

    logger.info(f"Validating ticker: {ticker}")
    if not is_valid_ticker(ticker):
        logger.error(f"Ticker '{ticker}' is invalid or has no recent data.")
        sys.exit(1)

    logger.info(f"Starting prediction for {ticker}...")
    try:
        pred, indicators, explanation = analyze_ticker(ticker, explain=explain)
    except Exception as e:
        logger.exception(f"Failed to run analysis for {ticker}: {e}")
        sys.exit(1)

    logger.info(f"Prediction: {'UP' if pred == 1 else 'DOWN'}")
    logger.info(f"Indicators: {indicators}")

    if explanation:
        logger.info("AI Explanation:")
        logger.info(explanation)


if __name__ == "__main__":
    main()

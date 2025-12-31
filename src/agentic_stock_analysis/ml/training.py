from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
from yahoo_fin import stock_info as si

import pandas as pd

from agentic_stock_analysis.services.fetch_data import get_stock_data_batch
from agentic_stock_analysis.ml.features import compute_features
from agentic_stock_analysis.ml.model import train_model, MODEL_PATH
from agentic_stock_analysis.ml.ticker_data import SP500_TICKERS

logger = logging.getLogger(__name__)

# Cached training data (so we don't re-download every restart)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Cache for Tickers
TICKERS_CACHE = DATA_DIR / "sp500_tickers.txt"

# Current feature list
FEATURES = ["RSI", "EMA_10", "EMA_50", "MACD"]


@dataclass
class TrainingConfig:
    years: int = 5
    max_tickers: int = 200
    min_tickers: int = 50


def get_default_universe(max_tickers: int) -> list[str]:
    """
    Robust universe getter:
    1) Use cached tickers file if present
    2) Otherwise fall back to a built-in list
    3) Optionally try yahoo_fin and cache it.
    """
    # 1. Cached list (best)
    if TICKERS_CACHE.exists():
        tickers = [
            line.strip().upper()
            for line in TICKERS_CACHE.read_text().splitlines()
            if line.strip()
        ]
        logger.info(
            f"[train] loaded tickers from cache: {TICKERS_CACHE} count={len(tickers)}"
        )
        return tickers[:max_tickers]

    # 2. Built-in fallback list (no network)
    fallback = get_fallback_universe(max_tickers)
    logger.warning(
        "[train] ticker cache missing; using fallback universe (no network)."
    )

    # 3. Try network fetch
    try:
        tickers = [t.replace(".", "-").upper() for t in si.tickers_sp500()]
        if tickers:
            TICKERS_CACHE.write_text("\n".join(tickers))
            logger.info(f"[train] cached S&P500 tickers to {TICKERS_CACHE}")
            return tickers[:max_tickers]
    except Exception as e:
        logger.warning(
            f"[train] unable to fetch S&P500 tickers online (will continue with fallback): {e}"
        )

    return fallback


def get_fallback_universe(max_tickers: int) -> list[str]:
    """
    S&P 500 list of tickers.
    """
    return SP500_TICKERS[:max_tickers]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Next-day directional target.
    """
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=["Target"])
    return df


def build_training_dataset(tickers: List[str], years: int) -> pd.DataFrame:
    frames = []
    data_map = get_stock_data_batch(tickers, period="5y")
    for t, df in tqdm(data_map.items()):
        try:
            if df is None or df.empty:
                continue
            df = compute_features(df)
            df = add_target(df)

            # keep only needed columns
            needed = FEATURES + ["Target"]
            if not all(col in df.columns for col in needed):
                logger.warning(f"[train] missing columns for {t}, skipping")
                continue

            df = df[needed].dropna()
            df["Ticker"] = t  # optional, can be used later
            frames.append(df)
        except Exception as e:
            logger.exception(f"[train] failed ticker={t}: {e}")

    if not frames:
        raise RuntimeError("No training data collected. Check data provider / tickers.")

    dataset = pd.concat(frames, axis=0, ignore_index=True)
    return dataset


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_dataset(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def ensure_model_trained(
    years: int = 5, max_tickers: int = 400, min_tickers: int = 100
) -> None:
    """
    Train model once if it does not exist:
    - fetch panel dataset across tickers
    - save dataset locally
    - train and save model locally
    """
    if MODEL_PATH.exists():
        logger.info(f"[train] model already exists at {MODEL_PATH}")
        return

    tickers = get_default_universe(max_tickers=max_tickers)

    dataset_path = DATA_DIR / f"train_dataset_{years}y_{len(tickers)}t.parquet"

    # Prefer cached dataset if already created
    df = load_dataset(dataset_path)
    if df is not None and not df.empty:
        logger.info(f"[train] using cached dataset: {dataset_path} rows={len(df)}")
    else:
        df = build_training_dataset(tickers=tickers, years=years)
        logger.info(f"[train] built dataset rows={len(df)}. Saving to {dataset_path}")
        save_dataset(df, dataset_path)

    # If dataset is too small, fail fast
    if df["Ticker"].nunique() < min_tickers:
        raise RuntimeError(
            f"Collected data for only {df['Ticker'].nunique()} tickers; expected at least {min_tickers}."
        )

    logger.info(f"[train] training model, saving to {MODEL_PATH}")
    train_model(df=df, features=FEATURES, model_path=MODEL_PATH)
    logger.info("[train] model training complete")

from pathlib import Path
import pandas as pd
import yfinance as yf
import logging
import time

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "ml" / "data" / "raw_prices"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _chunked(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def get_stock_data(ticker, period="1y"):
    """
    Download historical OHLCV data for a given ticker using yfinance.
    """
    logger.info(f"Downloading {ticker} data for period={period}...")
    df = yf.download(ticker, period=period, progress=False)

    # Ensure we have a DataFrame with data
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # Flatten MultiIndex columns (e.g. ('Close','') -> 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    logger.info(f"Download complete. Rows: {len(df)}")
    return df


def get_stock_data_batch(
    tickers: list[str],
    period: str = "5y",
    interval: str = "1d",
    batch_size: int = 150,
    sleep_between_batches: float = 2.0,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for many tickers.
    Uses local cache to avoid re-downloading.
    """
    all_data: dict[str, pd.DataFrame] = {}

    tickers = [t.replace(".", "-").upper() for t in tickers]

    # Load cached data first
    if use_cache:
        for t in tickers:
            path = RAW_DATA_DIR / f"{t}.parquet"
            if path.exists():
                try:
                    all_data[t] = pd.read_parquet(path)
                except Exception:
                    logger.warning(f"[cache] failed reading {path}")

    missing = [t for t in tickers if t not in all_data]

    if not missing:
        logger.info("[data] all tickers loaded from cache")
        return all_data

    logger.info(f"[data] downloading {len(missing)} missing tickers")

    # Download only missing tickers
    for batch in _chunked(missing, batch_size):
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.exception(f"[data] batch failed: {e}")
            continue

        for t in batch:
            try:
                if t not in raw:
                    continue

                df = raw[t].dropna(how="all")
                if df.empty:
                    continue

                df = df.reset_index()
                all_data[t] = df

                # Save to cache
                df.to_parquet(RAW_DATA_DIR / f"{t}.parquet", index=False)

            except Exception as e:
                logger.debug(f"[data] failed ticker={t}: {e}")

        time.sleep(sleep_between_batches)

    logger.info(f"[data] completed. loaded={len(all_data)} / requested={len(tickers)}")

    return all_data

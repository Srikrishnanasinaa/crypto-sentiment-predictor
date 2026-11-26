"""Binance market data (public endpoints, no key required)."""
import logging
import time

import pandas as pd
import requests

from . import config
from .storage import append_unique

log = logging.getLogger(__name__)
KLINES_URLS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",  # public market-data mirror
]
COLUMNS = ["time", "coin", "price"]


def fetch_klines(symbol: str, interval: str = "5m", limit: int = 1000, end_time_ms=None) -> pd.DataFrame:
    """Return candles as DataFrame[time (UTC, naive), price (close)]."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    data, last_error = None, None
    for url in KLINES_URLS:
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as exc:  # blocked region, timeout, rate limit -> try the next server
            last_error = exc
            log.warning("Binance endpoint failed (%s): %s", url, exc)
    if data is None:
        raise last_error
    return pd.DataFrame({
        "time": pd.to_datetime([c[0] for c in data], unit="ms"),  # already UTC
        "price": [float(c[4]) for c in data],
    })


def fetch_history(symbol: str, days: int) -> pd.DataFrame:
    """Page backwards through Binance to get `days` of 5-minute candles."""
    needed = days * 24 * 60 // config.CANDLE_MINUTES
    frames, end = [], None
    while needed > 0:
        chunk = fetch_klines(symbol, "5m", min(1000, needed), end)
        if chunk.empty:
            break
        frames.append(chunk)
        needed -= len(chunk)
        end = int(chunk["time"].iloc[0].timestamp() * 1000) - 1
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=["time", "price"])
    return pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def update() -> int:
    """Create (first run: N days of history) or incrementally update the price store."""
    first_run = not config.PRICE_FILE.exists()
    total = 0
    for coin, symbol in config.COINS.items():
        try:
            df = (fetch_history(symbol, config.INITIAL_HISTORY_DAYS) if first_run
                  else fetch_klines(symbol, "5m", 10))
        except Exception as exc:
            log.warning("Price fetch failed for %s: %s", coin, exc)
            continue
        df["coin"] = coin
        append_unique(config.PRICE_FILE, df[COLUMNS], ["time", "coin"], "time")
        total += len(df)
    return total


def load(coin: str = None) -> pd.DataFrame:
    if not config.PRICE_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(config.PRICE_FILE, parse_dates=["time"])
    return df[df["coin"] == coin].reset_index(drop=True) if coin else df

"""Feature engineering shared by training and live prediction (so they can never disagree)."""
import numpy as np
import pandas as pd

from . import config
from .storage import utc_now


def market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add volatility and momentum_pct to a frame [time, price] sorted by time.

    Uses only past/current candles (rolling windows) -> no look-ahead.
    """
    df = prices.sort_values("time").reset_index(drop=True).copy()
    ret = df["price"].pct_change()
    df["volatility"] = ret.rolling(config.VOLATILITY_WINDOW).std()
    df["momentum_pct"] = df["price"].pct_change(config.MOMENTUM_WINDOW) * 100
    return df


def news_window_stats(news: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    """For each time t, aggregate headlines published in (t - window, t]."""
    n = news.sort_values("published_at")
    nt = n["published_at"].values.astype("datetime64[ns]")
    s = n["sentiment_score"].to_numpy(dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(s)])
    ca = np.concatenate([[0.0], np.cumsum(np.abs(s))])

    t = times.values.astype("datetime64[ns]")
    window = np.timedelta64(config.SENTIMENT_WINDOW_MINUTES, "m")
    hi = np.searchsorted(nt, t, side="right")
    lo = np.searchsorted(nt, t - window, side="right")
    count = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, (cs[hi] - cs[lo]) / count, 0.0)
        strength = np.where(count > 0, (ca[hi] - ca[lo]) / count, 0.0)
    return pd.DataFrame({"sentiment_mean": mean, "sentiment_strength": strength, "n_headlines": count})


def build_dataset(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Training table for one coin.

    Row = one 5-minute candle time t that had >=1 headline in the previous hour.
    Target = % price change from t to t + HORIZON (future price is used ONLY as the label).
    """
    if news.empty or prices.empty:
        return pd.DataFrame()
    mf = market_features(prices[["time", "price"]])
    stats = news_window_stats(news, mf["time"])
    df = pd.concat([mf, stats], axis=1)

    future = mf.set_index("time")["price"]
    target_time = mf["time"] + pd.Timedelta(minutes=config.HORIZON_MINUTES)
    df["price_future"] = future.reindex(target_time).to_numpy()
    df["target_pct"] = (df["price_future"] - df["price"]) / df["price"] * 100
    df["hour"] = df["time"].dt.hour

    df = df[df["n_headlines"] > 0].dropna(subset=config.FEATURES + ["target_pct"])
    return df.reset_index(drop=True)


def live_row(news: pd.DataFrame, prices_5m: pd.DataFrame) -> pd.DataFrame:
    """Single-row feature frame for 'now', built from the latest candles + headlines."""
    mf = market_features(prices_5m[["time", "price"]])
    last = mf.iloc[[-1]].copy()
    now = max(utc_now(), last["time"].iloc[0])
    stats = news_window_stats(news, pd.Series([now]))  # includes headlines after the last candle
    last = pd.concat([last.reset_index(drop=True), stats], axis=1)
    last["hour"] = int(now.hour)
    return last[config.FEATURES]

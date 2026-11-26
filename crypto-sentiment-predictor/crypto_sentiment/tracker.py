"""Live prediction log: every forecast is stored, then scored once its horizon has passed."""
import numpy as np
import pandas as pd

from . import config
from .evaluate import OVERLAP, directional_pvalue
from .storage import utc_now

COLUMNS = ["time", "coin", "price", "predicted_change_pct", "signal",
           "actual_change_pct", "correct", "settled"]


def load() -> pd.DataFrame:
    if not config.PRED_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(config.PRED_FILE, parse_dates=["time"])


def _save(df: pd.DataFrame) -> None:
    config.PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PRED_FILE, index=False)


def log(coin: str, price: float, result: dict, when=None) -> bool:
    """Record a forecast (one per coin per 5-minute slot). Returns False if already logged."""
    t = pd.Timestamp(when if when is not None else utc_now()).floor(f"{config.CANDLE_MINUTES}min")
    df = load()
    if ((df["time"] == t) & (df["coin"] == coin)).any():
        return False
    row = {"time": t, "coin": coin, "price": float(price),
           "predicted_change_pct": float(result["predicted_change_pct"]), "signal": result["signal"],
           "actual_change_pct": np.nan, "correct": np.nan, "settled": False}
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True


def settle(prices: pd.DataFrame) -> int:
    """Fill in actual outcomes for forecasts whose horizon has elapsed. Returns rows settled."""
    df = load()
    if df.empty:
        return 0
    horizon = pd.Timedelta(minutes=config.HORIZON_MINUTES)
    lookup = prices.set_index(["coin", "time"])["price"]
    done = 0
    for i in df.index[~df["settled"].astype(bool)]:
        key = (df.at[i, "coin"], df.at[i, "time"] + horizon)
        if key not in lookup.index:
            continue
        actual = (float(lookup.loc[key]) - df.at[i, "price"]) / df.at[i, "price"] * 100
        df.at[i, "actual_change_pct"] = actual
        df.at[i, "settled"] = True
        if df.at[i, "signal"] != "Neutral":
            df.at[i, "correct"] = float(np.sign(actual) == np.sign(df.at[i, "predicted_change_pct"]))
        done += 1
    if done:
        _save(df)
    return done


def scoreboard(coin: str = None) -> dict:
    df = load()
    if coin:
        df = df[df["coin"] == coin]
    settled = df[df["settled"].astype(bool)]
    directional = settled.dropna(subset=["correct"])
    out = {"n_logged": int(len(df)), "n_settled": int(len(settled)),
           "n_directional": int(len(directional)), "directional_accuracy": None, "p_value": None,
           "mean_abs_error_pct": None, "avg_actual_when_bullish": None, "avg_actual_when_bearish": None}
    if len(settled):
        out["mean_abs_error_pct"] = float((settled["actual_change_pct"] - settled["predicted_change_pct"]).abs().mean())
        for sig, key in (("Bullish", "avg_actual_when_bullish"), ("Bearish", "avg_actual_when_bearish")):
            s = settled[settled["signal"] == sig]
            out[key] = float(s["actual_change_pct"].mean()) if len(s) else None
    if len(directional):
        acc = float(directional["correct"].mean())
        out["directional_accuracy"] = acc
        out["p_value"] = directional_pvalue(acc, len(directional))
    return out

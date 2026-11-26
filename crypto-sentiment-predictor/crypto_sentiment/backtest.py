"""Backtest the model's signals on out-of-sample (walk-forward) predictions.

Rules: enter when the prediction exceeds the threshold, hold for the prediction horizon, pay
Binance-style fees on both sides, never hold two overlapping trades. Long-only by default (spot).
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

from . import config, evaluate

log = logging.getLogger(__name__)


def run_backtest(oos: pd.DataFrame, pred_col: str = "pred_random_forest",
                 threshold: float = config.SIGNAL_THRESHOLD_PCT, fee_pct: float = config.FEE_PCT,
                 allow_short: bool = False) -> dict:
    df = oos.sort_values("time").reset_index(drop=True)
    horizon = pd.Timedelta(minutes=config.HORIZON_MINUTES)
    trades, next_free = [], None

    for row in df.itertuples(index=False):
        t = pd.Timestamp(row.time)
        if next_free is not None and t < next_free:
            continue
        pred = getattr(row, pred_col)
        if pred > threshold:
            side = 1
        elif allow_short and pred < -threshold:
            side = -1
        else:
            continue
        net = side * row.target_pct / 100 - 2 * fee_pct / 100
        trades.append({"time": t, "side": side, "net_return": net})
        next_free = t + horizon

    span_days = max((pd.Timestamp(df["time"].iloc[-1]) - pd.Timestamp(df["time"].iloc[0])).days, 1)
    bh = (df["price"].iloc[-1] * (1 + df["target_pct"].iloc[-1] / 100)) / df["price"].iloc[0] - 1
    result = {
        "pred_col": pred_col, "threshold_pct": threshold, "fee_pct_per_side": fee_pct,
        "allow_short": allow_short, "n_trades": len(trades),
        "period_start": str(df["time"].iloc[0]), "period_end": str(df["time"].iloc[-1]),
        "buy_hold_return_pct": float(bh * 100),
    }
    if not trades:
        result.update(win_rate=None, total_return_pct=0.0, avg_trade_pct=None, sharpe=None,
                      max_drawdown_pct=0.0, equity={"time": [], "value": []})
        return result

    rets = np.array([t["net_return"] for t in trades])
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    sharpe = None
    if len(rets) >= 5 and rets.std() > 0:
        trades_per_year = len(rets) / span_days * 365
        sharpe = float(rets.mean() / rets.std() * np.sqrt(trades_per_year))
    result.update(
        win_rate=float((rets > 0).mean()),
        total_return_pct=float((equity[-1] - 1) * 100),
        avg_trade_pct=float(rets.mean() * 100),
        sharpe=sharpe,
        max_drawdown_pct=float(((equity / peak) - 1).min() * 100),
        equity={"time": [str(t["time"]) for t in trades], "value": [float(v) for v in equity]},
    )
    return result


def backtest_path(coin):
    return config.MODEL_DIR / f"backtest_{coin}.json"


def run(coin: str, pred_col: str = "pred_random_forest", allow_short: bool = False) -> dict:
    if not evaluate.oos_path(coin).exists():
        evaluate.run(coin)
    oos = pd.read_csv(evaluate.oos_path(coin), parse_dates=["time"])
    result = run_backtest(oos, pred_col, allow_short=allow_short)
    result["coin"] = coin
    result["synthetic_data"] = config.DEMO_MARKER.exists()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backtest_path(coin).write_text(json.dumps(result, indent=2))
    log.info("%s backtest | %d trades | return %.2f%% vs buy&hold %.2f%% | max DD %.2f%%", coin,
             result["n_trades"], result["total_return_pct"], result["buy_hold_return_pct"],
             result["max_drawdown_pct"])
    return result


def load(coin: str):
    p = backtest_path(coin)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Backtest model signals (out-of-sample)")
    ap.add_argument("--coin", choices=list(config.COINS))
    ap.add_argument("--model", default="random_forest", choices=list(evaluate.MODELS))
    ap.add_argument("--allow-short", action="store_true")
    args = ap.parse_args()
    for coin in ([args.coin] if args.coin else config.COINS):
        try:
            run(coin, f"pred_{args.model}", args.allow_short)
        except ValueError as exc:
            log.warning(str(exc))


if __name__ == "__main__":
    main()

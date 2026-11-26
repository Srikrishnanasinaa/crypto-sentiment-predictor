import numpy as np
import pandas as pd

from crypto_sentiment import backtest, config


def frame(n=1000, pred=1.0, target=1.0):
    t = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({"time": t, "price": 100.0, "target_pct": target, "pred_random_forest": pred})


def test_trades_never_overlap_and_net_of_fees():
    r = backtest.run_backtest(frame(), fee_pct=0.1)
    times = pd.to_datetime(r["equity"]["time"])
    assert (times.to_series().diff().dropna() >= pd.Timedelta(minutes=config.HORIZON_MINUTES)).all()
    assert abs(r["avg_trade_pct"] - 0.8) < 1e-9  # +1.0% minus 0.2% round-trip fees
    assert r["win_rate"] == 1.0
    assert r["max_drawdown_pct"] == 0.0


def test_fees_reduce_return():
    free = backtest.run_backtest(frame(), fee_pct=0.0)["total_return_pct"]
    paid = backtest.run_backtest(frame(), fee_pct=0.1)["total_return_pct"]
    assert paid < free


def test_no_signal_means_no_trades_and_no_crash():
    r = backtest.run_backtest(frame(pred=0.0))
    assert r["n_trades"] == 0 and r["total_return_pct"] == 0.0 and r["win_rate"] is None


def test_short_only_when_allowed():
    df = frame(pred=-1.0, target=-1.0)
    assert backtest.run_backtest(df, allow_short=False)["n_trades"] == 0
    r = backtest.run_backtest(df, allow_short=True)
    assert r["n_trades"] > 0 and r["total_return_pct"] > 0


def test_losing_strategy_has_drawdown():
    r = backtest.run_backtest(frame(pred=1.0, target=-1.0))
    assert r["total_return_pct"] < 0 and r["max_drawdown_pct"] < 0 and r["win_rate"] == 0.0

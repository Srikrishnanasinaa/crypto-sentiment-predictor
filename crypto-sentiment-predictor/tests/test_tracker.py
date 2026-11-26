import pandas as pd

from crypto_sentiment import config, tracker


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRED_FILE", tmp_path / "predictions.csv")


def prices_frame(t0, up=True):
    t = pd.date_range(t0, periods=30, freq="5min")
    price = [100.0 + i * (1 if up else -1) for i in range(30)]
    return pd.DataFrame({"time": t, "coin": "BTC", "price": price})


def test_log_dedupes_per_slot(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    res = {"predicted_change_pct": 0.5, "signal": "Bullish"}
    t0 = pd.Timestamp("2026-01-01 00:02")
    assert tracker.log("BTC", 100.0, res, when=t0) is True
    assert tracker.log("BTC", 100.0, res, when=t0 + pd.Timedelta(minutes=1)) is False  # same slot
    assert len(tracker.load()) == 1


def test_settles_only_after_horizon(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    t0 = pd.Timestamp("2026-01-01 00:00")
    tracker.log("BTC", 100.0, {"predicted_change_pct": 0.5, "signal": "Bullish"}, when=t0)
    early = prices_frame(t0).head(6)            # only 25 minutes of prices
    assert tracker.settle(early) == 0
    assert tracker.settle(prices_frame(t0)) == 1
    row = tracker.load().iloc[0]
    assert abs(row["actual_change_pct"] - 12.0) < 1e-9   # price 100 -> 112 after 12 candles
    assert row["correct"] == 1.0


def test_scoreboard_counts_wrong_and_ignores_neutral(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    t0 = pd.Timestamp("2026-01-01 00:00")
    tracker.log("BTC", 100.0, {"predicted_change_pct": -0.5, "signal": "Bearish"}, when=t0)
    tracker.log("BTC", 101.0, {"predicted_change_pct": 0.01, "signal": "Neutral"}, when=t0 + pd.Timedelta(minutes=5))
    tracker.settle(prices_frame(t0, up=True))
    sb = tracker.scoreboard("BTC")
    assert sb["n_settled"] == 2 and sb["n_directional"] == 1
    assert sb["directional_accuracy"] == 0.0    # predicted down, price went up

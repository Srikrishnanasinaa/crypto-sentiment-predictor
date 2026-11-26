import numpy as np
import pandas as pd
import pytest

from crypto_sentiment import config, model


def synthetic(n=6000):
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-01-01", periods=n, freq="5min")
    prices = pd.DataFrame({"time": t, "price": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))})
    nt = pd.date_range(t[0], t[-1], periods=1500)
    news = pd.DataFrame({"title": [f"h{i}" for i in range(1500)], "published_at": nt,
                         "sentiment_score": rng.uniform(-1, 1, 1500)})
    return news, prices


def test_train_saves_model_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    news, prices = synthetic()
    m = model.train("BTC", news=news, prices=prices)
    for key in ("mae", "baseline_mae", "r2", "directional_accuracy", "rows_test"):
        assert key in m
    assert (tmp_path / "model_BTC.pkl").exists()
    assert 0 <= m["directional_accuracy"] <= 1


def test_predict_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    news, prices = synthetic()
    model.train("BTC", news=news, prices=prices)
    row = pd.DataFrame([[0.4, 0.5, 3, 12, 0.002, 0.3]], columns=config.FEATURES)
    out = model.predict("BTC", row)
    assert out["signal"] in {"Bullish", "Bearish", "Neutral"}


def test_too_little_data_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    news, prices = synthetic(n=300)
    with pytest.raises(ValueError):
        model.train("BTC", news=news.head(5), prices=prices)

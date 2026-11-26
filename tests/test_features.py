import numpy as np
import pandas as pd

from crypto_sentiment import config, features


def make_prices(n=400, start="2026-01-01"):
    t = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({"time": t, "price": 100 + np.arange(n, dtype=float)})


def make_news(prices, score=0.5):
    t = prices["time"].iloc[30:300:7]
    return pd.DataFrame({"title": [f"h{i}" for i in range(len(t))], "published_at": t.values,
                         "sentiment_score": score})


def test_target_is_true_future_change():
    prices = make_prices()
    df = features.build_dataset(make_news(prices), prices)
    row = df.iloc[0]
    expected = (row["price_future"] - row["price"]) / row["price"] * 100
    assert np.isclose(row["target_pct"], expected)
    # price rises 1 per candle -> 12 candles ahead = +12
    assert np.isclose(row["price_future"] - row["price"], 12.0)


def test_features_do_not_use_future_news():
    prices = make_prices()
    news = make_news(prices)
    base = features.build_dataset(news, prices)
    # add a huge-sentiment headline in the far future; earlier rows must not change
    extra = pd.DataFrame({"title": ["future"], "published_at": [prices["time"].iloc[-1]],
                          "sentiment_score": [-1.0]})
    changed = features.build_dataset(pd.concat([news, extra], ignore_index=True), prices)
    cols = config.FEATURES
    merged = base.merge(changed, on="time", suffixes=("_a", "_b"))
    early = merged[merged["time"] < prices["time"].iloc[-1] - pd.Timedelta(hours=2)]
    for c in cols:
        assert np.allclose(early[c + "_a"], early[c + "_b"])


def test_window_stats_counts():
    prices = make_prices()
    news = make_news(prices)
    stats = features.news_window_stats(news, prices["time"])
    assert (stats["n_headlines"] >= 0).all()
    assert stats["sentiment_mean"].between(-1, 1).all()


def test_only_rows_with_news_kept():
    prices = make_prices()
    df = features.build_dataset(make_news(prices), prices)
    assert (df["n_headlines"] > 0).all()
    assert df[config.FEATURES].notna().all().all()

import pandas as pd
import pytest

from crypto_sentiment import config, demo, status
from crypto_sentiment.storage import utc_now


def setup(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", d)
    monkeypatch.setattr(config, "NEWS_FILE", d / "news.csv")
    monkeypatch.setattr(config, "PRICE_FILE", d / "prices.csv")
    monkeypatch.setattr(config, "PRED_FILE", d / "predictions.csv")
    monkeypatch.setattr(config, "DEMO_MARKER", d / ".demo")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")


def write(age_price_min, age_news_h):
    now = utc_now()
    pd.DataFrame({"time": [now - pd.Timedelta(minutes=age_price_min)], "coin": "BTC",
                  "price": [100.0]}).to_csv(config.PRICE_FILE, index=False)
    pd.DataFrame({"title": ["a"], "published_at": [now - pd.Timedelta(hours=age_news_h)],
                  "source": "x", "url": "", "sentiment_score": 0.1,
                  "sentiment_label": "Positive"}).to_csv(config.NEWS_FILE, index=False)


def test_status_live_when_fresh(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    write(3, 1)
    s = status.check()
    assert s["live"] and s["price_fresh"] and s["news_fresh"]


def test_status_not_live_when_stale(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    write(120, 1)
    assert not status.check()["live"]
    write(3, 48)
    assert not status.check()["live"]


def test_status_handles_empty_dir(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    s = status.check()
    assert s["live"] is False and s["price_rows"] == 0 and s["news_rows"] == 0


def test_status_demo_is_never_live(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    write(3, 1)
    config.DEMO_MARKER.write_text("synthetic")
    assert status.check()["live"] is False


def test_demo_refuses_to_overwrite_real_data(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    write(3, 1)                       # real data present, no demo marker
    with pytest.raises(SystemExit):
        demo.main([])
    assert not config.DEMO_MARKER.exists()

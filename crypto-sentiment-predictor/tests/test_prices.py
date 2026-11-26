import pandas as pd
import pytest

from crypto_sentiment import config, prices


class Resp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def candles(end_ms, limit, step=300_000):
    return [[end_ms - (limit - i) * step, 0, 0, 0, str(100 + i), 0] for i in range(limit)]


def test_falls_back_to_second_server(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if "api.binance.com" in url:
            raise RuntimeError("451 blocked in this region")
        return Resp(candles(1_700_000_000_000, params["limit"]))

    monkeypatch.setattr(prices.requests, "get", fake_get)
    df = prices.fetch_klines("BTCUSDT", "5m", 10)
    assert len(df) == 10 and len(calls) == 2 and "binance.vision" in calls[1]


def test_raises_when_all_servers_fail(monkeypatch):
    def boom(url, params=None, timeout=None):
        raise RuntimeError("down")

    monkeypatch.setattr(prices.requests, "get", boom)
    with pytest.raises(RuntimeError):
        prices.fetch_klines("BTCUSDT", "5m", 10)


def test_history_paginates_without_duplicates(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        end = params.get("endTime", 1_700_000_000_000)
        return Resp(candles(end, params["limit"]))

    monkeypatch.setattr(prices.requests, "get", fake_get)
    monkeypatch.setattr(prices.time, "sleep", lambda s: None)
    df = prices.fetch_history("BTCUSDT", days=7)
    assert len(df) == 7 * 24 * 12            # 2016 five-minute candles
    assert df["time"].is_unique and df["time"].is_monotonic_increasing


def test_update_creates_then_extends_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PRICE_FILE", tmp_path / "prices.csv")
    monkeypatch.setattr(config, "INITIAL_HISTORY_DAYS", 1)

    def fake_get(url, params=None, timeout=None):
        end = params.get("endTime", 1_700_000_000_000)
        return Resp(candles(end, params["limit"]))

    monkeypatch.setattr(prices.requests, "get", fake_get)
    monkeypatch.setattr(prices.time, "sleep", lambda s: None)
    prices.update()
    first = prices.load()
    assert set(first["coin"]) == {"BTC", "ETH", "SOL"}
    prices.update()                           # incremental run must not duplicate rows
    second = prices.load()
    assert second.duplicated(["time", "coin"]).sum() == 0

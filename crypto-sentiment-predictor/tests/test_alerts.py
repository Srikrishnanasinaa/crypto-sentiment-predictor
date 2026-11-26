from crypto_sentiment import alerts, config


class R:
    def raise_for_status(self):
        pass


def test_no_url_no_alert(monkeypatch):
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "")
    assert alerts.maybe_alert("BTC", {"predicted_change_pct": 1.0, "signal": "Bullish"}, 100) is False


def test_weak_signal_no_alert(monkeypatch):
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "http://x")
    assert alerts.maybe_alert("BTC", {"predicted_change_pct": 0.1, "signal": "Bullish"}, 100) is False


def test_strong_signal_posts(monkeypatch):
    sent = {}
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "http://x")
    monkeypatch.setattr(alerts.requests, "post", lambda url, json, timeout: sent.update(j=json) or R())
    assert alerts.maybe_alert("BTC", {"predicted_change_pct": 0.8, "signal": "Bullish"}, 100) is True
    assert "BTC" in sent["j"]["content"]

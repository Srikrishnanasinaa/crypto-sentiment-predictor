"""Optional Discord-compatible webhook alerts for strong signals."""
import logging

import requests

from . import config

log = logging.getLogger(__name__)


def maybe_alert(coin: str, result: dict, price: float) -> bool:
    url = config.ALERT_WEBHOOK_URL
    if not url or abs(result["predicted_change_pct"]) < config.ALERT_THRESHOLD_PCT:
        return False
    msg = (f"{result['signal']} signal for {coin}: predicted {result['predicted_change_pct']:+.2f}% "
           f"over the next hour (price ${price:,.2f}). Educational project, not financial advice.")
    try:
        requests.post(url, json={"content": msg}, timeout=10).raise_for_status()
        return True
    except Exception as exc:
        log.warning("Alert failed: %s", exc)
        return False

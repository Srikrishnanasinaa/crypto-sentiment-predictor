"""Health check: python -m crypto_sentiment.status  (exit code 1 if anything is stale)."""
import sys

import pandas as pd

from . import config, model, news, prices, tracker
from .storage import utc_now

PRICE_MAX_AGE_MIN = 15
NEWS_MAX_AGE_HOURS = 12


def check() -> dict:
    now = utc_now()
    p, n = prices.load(), news.load()
    out = {"now_utc": str(now), "demo_data": config.DEMO_MARKER.exists()}

    out["price_rows"] = int(len(p))
    if len(p):
        age = (now - p["time"].max()).total_seconds() / 60
        out.update(latest_price=str(p["time"].max()), price_age_min=round(age, 1),
                   price_fresh=age <= PRICE_MAX_AGE_MIN)
    else:
        out.update(latest_price=None, price_age_min=None, price_fresh=False)

    out["news_rows"] = int(len(n))
    if len(n):
        age = (now - n["published_at"].max()).total_seconds() / 3600
        out.update(latest_news=str(n["published_at"].max()), news_age_hours=round(age, 1),
                   news_fresh=age <= NEWS_MAX_AGE_HOURS,
                   headlines_24h=int((n["published_at"] > now - pd.Timedelta(hours=24)).sum()))
    else:
        out.update(latest_news=None, news_age_hours=None, news_fresh=False, headlines_24h=0)

    out["models"] = {c: model.model_path(c).exists() for c in config.COINS}
    sb = tracker.scoreboard()
    out["forecasts_logged"], out["forecasts_scored"] = sb["n_logged"], sb["n_settled"]
    out["live"] = bool(out["price_fresh"] and out["news_fresh"] and not out["demo_data"])
    return out


def main() -> None:
    s = check()
    ok = lambda b: "OK  " if b else "STALE"
    print(f"Time (UTC)      : {s['now_utc']}")
    print(f"Prices          : {ok(s['price_fresh'])} {s['price_rows']} rows, latest {s['latest_price']} "
          f"({s['price_age_min']} min old)")
    print(f"News            : {ok(s['news_fresh'])} {s['news_rows']} rows, latest {s['latest_news']} "
          f"({s['news_age_hours']} h old), {s['headlines_24h']} in last 24h")
    for coin, has in s["models"].items():
        print(f"Model {coin:<9} : {'trained' if has else 'not trained yet'}")
    print(f"Forecast log    : {s['forecasts_logged']} logged, {s['forecasts_scored']} scored")
    if s["demo_data"]:
        print("\nRESULT: DEMO (synthetic) data - not live.")
    elif s["live"]:
        print("\nRESULT: LIVE - data is fresh.")
    else:
        print("\nRESULT: NOT LIVE - start the collector: python -m crypto_sentiment.pipeline --loop")
    sys.exit(0 if s["live"] else 1)


if __name__ == "__main__":
    main()

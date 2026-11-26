"""Generate SYNTHETIC data to try the project offline: python -m crypto_sentiment.demo

Never quote results from demo data. A marker file makes the dashboard show a warning.
"""
import argparse

import numpy as np
import pandas as pd

from . import config, sentiment
from .storage import utc_now

HEADLINES = [
    "Bitcoin surges as ETF inflows hit record", "Ethereum rally continues after upgrade",
    "Exchange hacked, millions lost", "Regulators ban crypto trading in major market",
    "Bitcoin price steady as traders wait for Fed", "Solana adoption grows among developers",
    "Crypto market crashes amid liquidation wave", "Analysts turn bearish on altcoins",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write synthetic demo data")
    ap.add_argument("--force", action="store_true", help="overwrite real collected data")
    args = ap.parse_args(argv)
    if config.NEWS_FILE.exists() and not config.DEMO_MARKER.exists() and not args.force:
        raise SystemExit("Refusing to overwrite your REAL data in data/. Delete data/*.csv first, "
                         "or use --force if you are sure.")
    rng = np.random.default_rng(42)
    config.DATA_DIR.mkdir(exist_ok=True)
    end = utc_now().floor("5min")
    times = pd.date_range(end=end, periods=12 * 24 * 10, freq="5min")  # 10 days
    rows = []
    for coin, base in {"BTC": 90000, "ETH": 3200, "SOL": 180}.items():
        price = base * np.exp(np.cumsum(rng.normal(0, 0.0007, len(times))))
        rows.append(pd.DataFrame({"time": times, "coin": coin, "price": price}))
    pd.concat(rows).to_csv(config.PRICE_FILE, index=False)

    ntimes = pd.date_range(times[0], times[-1] - pd.Timedelta(hours=1), periods=900)
    titles = rng.choice(HEADLINES, len(ntimes))
    news = pd.DataFrame({"title": [f"{t} #{i}" for i, t in enumerate(titles)],
                         "published_at": ntimes, "source": "demo", "url": ""})
    sentiment.add_sentiment(news).to_csv(config.NEWS_FILE, index=False)
    config.DEMO_MARKER.write_text("synthetic")
    print("Synthetic demo data written. Now run: python -m crypto_sentiment.train")


if __name__ == "__main__":
    main()

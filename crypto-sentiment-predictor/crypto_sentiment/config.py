"""Central configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
NEWS_FILE = DATA_DIR / "news.csv"
PRICE_FILE = DATA_DIR / "prices.csv"
DEMO_MARKER = DATA_DIR / ".demo"
PRED_FILE = DATA_DIR / "predictions.csv"

COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Free, key-less news sources (verify they are reachable from your network).
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
]
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "").strip()

CANDLE_MINUTES = 5
HORIZON_MINUTES = 60            # predict price change this far ahead
SENTIMENT_WINDOW_MINUTES = 60   # headlines considered before "now"
VOLATILITY_WINDOW = 20          # candles
MOMENTUM_WINDOW = 12            # candles (= 1 hour)
INITIAL_HISTORY_DAYS = 7
MIN_TRAIN_ROWS = 200
SIGNAL_THRESHOLD_PCT = 0.05     # |prediction| below this => Neutral

# One feature list used by training AND live prediction.
FEATURES = [
    "sentiment_mean",
    "sentiment_strength",
    "n_headlines",
    "hour",
    "volatility",
    "momentum_pct",
]

# Backtesting / alerts
FEE_PCT = 0.1                    # Binance spot fee per side, in percent
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()   # Discord-compatible webhook
ALERT_THRESHOLD_PCT = 0.3        # alert only when |prediction| >= this

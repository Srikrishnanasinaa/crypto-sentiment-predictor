"""Real-time news collection: free RSS feeds (no key) + optional NewsData.io."""
import logging
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

from . import config, sentiment
from .storage import append_unique

log = logging.getLogger(__name__)
COLUMNS = ["title", "published_at", "source", "url"]
HEADERS = {"User-Agent": "crypto-sentiment-predictor/1.0"}


def _to_utc_naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def parse_rss(xml_text: str, source: str = "rss") -> pd.DataFrame:
    """Parse RSS 2.0 XML into a news frame. Bad items are skipped."""
    rows = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = " ".join((item.findtext("title") or "").split())
        pub = item.findtext("pubDate")
        if not title or not pub:
            continue
        try:
            published = _to_utc_naive(parsedate_to_datetime(pub))
        except (TypeError, ValueError):
            continue
        rows.append({"title": title, "published_at": published, "source": source,
                     "url": (item.findtext("link") or "").strip()})
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_rss(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return parse_rss(r.text, source=url.split("/")[2])
    except Exception as exc:  # network / XML errors must not stop the pipeline
        log.warning("RSS fetch failed for %s: %s", url, exc)
        return pd.DataFrame(columns=COLUMNS)


def fetch_newsdata() -> pd.DataFrame:
    if not config.NEWSDATA_API_KEY:
        return pd.DataFrame(columns=COLUMNS)
    try:
        r = requests.get(
            "https://newsdata.io/api/1/news",
            params={"apikey": config.NEWSDATA_API_KEY, "q": "bitcoin OR ethereum OR crypto",
                    "language": "en"},
            timeout=15,
        )
        r.raise_for_status()
        rows = []
        for p in r.json().get("results", []):
            if not p.get("title") or not p.get("pubDate"):
                continue
            rows.append({"title": " ".join(p["title"].split()),
                         "published_at": _to_utc_naive(pd.to_datetime(p["pubDate"], utc=True)),
                         "source": "newsdata.io", "url": p.get("link") or ""})
        return pd.DataFrame(rows, columns=COLUMNS)
    except Exception as exc:
        log.warning("NewsData.io fetch failed: %s", exc)
        return pd.DataFrame(columns=COLUMNS)


def fetch_latest() -> pd.DataFrame:
    """Fetch + score the newest headlines from all sources (nothing is saved)."""
    frames = [fetch_rss(u) for u in config.RSS_FEEDS] + [fetch_newsdata()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS + ["sentiment_score", "sentiment_label"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["title"])
    return sentiment.add_sentiment(df)


def collect() -> int:
    """Fetch, score and store new headlines. Returns the number of NEW rows added."""
    latest = fetch_latest()
    if latest.empty:
        return 0
    before = len(load())
    after = len(append_unique(config.NEWS_FILE, latest, ["title"], "published_at"))
    return after - before


def load() -> pd.DataFrame:
    if not config.NEWS_FILE.exists():
        return pd.DataFrame(columns=COLUMNS + ["sentiment_score", "sentiment_label"])
    return pd.read_csv(config.NEWS_FILE, parse_dates=["published_at"])

"""Backfill history so the models can be trained immediately (no multi-day wait).

News   : CryptoCompare news API (default; free, key optional) paged backwards in time.
         GDELT is available with --source gdelt but is strictly rate-limited.
Prices : Binance 5-minute candles.

Usage:  python -m crypto_sentiment.backfill --days 14
Optional: set CRYPTOCOMPARE_API_KEY in .env (free key from cryptocompare.com) for higher limits.
"""
import argparse
import logging
import os
import time

import pandas as pd
import requests

from . import config, news, prices, sentiment
from .storage import append_unique, utc_now

log = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = "(bitcoin OR ethereum OR solana OR crypto OR cryptocurrency) sourcelang:english"
COLUMNS = ["title", "published_at", "source", "url"]
CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/"
COINDESK_URL = "https://data-api.coindesk.com/news/v1/article/list"
REQUEST_GAP_SECONDS = 5.5          # GDELT asks for at most one request every 5 seconds
HEADERS = {"User-Agent": "crypto-sentiment-predictor/1.0"}


def parse_gdelt(payload: dict) -> pd.DataFrame:
    """Turn a GDELT 'artlist' JSON payload into a news frame. Bad rows are skipped."""
    rows = []
    for a in (payload or {}).get("articles", []):
        title = " ".join(str(a.get("title") or "").split())
        seen = pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce")
        if not title or pd.isna(seen):
            continue
        rows.append({"title": title, "published_at": seen,
                     "source": a.get("domain") or "gdelt", "url": a.get("url") or ""})
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_window(start: pd.Timestamp, end: pd.Timestamp, attempts: int = 3) -> pd.DataFrame:
    """Headlines first seen in [start, end). Retries when GDELT answers with a non-JSON error page."""
    params = {"query": QUERY, "mode": "artlist", "format": "json", "maxrecords": 250,
              "sort": "datedesc", "startdatetime": start.strftime("%Y%m%d%H%M%S"),
              "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(GDELT_URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            text = r.text.strip()
            if not text:                        # empty body = no articles in this window
                return pd.DataFrame(columns=COLUMNS)
            if not text.startswith("{"):        # e.g. "Please limit requests..." -> wait and retry
                raise ValueError(text[:80])
            return parse_gdelt(r.json())
        except Exception as exc:
            log.warning("GDELT window %s attempt %d/%d failed: %s", start, attempt, attempts, exc)
            time.sleep(10)
    return pd.DataFrame(columns=COLUMNS)


def _windows(days: int, window_hours: int):
    end = utc_now().floor("min")
    t, out = end - pd.Timedelta(days=days), []
    while t < end:
        out.append((t, min(t + pd.Timedelta(hours=window_hours), end)))
        t += pd.Timedelta(hours=window_hours)
    return out


def backfill_news(days: int = 14, window_hours: int = 6, gap: float = REQUEST_GAP_SECONDS) -> int:
    """Download, score and store headlines. Returns the number of NEW rows added."""
    windows, pending, added = _windows(days, window_hours), [], 0

    def flush() -> int:
        if not pending:
            return 0
        df = pd.concat(pending, ignore_index=True).drop_duplicates("title")
        pending.clear()
        before = len(news.load())
        after = len(append_unique(config.NEWS_FILE, sentiment.add_sentiment(df), ["title"], "published_at"))
        return after - before

    for i, (a, b) in enumerate(windows, 1):
        df = fetch_window(a, b)
        log.info("news window %d/%d (%s): %d headlines", i, len(windows), a.strftime("%Y-%m-%d %H:%M"), len(df))
        if not df.empty:
            pending.append(df)
        if i % 10 == 0 or i == len(windows):    # save progress regularly
            added += flush()
        if i < len(windows):
            time.sleep(gap)
    return added


def parse_cryptocompare(payload: dict) -> pd.DataFrame:
    """Turn a CryptoCompare news payload into a news frame. Bad rows are skipped."""
    data = (payload or {}).get("Data", [])
    rows = []
    if not isinstance(data, list):
        return pd.DataFrame(columns=COLUMNS)
    for a in data:
        title = " ".join(str(a.get("title") or "").split())
        ts = pd.to_numeric(a.get("published_on"), errors="coerce")
        if not title or pd.isna(ts):
            continue
        rows.append({"title": title, "published_at": pd.to_datetime(int(ts), unit="s"),  # UTC, naive
                     "source": a.get("source") or "cryptocompare", "url": a.get("url") or ""})
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_coindesk(payload: dict) -> pd.DataFrame:
    """Parse the CoinDesk Data news payload (keys are UPPER-case; lower-case is accepted too)."""
    data = (payload or {}).get("Data", [])
    rows = []
    if not isinstance(data, list):
        return pd.DataFrame(columns=COLUMNS)
    for a in data:
        g = lambda k: a.get(k.upper(), a.get(k.lower()))
        title = " ".join(str(g("TITLE") or "").split())
        ts = pd.to_numeric(g("PUBLISHED_ON"), errors="coerce")
        if not title or pd.isna(ts):
            continue
        src = g("SOURCE_DATA")
        name = src.get("NAME") if isinstance(src, dict) else None
        rows.append({"title": title, "published_at": pd.to_datetime(int(ts), unit="s"),
                     "source": name or g("SOURCE") or "coindesk", "url": g("URL") or ""})
    return pd.DataFrame(rows, columns=COLUMNS)


class RateLimited(Exception):
    """The news API said 429 Too Many Requests."""


_working_api = None      # remembers which news API worked, so we do not keep calling the broken one


def _request_page(api: str, before, key: str):
    """One request to one news API. Raises on HTTP / API errors."""
    headers = dict(HEADERS)
    if key:
        headers["authorization"] = f"Apikey {key}"
    if api == "coindesk":
        params = {"lang": "EN", "limit": 100}
        if before is not None:
            params["to_ts"] = int(before)
        if key:
            params["api_key"] = key
        url, parser = COINDESK_URL, parse_coindesk
    else:
        params = {"lang": "EN"}
        if before is not None:
            params["lTs"] = int(before)
        url, parser = CRYPTOCOMPARE_URL, parse_cryptocompare
    r = requests.get(url, params=params, headers=headers, timeout=30)
    if getattr(r, "status_code", 200) == 429:
        raise RateLimited("429 Too Many Requests")
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("Response", "Success")).lower() == "error":
        raise ValueError(payload.get("Message", "API error"))
    if payload.get("Err"):
        raise ValueError(str(payload["Err"])[:120])
    return parser(payload)


def _redact(text: str, key: str) -> str:
    """Never let the API key reach the screen or a log file."""
    return text.replace(key, "***") if key else text


def fetch_page(before=None, attempts: int = 2, rate_limit_waits: int = 6):
    """One page of headlines older than `before` (unix seconds). Tries the CoinDesk Data API and the
    older CryptoCompare API. Waits patiently when rate-limited. Returns None if everything fails."""
    global _working_api
    key = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()
    order = [_working_api] if _working_api else ["coindesk", "cryptocompare"]
    for api in order:
        attempt = waits = 0
        while attempt < attempts:
            try:
                page = _request_page(api, before, key)
                if _working_api != api:
                    log.info("using the %s news API", api)
                _working_api = api
                return page
            except RateLimited:
                waits += 1
                if waits > rate_limit_waits:
                    break
                log.warning("%s news API: too many requests, waiting 65 seconds (%d/%d)...",
                            api, waits, rate_limit_waits)
                time.sleep(65)
            except Exception as exc:
                attempt += 1
                log.warning("%s news API attempt %d/%d failed: %s", api, attempt, attempts,
                            _redact(str(exc), key)[:160])
                time.sleep(3)
    return None


def _store(frames, cutoff=None) -> int:
    """Score and merge headline frames into data/news.csv. Returns NEW rows added."""
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True).drop_duplicates("title")
    if cutoff is not None:
        df = df[df["published_at"] >= cutoff]
    if df.empty:
        return 0
    before = len(news.load())
    after = len(append_unique(config.NEWS_FILE, sentiment.add_sentiment(df), ["title"], "published_at"))
    return after - before


def backfill_news_cryptocompare(days: int = 14, gap: float = 3.5, max_pages: int = 600) -> int:
    """Page backwards through CryptoCompare news until `days` of history are collected."""
    cutoff = utc_now() - pd.Timedelta(days=days)
    pending, added, before, pages = [], 0, None, 0
    while pages < max_pages:
        pages += 1
        page = fetch_page(before)
        if page is None or page.empty:
            break
        pending.append(page)
        oldest = page["published_at"].min()
        log.info("news page %d: %d headlines, oldest %s", pages, len(page), oldest.strftime("%Y-%m-%d %H:%M"))
        if pages % 10 == 0:                     # save progress regularly
            added += _store(pending, cutoff)
            pending.clear()
        if oldest <= cutoff:
            break
        new_before = int(oldest.timestamp())
        if before is not None and new_before >= before:   # no progress: stop instead of looping forever
            break
        before = new_before
        time.sleep(gap)
    return added + _store(pending, cutoff)


def backfill_prices(days: int = 14) -> int:
    """Download `days` of 5-minute candles for every coin and merge into the price store."""
    total = 0
    for coin, symbol in config.COINS.items():
        try:
            df = prices.fetch_history(symbol, days)
        except Exception as exc:
            log.warning("Price backfill failed for %s: %s", coin, exc)
            continue
        df["coin"] = coin
        append_unique(config.PRICE_FILE, df[prices.COLUMNS], ["time", "coin"], "time")
        total += len(df)
    return total


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download history so you can train immediately")
    ap.add_argument("--days", type=int, default=14, help="days of history (1-90, default 14)")
    ap.add_argument("--source", choices=["cryptocompare", "gdelt"], default="cryptocompare")
    ap.add_argument("--window-hours", type=int, default=6, help="gdelt only")
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--skip-prices", action="store_true")
    args = ap.parse_args(argv)
    days = max(1, min(args.days, 90))

    if config.DEMO_MARKER.exists():
        raise SystemExit("Demo (fake) data is present. Delete data/news.csv, data/prices.csv and "
                         "data/.demo first, then run this again.")
    key = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()
    if key:
        log.info("API key found in .env (ends with ...%s)", key[-4:])
    else:
        log.warning("NO API key found. Create a file named .env in the project folder with the line "
                    "CRYPTOCOMPARE_API_KEY=yourkey and save it.")
    if not args.skip_prices:
        log.info("Downloading %d days of prices...", days)
        log.info("prices fetched: %d rows", backfill_prices(days))
    if not args.skip_news:
        if args.source == "cryptocompare":
            log.info("Downloading %d days of headlines (about 5-10 minutes, please wait)...", days)
            added = backfill_news_cryptocompare(days)
        else:
            steps = -(-days * 24 // args.window_hours)
            log.info("Downloading %d days of headlines (~%d minutes, please wait)...", days,
                     round(steps * REQUEST_GAP_SECONDS / 60) + 1)
            added = backfill_news(days, args.window_hours)
        log.info("new headlines stored: %d", added)
        if added == 0:
            print("\nNo headlines could be downloaded. Check your internet, or get a free key at "
                  "coindesk/cryptocompare and add CRYPTOCOMPARE_API_KEY=yourkey to the .env file.")
    print("\nDone. Now run:  python -m crypto_sentiment.train --full")


if __name__ == "__main__":
    main()

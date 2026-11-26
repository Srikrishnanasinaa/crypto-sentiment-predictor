"""Real-time collector: python -m crypto_sentiment.pipeline --loop"""
import argparse
import logging
import time

from . import alerts, config, evaluate, backtest, features, model, news, prices, tracker

log = logging.getLogger("pipeline")


def predict_and_log() -> None:
    """Make a live forecast per coin, log it, alert on strong signals, score matured forecasts."""
    news_df = news.load()
    for coin in config.COINS:
        if not model.model_path(coin).exists():
            continue
        try:
            p = prices.load(coin).tail(60)
            row = features.live_row(news_df, p)
            result = model.predict(coin, row)
            price = float(p["price"].iloc[-1])
            if tracker.log(coin, price, result):
                alerts.maybe_alert(coin, result, price)
        except Exception as exc:
            log.warning("Forecast skipped for %s: %s", coin, exc)
    settled = tracker.settle(prices.load())
    if settled:
        log.info("settled %d past forecasts", settled)


def run_once() -> None:
    added = news.collect()
    updated = prices.update()
    log.info("cycle done | +%d headlines | %d price rows fetched", added, updated)
    predict_and_log()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Collect news + prices (real time)")
    ap.add_argument("--loop", action="store_true", help="run forever (default: run once)")
    ap.add_argument("--interval", type=int, default=300, help="seconds between cycles")
    ap.add_argument("--retrain-every", type=int, default=0,
                    help="retrain models every N cycles (0 = never)")
    args = ap.parse_args()

    cycle = 0
    try:
        while True:
            cycle += 1
            try:
                run_once()
            except Exception:
                log.exception("cycle failed; continuing")
            if args.retrain_every and cycle % args.retrain_every == 0:
                for coin in config.COINS:
                    try:
                        model.train(coin)
                        evaluate.run(coin)
                        backtest.run(coin)
                    except ValueError as exc:
                        log.info(str(exc))
            if not args.loop:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()

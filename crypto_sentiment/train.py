"""CLI: python -m crypto_sentiment.train [--coin BTC]"""
import argparse
import logging

from . import backtest, config, evaluate, model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train per-coin models")
    ap.add_argument("--coin", choices=list(config.COINS), help="default: all coins")
    ap.add_argument("--full", action="store_true", help="also run walk-forward comparison + backtest")
    args = ap.parse_args()
    for coin in ([args.coin] if args.coin else config.COINS):
        try:
            model.train(coin)
            if args.full:
                evaluate.run(coin)
                backtest.run(coin)
        except ValueError as exc:
            logging.warning(str(exc))


if __name__ == "__main__":
    main()

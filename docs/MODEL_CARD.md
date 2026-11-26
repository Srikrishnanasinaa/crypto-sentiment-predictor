# Model card

**Task:** regress the percentage price change of BTC / ETH / SOL over the next 60 minutes from recent news sentiment and market features. A signal (Bullish / Bearish / Neutral) is derived with a ±0.05% dead zone.

**Models compared:** mean baseline, Ridge, Random Forest (deployed), Gradient Boosting.
**Deployed model:** Random Forest (300 trees, depth 6, min leaf 5), one per coin.

## Data
- News: public crypto RSS feeds (+ optional NewsData.io), headlines only, English.
- Prices: Binance spot 5-minute closes (USDT pairs).
- Sentiment: VADER compound score with a crypto-specific lexicon.

## Evaluation
Walk-forward validation (5 folds, expanding window, gap = horizon). Metrics: MAE vs mean baseline, directional accuracy with an overlap-adjusted p-value, and a fee-aware backtest. A live scoreboard scores every real forecast after one hour.

## Results (fill in from your own data)
| Coin | MAE | Baseline MAE | Direction acc. | p-value | Backtest return | Buy & hold | Max drawdown |
|---|---|---|---|---|---|---|---|
| BTC | | | | | | | |

## Limitations
- Headlines carry a weak, noisy signal for 1-hour moves; a result near the baseline is expected.
- Headline text only: no social media, order book, on-chain data or macro events.
- VADER is lexicon-based and misses context and sarcasm.
- News arrives mostly after the market has moved; causality is not claimed.
- Training data covers a short window and may not reflect other market regimes.
- Backtest ignores slippage, latency and order-book depth.

## Intended use
Education and research only. Not financial advice and not suitable for automated trading.

## Maintenance
The collector can retrain hourly (`--retrain-every`). Re-check the walk-forward comparison and live scoreboard after retraining to detect concept drift.

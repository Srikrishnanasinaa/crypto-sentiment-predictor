# Interview guide

> Read every module once before interviews. Only claim what you can explain. Fill the numbers table from YOUR real run.

## 30-second pitch
"I built a real-time system that reads crypto news, scores sentiment, combines it with live Binance prices, and predicts the next-hour price move for BTC, ETH and SOL. What I'm proudest of is the evaluation: walk-forward validation, a significance test that accounts for overlapping samples, a fee-aware backtest, and a live scoreboard that scores each forecast an hour later. I try to disprove my own results rather than just report an accuracy number."

## 2-minute walkthrough
1. **Problem:** can news sentiment help predict short-term crypto moves?
2. **Data:** RSS headlines + Binance 5-minute candles, collected continuously.
3. **Method:** VADER with a crypto lexicon; rolling 60-minute sentiment, volatility, momentum; Random Forest per coin.
4. **Evaluation:** chronological split with a gap, walk-forward over 5 folds against a naive baseline, overlap-adjusted p-value, backtest with fees.
5. **Result:** state your real numbers and what they mean (even if "no significant edge").
6. **Engineering:** tests, CI, Docker, health check, failure handling.
7. **Next:** FinBERT, more features (funding rates, Fear & Greed), database storage.

## 5-minute live demo script
1. Run `check_live.bat` (or `python -m crypto_sentiment.status`): show the data is fresh.
2. Open the dashboard **Live** tab: price matches Binance, headlines with sentiment, forecast signal.
3. **Model comparison** tab: "the model only matters if it beats the baseline; here is what happened."
4. **Backtest** tab: equity curve vs buy-and-hold, fees included.
5. **Live scoreboard**: "these forecasts were made before outcomes existed, so they cannot be tuned afterwards."
6. Show the GitHub repo: green CI badge, tests, docs.

## Numbers to memorise (fill in)
| Item | Value |
|---|---|
| Days of data collected | |
| Headlines / training rows | |
| Best model, MAE vs baseline | |
| Direction accuracy, p-value | |
| Backtest return vs buy-and-hold, trades, max drawdown | |
| Live forecasts scored | |

## Technical Q&A
**What is data leakage and where could it happen here?**
Using information from the future. Examples: using the future price as a feature, shuffling time series, or overlapping train/test labels. I use only past data for features, the future price only as the label, a chronological split, and a gap equal to the horizon. A unit test checks that adding a future headline does not change earlier features.

**Why compare against a baseline?**
Without it, a number like 0.2% MAE means nothing. Predicting the average is the bar every model must clear.

**Why is the p-value computed on n/12 samples?**
Rows are 5 minutes apart but each label spans 60 minutes, so about 12 consecutive rows are nearly the same observation. Treating them as independent makes weak accuracy look significant (52% on 10,000 rows looks "highly significant" naively; it is not).

**Why walk-forward instead of one split or k-fold?**
Random k-fold lets the model train on the future. Walk-forward mimics deployment and shows whether performance is stable across time.

**Why Random Forest?**
Non-linear interactions with little tuning, robust to scale, interpretable importances. I still compared it with Ridge and Gradient Boosting rather than assuming.

**What are VADER's limits?**
Lexicon-based, no context or sarcasm; I extended it with crypto terms (hack, surge, ban). A transformer such as FinBERT is the next step, and I would compare them with the same evaluation.

**Why 5-minute snapshots instead of one row per headline?**
Live prediction must work at any moment, not only when a headline arrives. A rolling window gives the same feature definition in training and production.

**How does the backtest avoid fooling itself?**
It uses out-of-sample walk-forward predictions, non-overlapping trades, 0.1% fee per side, long-only spot rules, and compares with buy-and-hold. It still ignores slippage and latency, which I state openly.

**What if the model doesn't beat the baseline?**
Then the honest finding is that headline sentiment alone has no detectable edge at this horizon. The system's value is that it can detect that.

**How would you detect concept drift?**
Retrain on a schedule, track the live scoreboard's rolling accuracy, and alert when it falls below the baseline.

**What breaks in production and how is it handled?**
Feeds going down, API rate limits, duplicate headlines, stale data. Failures are logged and skipped, headlines are de-duplicated by title, and the `status` command reports freshness.

**How would you scale this?**
Move CSV to Postgres, collectors to a queue (Kafka), schedule with Airflow, serve the model behind an API, monitor with Prometheus/Grafana.

**How did you test it?**
30+ pytest tests: leakage, alignment, RSS parsing, training, walk-forward gaps, backtest arithmetic, tracker settlement, health check. CI runs them on every push.

## Stories (STAR)
**A bug I found:** my first version mixed local-time price stamps with UTC news times, shifting every alignment by hours. *Situation:* results looked plausible but were wrong. *Action:* checked timestamps against raw API data, switched everything to UTC and added tests. *Result:* alignments verified by a unit test.
**A flaw I fixed:** the original dashboard fed the model a future price as a feature. I removed it, shared one feature function between training and serving, and added a test that proves no future information is used.
**A mistake in my own test:** I assumed 55% accuracy on 10,000 rows was not significant; the math showed it was. I corrected the test, which taught me to verify statistical claims rather than trust intuition.

## Things NOT to say
- "It predicts the market" or "it makes money."
- Any number you did not produce yourself from your own data.
- That synthetic-demo metrics are real results.

## Questions to ask them
- How do you evaluate models when labels overlap in time?
- How do you monitor models for drift in production?

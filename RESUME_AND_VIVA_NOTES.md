# Resume entry (copy-paste; fill [brackets] with YOUR real numbers)

**Real-Time Crypto Price Prediction using News Sentiment** | Python, scikit-learn, VADER, Streamlit, Plotly, Docker, GitHub Actions | github.com/<you>/crypto-sentiment-predictor
- Built a real-time pipeline ingesting crypto news (RSS/NewsData.io) and Binance 5-minute candles for BTC, ETH and SOL, with de-duplication and fault-tolerant incremental updates.
- Engineered leakage-free features (crypto-tuned VADER sentiment, rolling volatility/momentum) and trained per-coin Random Forest models with a chronological train/test split and a gap equal to the prediction horizon.
- Benchmarked baseline, Ridge, Random Forest and Gradient Boosting with walk-forward validation (5 folds); MAE [x]% vs baseline [y]%, direction accuracy [z]% (p = [p], overlap-adjusted).
- Built a fee-aware backtester (non-overlapping trades, Sharpe, max drawdown): [r]% vs [bh]% buy-and-hold over [n] trades.
- Built a live prediction tracker that scores every forecast one hour later ([k] out-of-sample forecasts so far).
- Shipped a 4-tab Streamlit dashboard, Dockerized the collector + app, and added a 25+ test pytest suite with GitHub Actions CI.

# One-liner
Real-time crypto forecasting system combining news sentiment (VADER) and market features with a Random Forest, served via a live Streamlit dashboard.

# Be ready to explain in interviews / viva
1. **Why does the model barely beat (or not beat) the baseline?** Crypto prices are close to a random walk over 1 hour; headline sentiment is a weak signal. Reporting this honestly shows maturity.
2. **What is data leakage and how did you avoid it?** Features only use data up to time t; the future price is only the label; a gap separates train/test; a unit test checks it.
3. **Why chronological split, not random?** Random shuffling lets the model see the future.
4. **Why a Random Forest?** Handles non-linear feature interactions with little tuning; feature importances are interpretable.
5. **Changes vs. the original report (be upfront):** news source is RSS + optional NewsData.io (the report's URL/CryptoPanic mention was inconsistent); events are 5-minute snapshots using a 1-hour sentiment window instead of one row per headline; API keys moved to `.env`; models are per coin.

6. **What is walk-forward validation?** Train on the past, test on the next block, move forward in time, repeat. It mimics real deployment and avoids the optimism of a single split.
7. **Why shrink the sample size in the p-value?** Rows are 5 minutes apart but each label covers 60 minutes, so ~12 consecutive rows are nearly the same sample. Treating them as independent would make weak accuracy look significant.
8. **What if the backtest loses money?** Say so. Fees (0.2% round trip) usually exceed the small edge from headlines; showing you can detect that is the point. A flat or negative result with a rigorous method is a stronger answer than a flattering number from a flawed one.
9. **What does the live scoreboard prove?** Each forecast is stored before its outcome exists, so it cannot be tuned after the fact. It is the cleanest evidence of whether the system works.

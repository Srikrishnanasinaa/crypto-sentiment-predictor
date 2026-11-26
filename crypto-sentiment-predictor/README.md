# 📈 Crypto News Sentiment & Price Prediction

![CI](https://github.com/YOUR-USERNAME/crypto-sentiment-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A real-time **research and forecasting system** that reads crypto news, scores sentiment with **VADER** (extended with crypto vocabulary), combines it with live **Binance** price features, and uses a **Random Forest** to forecast the next-hour price change for **BTC, ETH and SOL**. A **Streamlit** dashboard auto-refreshes with live prices, sentiment, the forecast signal and honest model metrics.

> B.Tech final-year capstone, CSE (Data Science), Audisankara College of Engineering & Technology (JNTUA), 2022-2026.
> Educational project - **not financial advice**.

<!-- Add screenshots here: docs/dashboard.png -->

## Documentation
[Architecture](docs/ARCHITECTURE.md) · [Model card](docs/MODEL_CARD.md) · [Interview guide](docs/INTERVIEW_GUIDE.md) · [GitHub setup](docs/GITHUB_SETUP.md) · [Changelog](CHANGELOG.md)

## How it works
```
RSS feeds (CoinDesk, Cointelegraph, Decrypt, ...)  ─┐
NewsData.io (optional)                              ─┴► news.py ─► sentiment.py (VADER + crypto lexicon) ─┐
                                                                                                         ├─► features.py ─► model.py (Random Forest) ─► dashboard
Binance public API (5-min candles)  ─► prices.py ────────────────────────────────────────────────────────┘
```
| Module | Role |
|---|---|
| `news.py` | Pulls headlines from free RSS feeds (no API key) + optional NewsData.io; de-duplicates, stores UTC timestamps |
| `prices.py` | Binance candles with paginated history and incremental updates |
| `sentiment.py` | VADER compound score + domain terms (hack, surge, ban, bullish...) |
| `features.py` | Rolling-window sentiment, volatility, momentum, hour - **one code path for training and live prediction** |
| `model.py` | Chronological split with a gap, per-coin Random Forest, evaluation vs a naive baseline |
| `pipeline.py` | Real-time collector loop with optional periodic retraining |
| `app/dashboard.py` | Auto-refreshing Streamlit dashboard |

## What makes it different
Most student projects stop at "accuracy". This one tries to **falsify its own results**:
| Feature | What it does |
|---|---|
| Walk-forward comparison | Baseline vs Ridge vs Random Forest vs Gradient Boosting over 5 forward-moving time folds |
| Honest significance test | Direction-accuracy p-value that shrinks overlapping samples (otherwise 52% looks "significant") |
| Backtester | Trades the out-of-sample signals with 0.1% fees per side, non-overlapping trades, Sharpe, max drawdown, vs buy-and-hold |
| Live scoreboard | Every live forecast is logged and scored one hour later - a real out-of-sample track record |
| Alerts | Optional Discord webhook for strong signals |

## Engineering decisions
- **No look-ahead leakage:** features use only past data; the future price is used only as the label (covered by unit tests).
- **Gap between train and test** equal to the prediction horizon, so overlapping windows cannot leak.
- **Honest evaluation:** MAE is compared against a naive "predict the average" baseline, with R² and directional accuracy. The dashboard warns when the model does not beat the baseline.
- **Resilient collection:** a failing feed or API is logged and skipped; the loop keeps running.
- **Tested & CI:** pytest suite runs on every push via GitHub Actions.

## Quick start
```bash
git clone https://github.com/YOUR-USERNAME/crypto-sentiment-predictor.git
cd crypto-sentiment-predictor
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add NEWSDATA_API_KEY

python -m crypto_sentiment.pipeline --loop --retrain-every 12   # leave running (collects every 5 min)
python -m crypto_sentiment.train --full                         # train + walk-forward comparison + backtest
streamlit run app/dashboard.py
```
RSS feeds only return recent articles, so **let the collector run for a few days** to build a training set (training needs 200+ rows).

### Try it instantly with synthetic data
```bash
python -m crypto_sentiment.demo && python -m crypto_sentiment.train --full && python -m streamlit run app/dashboard.py
```
The dashboard shows a warning banner for synthetic data. Never quote its metrics.

### Windows shortcuts (double-click)
| File | What it does |
|---|---|
| `run_demo.bat` | Try everything with synthetic data (will not overwrite real data) |
| `start_live.bat` | Real-time collector + dashboard |
| `check_live.bat` | Is the data fresh? (`python -m crypto_sentiment.status`) |

### Is it live?
```bash
python -m crypto_sentiment.status
```
Prints data freshness, model status and the forecast log, and says **LIVE** or **NOT LIVE**.

### Docker
```bash
docker compose up --build      # collector + dashboard on http://localhost:8501
```

## Results
After `train --full` on **your own** data, copy the numbers from the dashboard's *Model comparison*, *Backtest* and *Live scoreboard* tabs (or `models/comparison_*.json`, `models/backtest_*.json`):

| Coin | Best model | MAE vs baseline | Direction acc. (p-value) | Backtest return vs buy&hold | Max drawdown | Live scored forecasts |
|---|---|---|---|---|---|---|
| BTC | | | | | | |

Short-term price prediction from headlines is a hard, noisy problem; results close to the baseline are expected and are reported as-is.

## Tests
```bash
pip install -r requirements-dev.txt && python -m pytest -q
```

## Optional alerts
Set `ALERT_WEBHOOK_URL` in `.env` (a Discord channel webhook) to receive a message when a forecast is at least 0.3% in either direction.

## Limitations & future work
- Headlines only; add social media (Reddit/X) and on-chain data.
- Add the Fear & Greed index and funding rates as features.
- Try FinBERT or a fine-tuned transformer instead of VADER.
- Add LSTM / gradient boosting and walk-forward cross-validation.
- Store data in SQLite/Postgres instead of CSV.

## Author
**Nasina Sri Krishna** - B.Tech CSE (Data Science)

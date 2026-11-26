# Architecture

## System overview
```mermaid
flowchart LR
    subgraph Sources
        RSS[Crypto RSS feeds]
        ND[NewsData.io - optional]
        BN[Binance public API]
    end
    RSS --> NEWS[news.py]
    ND --> NEWS
    NEWS --> SENT[sentiment.py<br/>VADER + crypto lexicon]
    BN --> PRICE[prices.py]
    SENT --> CSVN[(data/news.csv)]
    PRICE --> CSVP[(data/prices.csv)]
    CSVN --> FEAT[features.py]
    CSVP --> FEAT
    FEAT --> TRAIN[model.py<br/>Random Forest per coin]
    FEAT --> EVAL[evaluate.py<br/>walk-forward, 4 models]
    EVAL --> BT[backtest.py<br/>fees, Sharpe, drawdown]
    TRAIN --> PRED[live forecast]
    FEAT --> PRED
    PRED --> TRK[tracker.py<br/>score after 1 hour]
    PRED --> ALR[alerts.py]
    TRAIN --> DASH[Streamlit dashboard]
    EVAL --> DASH
    BT --> DASH
    TRK --> DASH
```

## Live forecast cycle (every 5 minutes)
```mermaid
sequenceDiagram
    participant P as pipeline.py
    participant N as news.py
    participant M as prices.py
    participant F as features.py
    participant R as model.py
    participant T as tracker.py
    P->>N: collect + score new headlines
    P->>M: append newest candles
    P->>F: build ONE feature row for "now"
    F->>R: predict next-hour change
    R->>T: log forecast (price, prediction, signal)
    P->>T: settle forecasts older than 1 hour
```

## Data model
| File | Columns |
|---|---|
| `data/news.csv` | title, published_at (UTC), source, url, sentiment_score, sentiment_label |
| `data/prices.csv` | time (UTC), coin, price (5-minute close) |
| `data/predictions.csv` | time, coin, price, predicted_change_pct, signal, actual_change_pct, correct, settled |
| `models/model_<COIN>.pkl` | trained Random Forest |
| `models/comparison_<COIN>.json`, `backtest_<COIN>.json` | evaluation results |

## Features (identical for training and live)
`sentiment_mean`, `sentiment_strength`, `n_headlines` (headlines in the previous 60 min), `hour` (UTC), `volatility` (std of 5-min returns, 20 candles), `momentum_pct` (previous 1-hour return).
Label: % price change from t to t + 60 min.

## Key design decisions
| Decision | Why |
|---|---|
| One feature function for training and live | Prevents train/serve skew |
| Rolling 60-min sentiment window | Defined at any time t, so live prediction is possible |
| Label uses future price only as target | No look-ahead leakage; covered by unit tests |
| Gap = horizon between train and test | Overlapping labels cannot leak across the split |
| p-value uses ~n/12 effective samples | Rows 5 min apart share a 60-min label, so they are not independent |
| Per-coin models | Different volatility and news sensitivity |
| CSV storage | Simple and inspectable; SQLite/Postgres is the next step |
| Failures are logged, not fatal | One broken feed must not stop the collector |

## Failure modes
| Failure | Behaviour |
|---|---|
| RSS feed down | Warning logged, other feeds continue |
| Binance unreachable | Falls back to the public market-data mirror; if both fail the cycle logs a warning and the dashboard shows an error instead of crashing |
| No trained model | Dashboard explains how to train; collector skips forecasting |
| Not enough rows | Training refuses with a clear message instead of fitting noise |
| Stale data | `python -m crypto_sentiment.status` reports NOT LIVE |

## Scaling path
CSV → SQLite/Postgres; cron loop → scheduler (Airflow/Prometheus alerts); single process → message queue (Kafka) between collectors and feature builder; VADER → FinBERT served behind an API.

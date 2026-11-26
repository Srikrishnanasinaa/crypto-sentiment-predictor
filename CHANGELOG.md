# Changelog

## v1.0.1
- Dashboard works with current Streamlit (width="stretch"), with a fallback for older versions
- Binance requests fall back to the public market-data mirror if the main server is blocked
- Added price-collector tests (fallback, pagination, incremental update)

## v1.0.0
- Real-time pipeline: RSS + optional NewsData.io news, Binance 5-minute candles (BTC, ETH, SOL)
- VADER sentiment with a crypto-specific lexicon
- Leakage-free feature engineering shared by training and live prediction
- Per-coin Random Forest with chronological split and horizon gap
- Walk-forward comparison of four models with overlap-adjusted significance test
- Fee-aware backtester (Sharpe, max drawdown, buy-and-hold comparison)
- Live prediction tracker scored one hour later
- Streamlit dashboard (Live / Model comparison / Backtest / Live scoreboard)
- Health check (`python -m crypto_sentiment.status`), Docker, CI, 30+ tests

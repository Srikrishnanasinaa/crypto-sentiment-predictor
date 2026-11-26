# Final checklist (do in order)

1. [ ] Revoke the old API key from the original report (it was printed in the PDF). Put the new key only in `.env`.
2. [ ] Double-click `start_live.bat` and leave it running 2-3 days. `check_live.bat` should say LIVE.
3. [ ] Run `python -m crypto_sentiment.train --full`. Write your real numbers in the table below.
4. [ ] Push to GitHub, wait for the green CI tick, add dashboard screenshots to the README.
5. [ ] Add repo description + topics, then pin it (see `GITHUB_SETUP.md`).
6. [ ] Put the resume bullet first under Projects, using your real numbers.
7. [ ] Practise the pitch and the 5-minute demo out loud twice (`INTERVIEW_GUIDE.md`).

## My numbers (fill from my own run)
| Item | Value |
|---|---|
| Days of data collected | |
| Training rows | |
| Best model, MAE vs baseline | |
| Direction accuracy, p-value | |
| Backtest return vs buy-and-hold, trades, max drawdown | |
| Live forecasts scored | |

## 30-second pitch
"I built a real-time system that reads crypto news, scores sentiment, combines it with live Binance prices, and forecasts the next-hour move. What I'm proudest of is the evaluation: walk-forward validation, a significance test that accounts for overlapping samples, a fee-aware backtest, and a live scoreboard that scores each forecast an hour later. I tried to disprove my own results instead of just reporting an accuracy number."

## Five answers to know cold
1. **Data leakage:** only past data for features, future price only as the label, chronological split, gap equal to the horizon.
2. **Baseline:** a number means nothing unless it beats "predict the average".
3. **Overlap-adjusted p-value:** rows 5 minutes apart share a 60-minute label, so about 12 rows count as one sample.
4. **Backtest losing money:** fees (0.2% round trip) often exceed the small edge from headlines; detecting that is the point.
5. **Next steps:** FinBERT, more features (funding rates, Fear & Greed), Postgres instead of CSV.

## Rules
- Only say numbers you produced from your own real data.
- Never say "it predicts the market" or "it makes money".
- A rigorous "no significant edge" is a strong answer. Say it confidently.
- Read each code file once before interviews.

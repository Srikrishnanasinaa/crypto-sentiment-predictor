"""Live Streamlit dashboard. Run: streamlit run app/dashboard.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from crypto_sentiment import backtest, config, evaluate, features, model, news as news_mod, prices as prices_mod, tracker
from crypto_sentiment.storage import utc_now


def _stretch(fn, *args, **kwargs):
    """Full-width rendering that works on old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def show_chart(fig):
    return _stretch(st.plotly_chart, fig)


def show_table(df):
    return _stretch(st.dataframe, df, hide_index=True)


st.set_page_config(page_title="Crypto Sentiment Predictor", page_icon="📈", layout="wide")
st.title("📈 Crypto News Sentiment & Price Prediction")
st.caption("Real-time headlines (VADER sentiment) + Binance prices → Random Forest 1-hour forecast. "
           "Educational project, not financial advice.")

if config.DEMO_MARKER.exists():
    st.warning("Showing SYNTHETIC demo data. Results are not real.")

coin = st.sidebar.selectbox("Coin", list(config.COINS))
refresh = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60, step=30)
st.sidebar.markdown("Data: Binance public API, crypto RSS feeds.")


@st.cache_data(ttl=60, show_spinner=False)
def live_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    return prices_mod.fetch_klines(symbol, interval, limit)


@st.cache_data(ttl=120, show_spinner=False)
def live_news() -> pd.DataFrame:
    return news_mod.fetch_latest()


def all_news() -> pd.DataFrame:
    stored, live = news_mod.load(), live_news()
    frames = [f for f in (stored, live) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=news_mod.COLUMNS + ["sentiment_score", "sentiment_label"])
    return (pd.concat(frames, ignore_index=True).drop_duplicates("title")
            .sort_values("published_at").reset_index(drop=True))


@st.fragment(run_every=refresh)
def live_panel():
    symbol = config.COINS[coin]
    try:
        c1m = live_candles(symbol, "1m", 120)
        c5m = live_candles(symbol, "5m", 60)
    except Exception as exc:
        st.error(f"Could not reach Binance: {exc}")
        return
    news_df = all_news()
    now = utc_now()
    recent = news_df[news_df["published_at"] > now - pd.Timedelta(hours=1)]
    day = news_df[news_df["published_at"] > now - pd.Timedelta(hours=24)]

    last, first = c1m["price"].iloc[-1], c1m["price"].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric(f"{coin}/USDT", f"${last:,.2f}", f"{(last / first - 1) * 100:+.2f}% (2h)")
    b.metric("Headlines (1h)", len(recent))
    b2 = recent["sentiment_score"].mean() if len(recent) else 0.0
    c.metric("Avg sentiment (1h)", f"{b2:+.2f}")
    d.metric("Headlines (24h)", len(day))

    fig = go.Figure(go.Scatter(x=c1m["time"], y=c1m["price"], mode="lines"))
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0),
                      title=f"{coin} – last 2 hours (1-minute candles, UTC)")
    show_chart(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("1-hour forecast")
        try:
            row = features.live_row(news_df, c5m)
            result = model.predict(coin, row)
            icon = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪"}[result["signal"]]
            st.metric("Predicted change (next 1h)", f"{result['predicted_change_pct']:+.3f}%")
            st.markdown(f"### {icon} {result['signal']}")
        except FileNotFoundError:
            st.info("No trained model yet. Run the collector, then `python -m crypto_sentiment.train`.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        m = model.load_metrics(coin)
        if m:
            st.caption(
                f"Held-out test: MAE {m['mae']:.3f}% vs naive baseline {m['baseline_mae']:.3f}% · "
                f"R² {m['r2']:.2f} · direction accuracy {m['directional_accuracy'] * 100:.0f}% "
                f"({m['rows_test']} test rows)")
            if m["r2"] <= 0 or m["mae"] >= m["baseline_mae"]:
                st.warning("The model does not beat the naive baseline on held-out data. "
                           "Treat the signal as informational only.")
    with right:
        st.subheader("Sentiment mix (24h)")
        if len(day):
            counts = day["sentiment_label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            show_chart(px.bar(counts, x="label", y="count", color="label",
                                   color_discrete_map={"Positive": "#2e9e5b", "Negative": "#d64545",
                                                       "Neutral": "#8a8f98"}))
        else:
            st.info("No headlines in the last 24h yet.")

    st.subheader("Latest headlines")
    show_table(news_df.sort_values("published_at", ascending=False).head(25)[
        ["published_at", "title", "sentiment_label", "sentiment_score", "source"]])
    st.caption(f"Updated {now:%Y-%m-%d %H:%M:%S} UTC")


tab_live, tab_compare, tab_bt, tab_score = st.tabs(
    ["Live", "Model comparison", "Backtest", "Live scoreboard"])

with tab_live:
    live_panel()

with tab_compare:
    st.subheader(f"{coin}: walk-forward model comparison")
    comp = evaluate.load(coin)
    if not comp:
        st.info("Run `python -m crypto_sentiment.train --full` after collecting data.")
    else:
        if comp.get("synthetic_data"):
            st.warning("Computed on SYNTHETIC data.")
        rows = [{"model": k, "MAE % (mean)": v["mean_mae"], "MAE std": v["std_mae"],
                 "Direction acc.": v["directional_accuracy"], "p-value (direction)": v["p_value_direction"],
                 "Folds beating baseline": f'{v["folds_beating_baseline"]}/{v["n_folds"]}'}
                for k, v in comp["summary"].items()]
        table = pd.DataFrame(rows)
        show_table(table)
        show_chart(px.bar(table, x="model", y="MAE % (mean)", title="Mean absolute error (lower is better)"))
        st.caption("Trained on the past, tested on the next block, repeated forward in time. "
                   "p-values account for overlapping samples. A model only matters if it beats the baseline.")

with tab_bt:
    st.subheader(f"{coin}: backtest on out-of-sample predictions")
    bt = backtest.load(coin)
    if not bt:
        st.info("Run `python -m crypto_sentiment.train --full` after collecting data.")
    else:
        if bt.get("synthetic_data"):
            st.warning("Computed on SYNTHETIC data.")
        a, b, c, d, e = st.columns(5)
        a.metric("Trades", bt["n_trades"])
        b.metric("Win rate", "n/a" if bt["win_rate"] is None else f'{bt["win_rate"] * 100:.0f}%')
        c.metric("Strategy return", f'{bt["total_return_pct"]:+.2f}%')
        d.metric("Buy & hold", f'{bt["buy_hold_return_pct"]:+.2f}%')
        e.metric("Max drawdown", f'{bt["max_drawdown_pct"]:.2f}%')
        if bt["equity"]["time"]:
            show_chart(px.line(x=bt["equity"]["time"], y=bt["equity"]["value"],
                                    labels={"x": "time", "y": "equity (start = 1.0)"}, title="Equity curve"))
        st.caption(f'Long-only, {bt["fee_pct_per_side"]}% fee per side, one trade at a time, '
                   f'threshold {bt["threshold_pct"]}%. Sharpe (approx.): '
                   f'{"n/a" if bt["sharpe"] is None else round(bt["sharpe"], 2)}.')

with tab_score:
    st.subheader(f"{coin}: live scoreboard (forecasts scored one hour later)")
    sb = tracker.scoreboard(coin)
    if sb["n_logged"] == 0:
        st.info("Needs a trained model and the collector running: "
                "`python -m crypto_sentiment.pipeline --loop`.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Forecasts logged", sb["n_logged"])
        b.metric("Scored", sb["n_settled"])
        c.metric("Direction accuracy",
                 "n/a" if sb["directional_accuracy"] is None else f'{sb["directional_accuracy"] * 100:.0f}%')
        d.metric("p-value", "n/a" if sb["p_value"] is None else f'{sb["p_value"]:.3f}')
        show_table(tracker.load().query("coin == @coin").sort_values("time", ascending=False).head(20))
        st.caption("These are genuine out-of-sample results: every forecast was made before the outcome existed.")

with st.expander("Model details"):
    m = model.load_metrics(coin)
    if m:
        st.json(m)
    else:
        st.write("Train a model to see metrics.")

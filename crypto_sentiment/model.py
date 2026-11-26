"""Training, evaluation and prediction."""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from . import config, features, news as news_mod, prices as prices_mod
from .storage import utc_now

log = logging.getLogger(__name__)
GAP = config.HORIZON_MINUTES // config.CANDLE_MINUTES  # rows removed between train/test


def model_path(coin):
    return config.MODEL_DIR / f"model_{coin}.pkl"


def metrics_path(coin):
    return config.MODEL_DIR / f"metrics_{coin}.json"


def train(coin: str, news=None, prices=None) -> dict:
    """Train a Random Forest for one coin and save model + metrics."""
    news = news_mod.load() if news is None else news
    prices = prices_mod.load(coin) if prices is None else prices
    df = features.build_dataset(news, prices)
    if len(df) < config.MIN_TRAIN_ROWS:
        raise ValueError(f"{coin}: only {len(df)} usable rows (need {config.MIN_TRAIN_ROWS}). "
                         "Let the collector run longer.")

    df = df.sort_values("time").reset_index(drop=True)
    split = int(len(df) * 0.8)
    # Chronological split with a gap equal to the horizon so no training label
    # overlaps the test period (prevents leakage from overlapping windows).
    train_df, test_df = df.iloc[: max(split - GAP, 1)], df.iloc[split:]
    X_tr, y_tr = train_df[config.FEATURES], train_df["target_pct"]
    X_te, y_te = test_df[config.FEATURES], test_df["target_pct"]

    model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    baseline = np.full(len(y_te), y_tr.mean())  # naive "always predict the average"
    nonzero = y_te != 0
    metrics = {
        "coin": coin,
        "rows_total": int(len(df)), "rows_train": int(len(train_df)), "rows_test": int(len(test_df)),
        "mae": float(mean_absolute_error(y_te, pred)),
        "baseline_mae": float(mean_absolute_error(y_te, baseline)),
        "r2": float(r2_score(y_te, pred)),
        "directional_accuracy": float((np.sign(pred[nonzero]) == np.sign(y_te[nonzero])).mean()),
        "feature_importance": dict(zip(config.FEATURES, map(float, model.feature_importances_))),
        "trained_at": utc_now().isoformat(),
        "synthetic_data": config.DEMO_MARKER.exists(),
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path(coin))
    metrics_path(coin).write_text(json.dumps(metrics, indent=2))
    log.info("%s | MAE %.4f (baseline %.4f) | R2 %.3f | dir.acc %.1f%%", coin, metrics["mae"],
             metrics["baseline_mae"], metrics["r2"], metrics["directional_accuracy"] * 100)
    return metrics


def load_metrics(coin: str):
    p = metrics_path(coin)
    return json.loads(p.read_text()) if p.exists() else None


def predict(coin: str, feature_row: pd.DataFrame) -> dict:
    """Predict the % change over the next hour and turn it into a signal."""
    p = model_path(coin)
    if not p.exists():
        raise FileNotFoundError(f"No trained model for {coin}. Run: python -m crypto_sentiment.train")
    value = float(joblib.load(p).predict(feature_row[config.FEATURES])[0])
    if value > config.SIGNAL_THRESHOLD_PCT:
        signal = "Bullish"
    elif value < -config.SIGNAL_THRESHOLD_PCT:
        signal = "Bearish"
    else:
        signal = "Neutral"
    return {"predicted_change_pct": value, "signal": signal}

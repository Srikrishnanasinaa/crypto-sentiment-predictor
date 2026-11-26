"""Walk-forward model comparison: baseline vs Ridge vs Random Forest vs Gradient Boosting.

Walk-forward = train on the past, test on the next block, move forward in time. A gap equal to the
prediction horizon is left between train and test so overlapping labels cannot leak.
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features, news as news_mod, prices as prices_mod

log = logging.getLogger(__name__)
GAP = config.HORIZON_MINUTES // config.CANDLE_MINUTES
OVERLAP = GAP  # consecutive rows share ~all of their horizon -> ~1 independent sample per GAP rows

MODELS = {
    "baseline_mean": lambda: DummyRegressor(strategy="mean"),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                                   random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                               max_iter=200, random_state=42),
}


def directional_accuracy(y, pred) -> float:
    mask = np.asarray(y) != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(np.asarray(pred)[mask]) == np.sign(np.asarray(y)[mask])).mean())


def directional_pvalue(acc: float, n_rows: int) -> float:
    """P(accuracy this high by chance). Rows overlap in time, so n is shrunk to the number of
    roughly independent samples - this keeps the test honest."""
    from scipy.stats import binomtest

    n_eff = max(int(n_rows / OVERLAP), 1)
    k = int(round(acc * n_eff))
    return float(binomtest(k, n_eff, 0.5, alternative="greater").pvalue)


def walk_forward(df: pd.DataFrame, n_splits: int = 5, min_train_frac: float = 0.4):
    """Return (oos_predictions, fold_results)."""
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    start = int(n * min_train_frac)
    fold_size = (n - start) // n_splits
    if fold_size < 20:
        raise ValueError("Not enough rows for walk-forward evaluation")

    oos_parts, folds = [], []
    for k in range(n_splits):
        test_lo = start + k * fold_size
        test_hi = n if k == n_splits - 1 else test_lo + fold_size
        train_hi = test_lo - GAP
        if train_hi < 30:
            continue
        train, test = df.iloc[:train_hi], df.iloc[test_lo:test_hi]
        part = test[["time", "price", "target_pct"]].copy()
        part["fold"] = k
        for name, factory in MODELS.items():
            mdl = factory().fit(train[config.FEATURES], train["target_pct"])
            pred = mdl.predict(test[config.FEATURES])
            part[f"pred_{name}"] = pred
            folds.append({
                "fold": k, "model": name, "n_test": int(len(test)),
                "mae": float(mean_absolute_error(test["target_pct"], pred)),
                "directional_accuracy": directional_accuracy(test["target_pct"], pred),
                "train_end": str(train["time"].iloc[-1]), "test_start": str(test["time"].iloc[0]),
            })
        oos_parts.append(part)
    return pd.concat(oos_parts, ignore_index=True), folds


def summarize(oos: pd.DataFrame, folds: list) -> dict:
    fd = pd.DataFrame(folds)
    base = fd[fd["model"] == "baseline_mean"].set_index("fold")["mae"]
    out = {}
    for name in MODELS:
        sub = fd[fd["model"] == name]
        acc = directional_accuracy(oos["target_pct"], oos[f"pred_{name}"])
        out[name] = {
            "mean_mae": float(sub["mae"].mean()),
            "std_mae": float(sub["mae"].std(ddof=0)),
            "folds_beating_baseline": int((sub.set_index("fold")["mae"] < base).sum()),
            "n_folds": int(len(sub)),
            "directional_accuracy": acc,
            "p_value_direction": directional_pvalue(acc, len(oos)) if not np.isnan(acc) else None,
        }
    return out


def comparison_path(coin):
    return config.MODEL_DIR / f"comparison_{coin}.json"


def oos_path(coin):
    return config.MODEL_DIR / f"oos_{coin}.csv"


def run(coin: str, news=None, prices=None, n_splits: int = 5) -> dict:
    news = news_mod.load() if news is None else news
    prices = prices_mod.load(coin) if prices is None else prices
    df = features.build_dataset(news, prices)
    if len(df) < config.MIN_TRAIN_ROWS:
        raise ValueError(f"{coin}: only {len(df)} usable rows (need {config.MIN_TRAIN_ROWS}).")
    oos, folds = walk_forward(df, n_splits)
    result = {"coin": coin, "rows": int(len(df)), "n_splits": n_splits,
              "synthetic_data": config.DEMO_MARKER.exists(),
              "summary": summarize(oos, folds), "folds": folds}
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path(coin).write_text(json.dumps(result, indent=2))
    oos.to_csv(oos_path(coin), index=False)
    for name, s in result["summary"].items():
        log.info("%s | %-18s MAE %.4f | dir.acc %.1f%% | beats baseline in %d/%d folds",
                 coin, name, s["mean_mae"], s["directional_accuracy"] * 100,
                 s["folds_beating_baseline"], s["n_folds"])
    return result


def load(coin: str):
    p = comparison_path(coin)
    return json.loads(p.read_text()) if p.exists() else None

import numpy as np
import pandas as pd

from crypto_sentiment import config, evaluate, features
from test_model import synthetic


def dataset():
    news, prices = synthetic(8000)
    return features.build_dataset(news, prices)


def test_walk_forward_is_chronological_with_gap():
    oos, folds = evaluate.walk_forward(dataset(), n_splits=3)
    fd = pd.DataFrame(folds)
    for _, r in fd.iterrows():
        gap = pd.Timestamp(r["test_start"]) - pd.Timestamp(r["train_end"])
        assert gap >= pd.Timedelta(minutes=config.HORIZON_MINUTES)
    starts = fd.groupby("fold")["test_start"].first().map(pd.Timestamp)
    assert starts.is_monotonic_increasing
    assert oos["time"].is_monotonic_increasing


def test_all_models_present_and_summary_sane():
    oos, folds = evaluate.walk_forward(dataset(), n_splits=3)
    summary = evaluate.summarize(oos, folds)
    assert set(summary) == set(evaluate.MODELS)
    for s in summary.values():
        assert s["mean_mae"] > 0
        assert 0 <= s["directional_accuracy"] <= 1
        assert 0 <= s["p_value_direction"] <= 1


def test_random_noise_is_not_significant():
    oos, folds = evaluate.walk_forward(dataset(), n_splits=3)
    summary = evaluate.summarize(oos, folds)
    # features are pure noise here, so no model should look "significantly" predictive
    assert summary["random_forest"]["p_value_direction"] > 0.01


def test_pvalue_shrinks_sample_size():
    from scipy.stats import binomtest

    # 52% on 10,000 overlapping rows looks "highly significant" if rows are wrongly treated as
    # independent, but only ~833 rows are independent -> not significant. The code must see that.
    naive = binomtest(5200, 10_000, 0.5, alternative="greater").pvalue
    assert naive < 0.001
    assert evaluate.directional_pvalue(0.52, 10_000) > 0.05
    assert evaluate.directional_pvalue(0.80, 10_000) < 0.001


def test_run_saves_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    news, prices = synthetic(8000)
    res = evaluate.run("BTC", news=news, prices=prices, n_splits=3)
    assert (tmp_path / "comparison_BTC.json").exists()
    assert (tmp_path / "oos_BTC.csv").exists()
    assert "random_forest" in res["summary"]

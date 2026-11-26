import pytest

pytest.importorskip("vaderSentiment")

from crypto_sentiment import sentiment


def test_positive_negative_and_labels():
    assert sentiment.score("Bitcoin surges to a record high, bullish rally") > 0.05
    assert sentiment.score("Exchange hacked, crypto market crashes") < -0.05
    assert sentiment.label(0.5) == "Positive"
    assert sentiment.label(-0.5) == "Negative"
    assert sentiment.label(0.0) == "Neutral"

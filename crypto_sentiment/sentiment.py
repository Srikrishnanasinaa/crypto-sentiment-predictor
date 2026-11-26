"""VADER sentiment, extended with crypto-specific vocabulary."""
import pandas as pd

CRYPTO_LEXICON = {
    "bullish": 2.2, "bearish": -2.2, "rally": 1.8, "rallies": 1.8, "surge": 1.9, "surges": 1.9,
    "soar": 2.1, "soars": 2.1, "moon": 1.5, "adoption": 1.5, "approval": 1.4, "approved": 1.4,
    "breakout": 1.6, "outperform": 1.5, "inflow": 1.3, "inflows": 1.3,
    "crash": -2.6, "crashes": -2.6, "plunge": -2.3, "plunges": -2.3, "dump": -2.0,
    "hack": -2.6, "hacked": -2.6, "exploit": -2.2, "scam": -2.5, "fraud": -2.6, "ban": -1.9,
    "banned": -1.9, "lawsuit": -1.6, "sues": -1.5, "liquidation": -1.8, "liquidations": -1.8,
    "outflow": -1.3, "outflows": -1.3, "bankruptcy": -2.6, "rug": -2.4, "selloff": -2.0,
}

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(CRYPTO_LEXICON)
    return _analyzer


def score(text: str) -> float:
    """Compound sentiment in [-1, 1]."""
    return float(_get_analyzer().polarity_scores(str(text))["compound"])


def label(compound: float) -> str:
    if compound > 0.05:
        return "Positive"
    if compound < -0.05:
        return "Negative"
    return "Neutral"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment_score / sentiment_label columns to a frame with a `title` column."""
    out = df.copy()
    out["sentiment_score"] = out["title"].map(score)
    out["sentiment_label"] = out["sentiment_score"].map(label)
    return out

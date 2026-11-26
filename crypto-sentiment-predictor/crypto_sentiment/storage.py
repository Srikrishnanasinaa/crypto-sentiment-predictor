"""Small CSV helpers."""
from pathlib import Path

import pandas as pd


def utc_now() -> pd.Timestamp:
    """Current UTC time as a timezone-naive Timestamp."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def append_unique(path: Path, new: pd.DataFrame, keys: list, sort_by: str) -> pd.DataFrame:
    """Append rows to a CSV, drop duplicates on `keys`, keep it sorted. Returns full frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=[sort_by])
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.drop_duplicates(subset=keys, keep="first").sort_values(sort_by).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df

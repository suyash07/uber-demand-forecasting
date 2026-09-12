"""
Inference: load the saved pipeline and score new rows.

The pipeline carries all preprocessing, so callers pass raw feature columns —
no manual encoding/scaling. This is the deployment entry point.
"""
from __future__ import annotations
import joblib
import pandas as pd

from . import config as C

_saved = None


def _load():
    global _saved
    if _saved is None:
        _saved = joblib.load(C.PIPELINE_FILE)
    return _saved


def predict_proba(rows: pd.DataFrame) -> pd.Series:
    """Return surge probability for each row (expects C.FEATURES columns)."""
    saved = _load()
    proba = saved["pipeline"].predict_proba(rows[saved["features"]])[:, 1]
    return pd.Series(proba, index=rows.index, name="surge_proba")


def predict_label(rows: pd.DataFrame, threshold: float | None = None) -> pd.Series:
    saved = _load()
    thr = saved["threshold"] if threshold is None else threshold
    return (predict_proba(rows) >= thr).astype(int).rename("is_surge_pred")


if __name__ == "__main__":
    df = pd.read_parquet(C.FEATURES_FILE).head(5)
    print(predict_proba(df))

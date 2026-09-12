"""
Final training: fit the tuned pipeline, cross-validate, save one artifact.

This is the REPEATABLE production trainer — it consumes the tuned params from
config and produces a deployable pipeline. It does not search hyperparameters
(that is tune.py's job, run occasionally).
"""
from __future__ import annotations
import logging
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from . import config as C
from .pipeline import build_pipeline

log = logging.getLogger(__name__)


def time_split(df: pd.DataFrame):
    df = df.copy()
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
    train = df[df[C.DATE_COL] < C.SPLIT_DATE]
    test  = df[df[C.DATE_COL] >= C.SPLIT_DATE]
    return (train[C.FEATURES], test[C.FEATURES],
            train[C.TARGET], test[C.TARGET])


def cross_validate(df: pd.DataFrame, pos_weight: float, n_splits: int = 5):
    """TimeSeriesSplit CV (temporal data) scored on PR-AUC — with variance."""
    df_sorted = df.sort_values(C.DATE_COL)
    X, y = df_sorted[C.FEATURES], df_sorted[C.TARGET]
    scores = cross_val_score(
        build_pipeline(pos_weight), X, y,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="average_precision", n_jobs=-1,
    )
    log.info("CV PR-AUC: %.4f +/- %.4f  folds=%s",
             scores.mean(), scores.std(), [round(s, 4) for s in scores])
    return scores


def train(do_cv: bool = True) -> None:
    df = pd.read_parquet(C.FEATURES_FILE)
    X_train, X_test, y_train, y_test = time_split(df)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    log.info("train=%s test=%s pos_weight=%.1f",
             f"{len(X_train):,}", f"{len(X_test):,}", pos_weight)

    if do_cv:
        cross_validate(df, pos_weight)

    pipe = build_pipeline(pos_weight)
    pipe.fit(X_train, y_train)

    joblib.dump(
        {"pipeline": pipe, "features": C.FEATURES, "threshold": C.DEFAULT_THRESHOLD},
        C.PIPELINE_FILE,
    )
    log.info("saved %s", C.PIPELINE_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()

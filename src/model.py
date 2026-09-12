"""
Baseline model: logistic regression, for comparison against XGBoost.

A simple, honest floor. If the tree model can't beat this by a meaningful
margin, the extra complexity isn't earning its place. Logistic needs scaling +
imputation (unlike XGBoost), so it uses its own preprocessing.
"""
from __future__ import annotations
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .train import time_split

log = logging.getLogger(__name__)


def build_baseline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scale", StandardScaler())])
    pre = ColumnTransformer([
        ("zone", OneHotEncoder(handle_unknown="ignore"), C.HIGH_CARD),
        ("boro", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL),
        ("num", numeric, C.NUMERIC),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("model", clf)])


def compare() -> None:
    """Train the logistic baseline and print its test metrics for comparison."""
    df = pd.read_parquet(C.FEATURES_FILE)
    X_train, X_test, y_train, y_test = time_split(df)
    pipe = build_baseline().fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    base = y_test.mean()
    print("=== Logistic baseline (test) ===")
    print(f"ROC-AUC : {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, proba):.4f} "
          f"(baseline {base:.4f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compare()

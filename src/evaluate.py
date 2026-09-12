"""
Evaluation: test metrics, threshold-vs-tradeoff table, and SHAP drivers.

Loads the saved pipeline and scores the held-out test period. PR-AUC is the
primary metric (imbalanced target); accuracy is deliberately ignored.
"""
from __future__ import annotations
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             classification_report)

from . import config as C
from .train import time_split

log = logging.getLogger(__name__)


def evaluate() -> None:
    saved = joblib.load(C.PIPELINE_FILE)
    pipe = saved["pipeline"]

    df = pd.read_parquet(C.FEATURES_FILE)
    X_train, X_test, y_train, y_test = time_split(df)
    proba = pipe.predict_proba(X_test)[:, 1]

    base = y_test.mean()
    print("=== Test performance ===")
    print(f"ROC-AUC : {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, proba):.4f}  "
          f"(baseline {base:.4f}, lift {average_precision_score(y_test, proba)/base:.1f}x)")
    pred = (proba >= saved["threshold"]).astype(int)
    print(classification_report(y_test, pred, digits=3))

    print("=== Threshold tradeoff ===")
    for t in (0.3, 0.5, 0.7, 0.9):
        p = proba >= t
        tp = int((p & (y_test == 1)).sum()); fp = int((p & (y_test == 0)).sum())
        fn = int((~p & (y_test == 1)).sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        print(f"thr={t}: precision={prec:.3f} recall={rec:.3f} flagged={p.mean()*100:.1f}%")


def shap_summary(sample: int = 20_000) -> None:
    """Optional: SHAP feature importance on a sample (needs `shap`)."""
    import shap
    saved = joblib.load(C.PIPELINE_FILE)
    pipe = saved["pipeline"]
    df = pd.read_parquet(C.FEATURES_FILE)
    _, X_test, _, _ = time_split(df)
    Xs = X_test.sample(min(sample, len(X_test)), random_state=42)

    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    Xs_enc = pre.transform(Xs)
    names = pre.get_feature_names_out()
    vals = shap.TreeExplainer(model).shap_values(Xs_enc)
    imp = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(vals).mean(0)})
             .sort_values("mean_abs_shap", ascending=False))
    print(imp.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate()

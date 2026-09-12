"""
The model pipeline: preprocessing + estimator bundled into one object.

Why a Pipeline:
  - transforms are fit on TRAIN only and applied identically everywhere
    (no manual re-scaling/re-encoding of test — the classic leakage bug);
  - cross-validation refits preprocessing inside each fold automatically;
  - deployment saves ONE object, not model + encoders separately.
"""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from . import config as C


def build_preprocessor() -> ColumnTransformer:
    """Route each column group to its encoder; numeric passes through."""
    return ColumnTransformer(
        transformers=[
            # PULocationID is high-cardinality but not safe to target-encode with the
            # current scikit-learn / category_encoders combination in this environment.
            # Use one-hot encoding instead to avoid the sklearn compatibility crash.
            ("zone", OneHotEncoder(handle_unknown="ignore"), C.HIGH_CARD),
            # low-cardinality borough -> one-hot; unseen categories -> all zeros
            ("boro", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL),
            # numeric -> passthrough (XGBoost needs no scaling, handles NaN)
            ("num", "passthrough", C.NUMERIC),
        ]
    )


def build_pipeline(pos_weight: float, params: dict | None = None) -> Pipeline:
    """Preprocessing + tuned XGBoost as a single fit/predict object."""
    xgb = XGBClassifier(scale_pos_weight=pos_weight, **(params or C.XGB_PARAMS))
    return Pipeline([("pre", build_preprocessor()), ("model", xgb)])

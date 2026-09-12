"""
Hyperparameter tuning with Optuna, using OUT-OF-TIME validation.

Run occasionally as an experiment — not part of routine retraining. The winning
params are copied into config.XGB_PARAMS, which train.py consumes.

Key correctness point: the validation slice is the LATEST portion of the sample
by time (not a random slice), so the score reflects future generalization and
matches the test set — avoiding in-distribution optimism.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import config as C
    from src.pipeline import build_preprocessor
else:
    from . import config as C
    from .pipeline import build_preprocessor

log = logging.getLogger(__name__)


def _time_split_sample(df: pd.DataFrame, n=2_000_000, val_frac=0.2):
    samp = df.sample(min(n, len(df)), random_state=42)
    samp = samp.sort_values(C.DATE_COL)
    cut = int(len(samp) * (1 - val_frac))
    tr, va = samp.iloc[:cut], samp.iloc[cut:]
    return tr[C.FEATURES], va[C.FEATURES], tr[C.TARGET], va[C.TARGET]


def run_study(n_trials: int = 30) -> optuna.Study:
    df = pd.read_parquet(C.FEATURES_FILE)
    X_tr, X_va, y_tr, y_va = _time_split_sample(df)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    pre = build_preprocessor()

    # preprocess once (encoders fit on train fold only, applied to val)
    X_tr_e = pre.fit_transform(X_tr, y_tr)
    X_va_e = pre.transform(X_va)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),          # tree pruning
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            scale_pos_weight=pos_weight,
            eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=42,
        )
        model = XGBClassifier(**params, early_stopping_rounds=30)  # early stopping
        model.fit(X_tr_e, y_tr, eval_set=[(X_va_e, y_va)], verbose=False)
        return float(average_precision_score(y_va, model.predict_proba(X_va_e)[:, 1]))

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner())  # trial pruning
    study.optimize(objective, n_trials=n_trials)
    log.info("best PR-AUC (time val): %.4f", study.best_value)
    log.info("best params: %s", study.best_params)
    return study


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = run_study()
    print("Best PR-AUC:", round(s.best_value, 4))
    print("Copy these into config.XGB_PARAMS:")
    print(s.best_params)

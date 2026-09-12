"""
Orchestrator / CLI. Run stages individually or end to end.

Examples:
    python -m src.main data       # raw -> clean
    python -m src.main features   # clean -> labeled + features
    python -m src.main tune       # Optuna search (occasional)
    python -m src.main train      # final fit + CV + save
    python -m src.main evaluate   # test metrics + threshold table
    python -m src.main baseline   # logistic comparison
    python -m src.main all        # features -> train -> evaluate
"""
from __future__ import annotations
import argparse

from . import config as C
from .utils import setup_logging


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "features", "tune", "train",
                                      "evaluate", "baseline", "shap", "all"])
    args = ap.parse_args()

    if args.stage == "data":
        from .data import build_clean; build_clean()
    elif args.stage == "features":
        import pandas as pd
        from .target import build_surge_labels
        from .features import build_feature_frame
        build_feature_frame(build_surge_labels(pd.read_parquet(C.CLEAN)))
    elif args.stage == "tune":
        from .tune import run_study; run_study()
    elif args.stage == "train":
        from .train import train; train()
    elif args.stage == "evaluate":
        from .evaluate import evaluate; evaluate()
    elif args.stage == "baseline":
        from .model import compare; compare()
    elif args.stage == "shap":
        from .evaluate import shap_summary; shap_summary()
    elif args.stage == "all":
        from .train import train
        from .evaluate import evaluate
        train(); evaluate()


if __name__ == "__main__":
    main()

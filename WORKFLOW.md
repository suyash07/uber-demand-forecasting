# Surge Classifier — Production Workflow & Interview Guide

A modular, production-style ML pipeline for the NYC surge-pricing classifier.
This document explains the architecture, how to run it, the design decisions,
and — for the technical round — a **debugging** and **code-review** guide.

---

## 1. Architecture — one module, one responsibility

```
src/
  config.py     Single source of truth: paths, feature groups, params, thresholds
  data.py       Raw TLC + zones + weather  ->  cleaned trips (df_clean)
  target.py     Engineer the is_surge label (expected-fare model)
  features.py   Time features + demand lag/rolling features
  pipeline.py   Preprocessing (encoders) + XGBoost, bundled as one Pipeline
  model.py      Logistic-regression baseline (for comparison)
  tune.py       Optuna hyperparameter search (out-of-time validation)
  train.py      Final fit + TimeSeriesSplit CV + save one artifact
  evaluate.py   Test metrics, threshold tradeoff, SHAP drivers
  predict.py    Load the saved pipeline and score new rows (deployment entry)
  main.py       CLI orchestrator
  utils.py      Logging setup
```

Why this split: each file is small and independently testable, config is
centralized, and the training path (`train.py`) is separate from the one-off
tuning experiment (`tune.py`) — you don't re-search params every retrain.

---

## 2. Data flow

```
raw parquet (Jan+Feb)                 data.py
   + zones + weather   ── merge ──>   df_clean.parquet
                                          │  target.py (expected-fare -> is_surge)
                                          ▼
                                      labeled trips
                                          │  features.py (time + demand lags)
                                          ▼
                        surcharge_df_features.parquet   <-- the modeling table
                                          │  train.py (time split)
                          ┌───────────────┴───────────────┐
                       train (< 2023-02-15)         test (>= 2023-02-15)
                          │  pipeline.py (fit)             │
                          ▼                                ▼
                 surge_pipeline.joblib  ──> evaluate.py / predict.py
```

Heavy steps are cached to `data/` as parquet, so routine work reads the
prepared `surcharge_df_features.parquet` and never re-runs the 36M-row build.

---

## 3. How to run

```bash
# rebuild the environment first (the old venv is broken)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# stages (run from the repo root)
python -m src.main train      # fit tuned pipeline + CV + save artifact
python -m src.main evaluate   # test metrics + threshold table
python -m src.main baseline   # logistic comparison
python -m src.main tune       # Optuna search (occasional, slow)
python -m src.main shap       # feature-importance drivers

# regenerate data from scratch (heavy):
python -m src.main data       # raw -> df_clean
python -m src.main features   # df_clean -> labeled + features
```

---

## 4. Key design decisions (be ready to defend these)

- **Target is engineered, not given.** `is_surge = fare > 1.5x expected_fare`,
  where expected fare is a robust Huber fit on distance+time. Robust (not OLS)
  so surge trips don't inflate the baseline they're measured against.
- **Time-based split, never random.** Random would leak the future through lag
  features. Test is the last two weeks.
- **Out-of-time validation for tuning.** A random validation slice reported an
  optimistic PR-AUC (0.37) that collapsed to 0.19 on test; a time-ordered
  validation slice fixed the illusion.
- **Pipeline bundles preprocessing + model.** Transforms fit on train only,
  applied identically everywhere; CV refits them per fold; deploy one object.
- **TargetEncoder for zone, one-hot for borough.** High vs low cardinality.
  Encoding lives inside the pipeline so it's leakage-safe automatically.
- **PR-AUC over accuracy.** Target is ~6% positive; accuracy is meaningless.
  Report PR-AUC as a multiple of the base rate.
- **Threshold is a business dial.** 0.5 is arbitrary; pick from the cost of a
  false alarm vs a miss.
- **Honest ceiling.** Surge is supply-driven; the data has no driver-supply
  signal, which caps performance — not the model.

---

## 5. Debugging guide (what breaks, and how to find it)

| Symptom | Likely cause | How to diagnose / fix |
|---|---|---|
| Logistic AUC far below XGBoost's, or nonsense | test scaled/encoded differently from train (positional mismatch) | Use the Pipeline (transforms locked); never transform test with a re-fit or reordered encoder |
| Validation score >> test score | in-distribution validation (random slice on temporal data) | Sort by time, validate on the latest slice; use `TimeSeriesSplit` |
| Model predicts all-negative / F1=0 | class imbalance untreated | check `y.mean()`; set `scale_pos_weight`; evaluate PR-AUC not accuracy |
| Lag features look wrong (shift != N hours) | sparse hourly grid | ensure the gapless `MultiIndex.from_product` grid + `fillna(0)` before `shift()` |
| Perfect / suspiciously high metrics | leakage | check for fare-derived columns in FEATURES (see `config.LEAKAGE_COLS`); check lag `shift()` |
| `NaN`/error in logistic, but XGBoost fine | logistic can't take NaN | impute in the baseline pipeline; XGBoost handles NaN natively |
| Kernel dies / OOM | 36M rows in memory | read cached parquet, downcast to float32/int32, sample for iteration |
| Unseen category at scoring | new zone/borough in production | TargetEncoder + `OneHotEncoder(handle_unknown='ignore')` handle it |

Quick checks to run when a metric looks off:
```python
y_train.mean(), y_test.mean()          # imbalance + regime shift
X_train.isna().mean().sort_values()    # NaN rates by column
set(X_train.columns) == set(X_test.columns)   # column alignment
```

---

## 6. Code-review checklist (per module)

- **config.py** — no logic, no secrets; feature groups match the data schema.
- **data.py** — cleaning thresholds justified (quantiles/knee), not magic numbers; window clamp present.
- **target.py** — robust regression; denominator guarded (`expected_fare > 0`); label rate sane (~5-20%).
- **features.py** — every aggregate `shift()`-ed; no fare-derived feature; gapless grid before lags.
- **pipeline.py** — encoders fit inside the pipeline (train-only); numeric passthrough for trees.
- **tune.py** — validation is out-of-time; early stopping + pruner present; params flow to config.
- **train.py** — time split (not random); CV uses `TimeSeriesSplit`; one artifact saved with feature list + threshold.
- **evaluate.py** — PR-AUC vs base rate; threshold tradeoff shown; SHAP confirms no leakage feature dominates.
- **predict.py** — consumes the saved feature list; no re-implementation of preprocessing.

---

## 7. What I'd do next (production maturity)

- Add unit tests (`tests/`) for leakage guards and feature shapes.
- Feature store for low-latency velocity/demand aggregates at serving time.
- Monitoring: PR-AUC decay, score drift, zone-distribution drift -> retrain trigger.
- CI to run `py_compile` + tests + a tiny-sample end-to-end on every PR.

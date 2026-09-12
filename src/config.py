"""
Central configuration: paths, constants, feature groups, and model params.

Keeping all of this in one place (rather than scattered across scripts) is a
production practice — one source of truth, easy to review, easy to change per
environment. Nothing here executes logic; it is pure configuration.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

RAW_JAN   = DATA / "url_jan.parquet"
RAW_FEB   = DATA / "url_feb.parquet"
ZONES     = DATA / "taxi_zone_lookup.csv"
WEATHER   = DATA / "weather.parquet"

CLEAN     = DATA / "df_clean.parquet"                 # cleaned trips
FEATURES_FILE = DATA / "surcharge_df_features.parquet"  # labeled + feature-engineered

PIPELINE_FILE = ARTIFACTS / "surge_pipeline.joblib"

# ----------------------------------------------------------------------
# Modeling constants
# ----------------------------------------------------------------------
TARGET   = "is_surge"
DATE_COL = "request_datetime"
SPLIT_DATE = "2023-02-15"          # time-based train/test cutoff
WINDOW = ("2023-01-01", "2023-03-01")  # valid data window (matches weather coverage)

# Cleaning thresholds (justified from EDA knee/quantile plots)
MIN_MILES = 0.5
MIN_FARE  = 5.0
FARE_CAP_Q = 0.995
SURGE_MULT_THRESHOLD = 1.5        # fare > 1.5x expected -> surge

# ----------------------------------------------------------------------
# Feature groups (single source of truth for the pipeline)
# ----------------------------------------------------------------------
HIGH_CARD   = ["PULocationID"]    # target-encoded
CATEGORICAL = ["Borough"]         # one-hot encoded
NUMERIC = [
    "temperature", "precipitation_inches", "is_raining",
    "hour", "weekend", "is_holiday",
    "hour_sine", "hour_cosine", "dow_sine", "dow_cosine",
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
    "demand_roll_3h", "demand_roll_24h",
]
FEATURES = HIGH_CARD + CATEGORICAL + NUMERIC

# Columns that must NEVER be features (leakage) — kept here as an explicit guard
LEAKAGE_COLS = [
    "base_passenger_fare", "expected_fare", "surge_mult", "fare_per_mile",
    "tips", "driver_pay", "tolls", "bcf", "sales_tax",
    "congestion_surcharge", "airport_fee",
]

# Tuned XGBoost params (from the Optuna study in tune.py)
XGB_PARAMS = dict(
    n_estimators=489,
    learning_rate=0.0166,
    max_depth=9,
    min_child_weight=2,
    gamma=3.93,
    subsample=0.927,
    colsample_bytree=0.873,
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

DEFAULT_THRESHOLD = 0.66          # decision cutoff (tunable per use case)

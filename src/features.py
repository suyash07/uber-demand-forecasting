"""
Feature engineering: time features + demand lag/rolling features.

All features are things known BEFORE the trip. Leakage guards:
  - demand aggregates are built on a gapless hourly grid and shifted, so a row
    never sees its own or future demand.
  - no fare-derived column is ever produced here.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

US_HOLIDAYS = pd.to_datetime(["2023-01-01", "2023-01-16", "2023-02-20"])


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[C.DATE_COL])
    df["hour"] = ts.dt.hour
    dow = ts.dt.dayofweek
    df["weekend"] = (dow >= 5).astype(int)
    df["date"] = ts.dt.normalize()
    df["is_holiday"] = df["date"].isin(US_HOLIDAYS).astype(int)
    # cyclical encodings (help the linear baseline; harmless for trees)
    df["hour_sine"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cosine"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sine"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cosine"] = np.cos(2 * np.pi * dow / 7)
    return df


def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zone-hour demand lags/rolling means, joined back onto each trip."""
    df = df.copy()
    df["hour_ts"] = pd.to_datetime(df[C.DATE_COL]).dt.floor("h")

    # 1) count trips per (zone, hour)
    demand = (df.groupby(["PULocationID", "hour_ts"]).size()
                .rename("demand").reset_index())
    demand["demand"] = demand["demand"].astype("int32")

    # 2) complete the hourly grid so "N rows back" == "N hours back"
    hours = pd.date_range(demand["hour_ts"].min(), demand["hour_ts"].max(), freq="h")
    zones = demand["PULocationID"].unique()
    grid = pd.MultiIndex.from_product(
        [zones, hours], names=["PULocationID", "hour_ts"]).to_frame(index=False)
    demand = grid.merge(demand, on=["PULocationID", "hour_ts"], how="left")
    demand["demand"] = demand["demand"].fillna(0).astype("int32")
    demand = demand.sort_values(["PULocationID", "hour_ts"])

    # 3) lags & rolling (shifted to avoid leaking the current hour)
    g = demand.groupby("PULocationID")["demand"]
    demand["demand_lag_1h"]   = g.shift(1).astype("float32")
    demand["demand_lag_24h"]  = g.shift(24).astype("float32")
    demand["demand_lag_168h"] = g.shift(168).astype("float32")
    demand["demand_roll_3h"]  = g.shift(1).rolling(3).mean().astype("float32")
    demand["demand_roll_24h"] = g.shift(1).rolling(24).mean().astype("float32")

    val = ["demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
           "demand_roll_3h", "demand_roll_24h"]
    return df.merge(demand[["PULocationID", "hour_ts"] + val],
                    on=["PULocationID", "hour_ts"], how="left")


def build_feature_frame(surcharge_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Add all features to the labeled frame and (optionally) cache it."""
    out = add_demand_features(add_time_features(surcharge_df))
    if save:
        out.to_parquet(C.FEATURES_FILE)
        log.info("wrote %s", C.FEATURES_FILE)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .target import build_surge_labels
    df = build_surge_labels(pd.read_parquet(C.CLEAN))
    build_feature_frame(df)

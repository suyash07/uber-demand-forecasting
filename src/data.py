"""
Data layer: load raw TLC trips, join zones + weather, and clean.

Produces `df_clean` (the cleaned trip set used both for demand aggregates and as
the base for label construction). Reads/writes the parquet files declared in
config, so heavy steps run once and are cached to disk.
"""
from __future__ import annotations
import logging
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


def load_and_merge() -> pd.DataFrame:
    """Concatenate Jan+Feb trips, attach borough (zones) and nearest-hour weather."""
    df = pd.concat(
        [pd.read_parquet(C.RAW_JAN), pd.read_parquet(C.RAW_FEB)],
        ignore_index=True,
    )
    zones = pd.read_csv(C.ZONES)
    df = df.merge(zones, left_on="PULocationID", right_on="LocationID", how="left")

    weather = pd.read_parquet(C.WEATHER)
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
    weather["weather_datetime"] = pd.to_datetime(weather["weather_datetime"])

    df = df.sort_values(C.DATE_COL).reset_index(drop=True)
    weather = weather.sort_values("weather_datetime").reset_index(drop=True)
    # nearest-hour join: attaches each trip to the closest weather reading
    df = pd.merge_asof(df, weather, left_on=C.DATE_COL,
                       right_on="weather_datetime", direction="nearest")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and invalid rows; clamp to the valid time window."""
    df = df.copy()
    df["trip_time_mins"] = df["trip_time"] / 60.0

    # drop the top 1% by duration and distance (data errors / anomalies)
    time_cap  = df["trip_time_mins"].quantile(0.99)
    miles_cap = df["trip_miles"].quantile(0.99)
    df = df[(df["trip_time_mins"] <= time_cap) & (df["trip_miles"] <= miles_cap)]

    # must have a borough (drops unmapped zones)
    df = df.dropna(subset=["Borough"])

    # clamp to the window where weather coverage exists
    lo, hi = C.WINDOW
    df = df[(df[C.DATE_COL] >= lo) & (df[C.DATE_COL] < hi)]

    log.info("cleaned rows: %s", f"{len(df):,}")
    return df.reset_index(drop=True)


def build_clean(save: bool = True) -> pd.DataFrame:
    """End-to-end: load -> merge -> clean -> (optionally) cache to disk."""
    df = clean(load_and_merge())
    if save:
        df.to_parquet(C.CLEAN)
        log.info("wrote %s", C.CLEAN)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_clean()

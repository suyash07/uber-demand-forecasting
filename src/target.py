"""
Target construction: engineer the `is_surge` label.

No surge label exists in the data, so we define it: a trip is 'surge' when its
fare exceeds 1.5x the *expected* fare for a trip of that distance and duration.
Expected fare comes from a robust (Huber) regression so surge trips themselves
do not drag the baseline upward.
"""
from __future__ import annotations
import logging
import pandas as pd
from sklearn.linear_model import HuberRegressor

from . import config as C

log = logging.getLogger(__name__)


def build_surge_labels(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid trips, fit expected-fare model, derive is_surge."""
    df = df_clean.copy()

    # rows that would break the fare ratio (fixed-cost-dominated / invalid)
    df = df[(df["trip_miles"] >= C.MIN_MILES) &
            (df["base_passenger_fare"] >= C.MIN_FARE) &
            (df["trip_time_mins"] > 0)].copy()

    # fit expected fare on a fare-capped subset so extreme fares don't skew it
    cap = df["base_passenger_fare"].quantile(C.FARE_CAP_Q)
    fit = df[df["base_passenger_fare"] <= cap]
    model = HuberRegressor().fit(
        fit[["trip_miles", "trip_time_mins"]], fit["base_passenger_fare"]
    )

    df["expected_fare"] = model.predict(df[["trip_miles", "trip_time_mins"]])
    df = df[df["expected_fare"] > 0].copy()                 # guard denominator

    df["surge_mult"] = df["base_passenger_fare"] / df["expected_fare"]
    df[C.TARGET] = (df["surge_mult"] > C.SURGE_MULT_THRESHOLD).astype(int)

    log.info("surge rate: %.3f", df[C.TARGET].mean())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_surge_labels(pd.read_parquet(C.CLEAN))
    print(df[C.TARGET].value_counts(normalize=True))

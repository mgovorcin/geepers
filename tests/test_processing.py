"""Tests for GPS/InSAR time-grid alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.processing import merge_gps_insar


@pytest.fixture
def daily_gps():
    """60 days of daily GPS, starting 2020-01-01."""
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "los_gps": np.linspace(0.0, 0.06, len(dates)),
            "sigma_los": np.full(len(dates), 0.001),
        },
        index=dates,
    )


@pytest.fixture
def acquisitions():
    """5 acquisitions on a 12-day repeat, all inside the GPS window."""
    dates = pd.date_range("2020-01-06", periods=5, freq="12D")
    return pd.DataFrame(
        {
            "los_insar": np.arange(len(dates), dtype=float) * 0.01,
            "temporal_coherence": np.full(len(dates), 0.9),
        },
        index=dates,
    )


def test_each_acquisition_appears_exactly_once(daily_gps, acquisitions):
    """Regression: asof-matching InSAR onto the daily GPS index replicated
    every acquisition onto each GPS day within the tolerance (up to 3 rows),
    inflating its weight in every downstream statistic.
    """
    merged = merge_gps_insar(daily_gps, acquisitions)

    with_insar = merged[merged["los_insar"].notna()]
    assert len(with_insar) == len(acquisitions)
    assert with_insar.index.equals(acquisitions.index)
    np.testing.assert_allclose(
        with_insar["los_insar"].to_numpy(), acquisitions["los_insar"].to_numpy()
    )
    assert not merged.index.has_duplicates


def test_dense_gps_record_is_preserved(daily_gps, acquisitions):
    """GPS days without an acquisition are kept, so rate fits still see
    the full daily series."""
    merged = merge_gps_insar(daily_gps, acquisitions)

    assert merged["los_gps"].notna().sum() == len(daily_gps)
    assert merged.index.is_monotonic_increasing
    assert set(daily_gps.columns) | set(acquisitions.columns) == set(merged.columns)


def test_acquisition_takes_nearest_gps_day_within_tolerance(daily_gps):
    """An acquisition offset from the GPS grid picks up the nearest day."""
    acq = pd.DataFrame({"los_insar": [0.5]}, index=pd.to_datetime(["2020-01-10 18:00"]))
    merged = merge_gps_insar(daily_gps, acq)

    row = merged[merged["los_insar"].notna()]
    assert len(row) == 1
    assert row["los_gps"].iloc[0] == daily_gps.loc["2020-01-11", "los_gps"]


def test_acquisition_outside_gps_coverage_is_kept_with_nan_gps(daily_gps):
    """A gap in the GPS record must not silently drop the acquisition."""
    gapped = daily_gps.drop(pd.date_range("2020-01-20", "2020-01-31", freq="D"))
    acq = pd.DataFrame({"los_insar": [0.5]}, index=pd.to_datetime(["2020-01-25"]))

    merged = merge_gps_insar(gapped, acq)

    assert len(merged[merged["los_insar"].notna()]) == 1
    assert np.isnan(merged.loc["2020-01-25", "los_gps"])


def test_station_without_gps_is_excluded(acquisitions):
    """No GPS in the window means no comparison, so the station drops out."""
    empty = pd.DataFrame(
        {
            "los_gps": pd.Series(dtype="float64"),
            "sigma_los": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([]),
    )
    merged = merge_gps_insar(empty, acquisitions)

    assert merged.empty
    assert set(merged.columns) == {
        "los_gps",
        "sigma_los",
        "los_insar",
        "temporal_coherence",
    }


def test_unsorted_inputs_are_handled(daily_gps, acquisitions):
    merged = merge_gps_insar(daily_gps.iloc[::-1], acquisitions.iloc[::-1])

    assert merged.index.is_monotonic_increasing
    assert merged["los_insar"].notna().sum() == len(acquisitions)

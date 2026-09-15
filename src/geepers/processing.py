"""InSAR data processing functions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

from geepers._types import DatetimeLike
from geepers.constants import SENTINEL_1_WAVELENGTH
from geepers.io import XarrayReader

logger = logging.getLogger("geepers")

# Largest GPS/InSAR time separation still treated as the same epoch.
DEFAULT_EPOCH_TOLERANCE = pd.Timedelta("1D")

PHASE_TO_METERS = float(SENTINEL_1_WAVELENGTH) / (4.0 * np.pi)


def phase_to_meters(wavelength: float) -> float:
    """Return the radians -> meters conversion factor for `wavelength` (m)."""
    return float(wavelength) / (4.0 * np.pi)


def sample_insar(
    reader: XarrayReader,
    stations_df: pd.DataFrame,
    buffer_pixels: int,
    buffer_meters: float | None = None,
) -> xr.DataArray:
    """Sample InSAR data at station locations with optional spatial buffering.

    Parameters
    ----------
    reader : XarrayReader
        InSAR data reader.
    stations_df : pd.DataFrame
        DataFrame with 'lon' and 'lat' columns.
    buffer_pixels : int
        Number of pixels to buffer around each station.
        If >0, samples a window and computes median.
    buffer_meters : float, optional
        Metric radius around each station instead of a pixel count; the
        median is taken over a circular footprint. Takes precedence
        over `buffer_pixels`.

    Returns
    -------
    xr.DataArray
        Array of shape (n_stations, n_times) with InSAR values.

    """
    lons = stations_df.lon.to_numpy()
    lats = stations_df.lat.to_numpy()

    desc = (
        f"Sampling {reader.da.name} (buffer {buffer_meters} m)"
        if buffer_meters is not None
        else f"Sampling {reader.da.name} (buffered by {buffer_pixels} pixels)"
    )
    with TqdmCallback(desc=desc):
        windows_list = reader.read_window(
            lons, lats, buffer_pixels, buffer_meters=buffer_meters
        )
        p = [w.median(dim=("x", "y"), skipna=True) for w in windows_list]
        averaged = xr.concat(p, dim="pixel")
        a = averaged.compute()
    return a


def process_insar_data(
    *,
    reader: XarrayReader,
    df_gps_stations: pd.DataFrame,
    reader_temporal_coherence: XarrayReader | None = None,
    reader_similarity: XarrayReader | None = None,
    insar_buffer: int = 0,
    insar_buffer_meters: float | None = None,
    wavelength: float = SENTINEL_1_WAVELENGTH,
) -> dict[str, pd.DataFrame]:
    """Sample InSAR rasters at all station locations in one pass.

    Parameters
    ----------
    reader : XarrayReader
        `XarrayReader` opened on the displacement stack.
    df_gps_stations
        DataFrame indexed by station name with at least `lon` and `lat`
        columns (decimal degrees).
    reader_temporal_coherence, reader_similarity
        Optional readers to sample alongside displacement to
        compute temporal coherence and phase similarity.
    insar_buffer : int
        Number of pixels to buffer around each GPS station when sampling InSAR
        data. Uses median averaging, ignoring NaN values, to reduce noise through
        spatial averaging. Default is 0 (single pixel).
    insar_buffer_meters : float, optional
        Metric radius (meters) around each station instead of a pixel
        count; sampling uses a circular median footprint. Applied
        identically to every station (reference and secondary alike).
        Takes precedence over `insar_buffer`.
    wavelength : float
        Radar wavelength in meters, used to convert phase (radians) to meters
        when the stack units are not already meters.
        Default is the Sentinel-1 C-band wavelength (~0.0555 m); pass the
        appropriate value for other sensors (e.g. ~0.2384 m for NISAR L-band).

    Returns
    -------
    dict[str, pandas.DataFrame]
        A mapping from station name to a dataframe that contains `los_insar`
        (in meters), `temporal_coherence` and `similarity` columns indexed
        by acquisition date.

    """
    # Sample InSAR data with optional buffering
    los_insar = sample_insar(reader, df_gps_stations, insar_buffer, insar_buffer_meters)

    if reader_temporal_coherence is not None:
        temp_coh = sample_insar(
            reader_temporal_coherence,
            df_gps_stations,
            insar_buffer,
            insar_buffer_meters,
        )
    else:
        temp_coh = None

    if reader_similarity is not None:
        similarity = sample_insar(
            reader_similarity,
            df_gps_stations,
            insar_buffer,
            insar_buffer_meters,
        )
    else:
        similarity = None

    if reader.units not in ("meters", "m"):
        logger.warning(
            "Stack units are %r; converting phase to meters using wavelength=%.4f m.",
            reader.units,
            wavelength,
        )
        los_insar *= phase_to_meters(wavelength)

    station_to_insar: dict[str, pd.DataFrame] = {}
    for i, station in tqdm(
        enumerate(df_gps_stations.index), total=len(df_gps_stations)
    ):
        data = {
            "los_insar": los_insar[i],
        }
        if similarity is not None:
            data["similarity"] = similarity[i]
        if temp_coh is not None:
            data["temporal_coherence"] = temp_coh[i]

        station_to_insar[station] = pd.DataFrame(index=reader.da.time, data=data)

    return station_to_insar


def get_quality_reader(
    quality_files: Sequence[str | Path] | None,
    time_array: Sequence[DatetimeLike],
    file_date_fmt: str = "%Y%m%d",
) -> XarrayReader | None:
    """Create a quality reader from file list and time array.

    Parameters
    ----------
    quality_files
        List of quality files (e.g., temporal coherence, similarity).
    time_array
        Array of time values for the stack.
    file_date_fmt
        Format string for parsing dates from filenames.

    Returns
    -------
    XarrayReader | None
        Quality reader, or None if no files provided.

    """
    if quality_files is None:
        return None

    # If there is only one file per ministack, then we need to use the range reader
    if len(quality_files) < len(time_array):
        return XarrayReader.from_range_file_list(
            quality_files,
            time_array,
            file_date_fmt=file_date_fmt,
            units="unitless",
        )
    else:
        # Otherwise, the reader will be like other readers
        return XarrayReader.from_file_list(
            quality_files, file_date_fmt, units="unitless"
        )


def merge_gps_insar(
    df_gps: pd.DataFrame,
    df_insar: pd.DataFrame,
    tolerance: pd.Timedelta = DEFAULT_EPOCH_TOLERANCE,
) -> pd.DataFrame:
    """Merge one station's daily GPS table with its InSAR acquisitions.

    The two series live on different time grids: GPS is (nearly) daily,
    while InSAR has one sample per acquisition. This aligns them so that

    * every acquisition appears exactly **once**, carrying the GPS value
      of the nearest day within `tolerance`, and
    * the GPS days that no acquisition claimed are kept as extra rows
      (with NaN InSAR), so the dense GPS record is still available to
      velocity fits.

    Aligning the other way round - asof-matching acquisitions onto the
    daily GPS index - replicates a single acquisition onto every GPS day
    within `tolerance` of it (up to three rows for a 1-day tolerance).
    That silently triples the weight of each epoch in every downstream
    statistic and splits one acquisition into several "epochs" that are
    each seen by a different subset of stations.

    Parameters
    ----------
    df_gps
        Indexed by date, with `los_gps` / `sigma_los` columns.
    df_insar
        Indexed by acquisition date, with `los_insar` and optionally
        `temporal_coherence` / `similarity` columns.
    tolerance
        Largest GPS/InSAR time separation still considered the same
        epoch. Default 1 day.

    Returns
    -------
    pandas.DataFrame
        Indexed by the union of acquisition dates and GPS days, sorted,
        with the union of both frames' columns.

    """
    columns = list(df_gps.columns) + list(df_insar.columns)
    if df_gps.empty:
        # No GPS in the window: nothing to compare against, so this
        # station stays out of the comparison table entirely.
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in columns},
            index=df_gps.index[:0],
        )

    df_gps = df_gps.sort_index()
    df_insar = df_insar.sort_index()

    # One row per acquisition, carrying the nearest GPS day within tolerance.
    at_epochs = pd.merge_asof(
        left=df_insar,
        right=df_gps,
        tolerance=tolerance,
        direction="nearest",
        left_index=True,
        right_index=True,
    )

    # Keep the GPS days no acquisition landed on, so the daily record
    # remains available for rate fitting.
    gps_only = df_gps[~df_gps.index.isin(at_epochs.index)]

    merged = pd.concat([at_epochs, gps_only]).sort_index()
    return merged[columns]

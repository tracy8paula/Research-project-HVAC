"""
Prepare the two original Building 59 streams for Research Pipeline V2.

The script:
- reads only zone_co2.csv and zone_temp_interior.csv;
- cleans implausible values without overwriting raw inputs;
- aggregates duplicate timestamps using medians;
- resamples both streams to 10-minute intervals;
- fills only short internal gaps (at most two 10-minute bins);
- aligns the streams by timestamp;
- writes a cleaned dataset and a quality-control report.

No synthetic Uganda data, model training, or test-set tuning occurs here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "windows.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_columns(frame: pd.DataFrame, prefix: str, suffix: str | None = None) -> list[str]:
    columns = [
        name for name in frame.columns
        if name.startswith(prefix) and (suffix is None or name.endswith(suffix))
    ]
    if not columns:
        raise ValueError(f"No sensor columns found for prefix={prefix!r}, suffix={suffix!r}.")
    return columns


def read_sensor_file(path: Path, timestamp_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required CSV not found: {path}\n"
            "Copy zone_co2.csv and zone_temp_interior.csv into data/raw/."
        )
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if timestamp_column not in frame.columns:
        raise ValueError(
            f"{path.name} has no {timestamp_column!r} column. "
            f"Found: {frame.columns.tolist()}"
        )
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_dates = int(frame[timestamp_column].isna().sum())
    frame = frame.dropna(subset=[timestamp_column]).copy()
    frame = frame.sort_values(timestamp_column, kind="mergesort")
    return frame, invalid_dates


def clean_sensor_stream(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    timestamp_column: str,
    lower: float,
    upper: float,
    frequency: str,
    interpolation_limit: int,
) -> tuple[pd.DataFrame, dict]:
    values = frame[sensor_columns].apply(pd.to_numeric, errors="coerce")
    original_missing = int(values.isna().sum().sum())
    invalid_low = int((values < lower).sum().sum())
    invalid_high = int((values > upper).sum().sum())

    # Sensor values outside plausible broad QC bounds are marked missing,
    # not clipped into apparently valid measurements.
    values = values.mask((values < lower) | (values > upper))
    working = pd.concat([frame[[timestamp_column]], values], axis=1)

    duplicate_rows = int(working.duplicated(subset=[timestamp_column], keep=False).sum())
    duplicate_timestamp_groups = int(
        working.loc[
            working.duplicated(subset=[timestamp_column], keep=False),
            timestamp_column
        ].nunique()
    )

    # Duplicate readings are collapsed sensor-by-sensor using the median.
    working = working.groupby(timestamp_column, as_index=True)[sensor_columns].median()
    working = working.sort_index()

    # Resampling to a shared ten-minute interval is done independently
    # for each sensor stream, never by concatenating sensor tables vertically.
    sampled = working.resample(frequency).median()

    # Only short internal gaps may be interpolated. Long gaps remain missing.
    sampled = sampled.interpolate(
        method="time",
        limit=interpolation_limit,
        limit_area="inside"
    )

    sampled.index.name = timestamp_column
    qc = {
        "input_rows_after_valid_timestamp_filter": int(len(frame)),
        "sensor_count": int(len(sensor_columns)),
        "original_missing_sensor_cells": original_missing,
        "values_below_qc_minimum": invalid_low,
        "values_above_qc_maximum": invalid_high,
        "duplicate_rows_involved": duplicate_rows,
        "duplicate_timestamp_groups": duplicate_timestamp_groups,
        "resampled_rows": int(len(sampled)),
        "remaining_missing_sensor_cells_after_short_gap_interpolation": int(sampled.isna().sum().sum()),
        "date_start": str(sampled.index.min()),
        "date_end": str(sampled.index.max()),
    }
    return sampled, qc


def main() -> None:
    config = load_config()
    paths = config["paths"]
    data_cfg = config["data"]
    qc_cfg = data_cfg["quality_control"]
    timestamp = data_cfg["timestamp_column"]
    frequency = data_cfg["resample_frequency"]
    interpolation_limit = int(qc_cfg["interpolation_limit_steps"])

    raw_dir = ROOT / "data" / "raw"
    co2_path = ROOT / paths["co2_csv"].replace("data/raw/", "")
    temp_path = ROOT / paths["temperature_csv"].replace("data/raw/", "")

    print("=" * 72)
    print("RESEARCH PIPELINE V2 - TWO-FILE DATA PREPARATION")
    print("=" * 72)
    print("CO2 source:", co2_path)
    print("Temperature source:", temp_path)

    co2_frame, co2_bad_dates = read_sensor_file(co2_path, timestamp)
    temp_frame, temp_bad_dates = read_sensor_file(temp_path, timestamp)

    co2_columns = require_columns(
        co2_frame,
        data_cfg["co2_columns_prefix"],
        data_cfg["co2_columns_suffix"]
    )
    temp_columns = require_columns(
        temp_frame,
        data_cfg["temperature_columns_prefix"]
    )

    co2_clean, co2_qc = clean_sensor_stream(
        co2_frame, co2_columns, timestamp,
        float(qc_cfg["co2_min_ppm"]), float(qc_cfg["co2_max_ppm"]),
        frequency, interpolation_limit
    )
    temp_clean, temp_qc = clean_sensor_stream(
        temp_frame, temp_columns, timestamp,
        float(qc_cfg["temperature_min_c"]), float(qc_cfg["temperature_max_c"]),
        frequency, interpolation_limit
    )

    # One robust summary per modality at each timestamp. Sensor-level streams
    # are retained in separate output files for the spatial encoding stage.
    co2_summary = pd.DataFrame({
        "co2_ppm": co2_clean.median(axis=1, skipna=True),
        "co2_sensor_count": co2_clean.notna().sum(axis=1),
    })
    temp_summary = pd.DataFrame({
        "temperature_c": temp_clean.median(axis=1, skipna=True),
        "temperature_sensor_count": temp_clean.notna().sum(axis=1),
    })

    merged = co2_summary.join(temp_summary, how="inner")
    merged = merged.dropna(subset=["co2_ppm", "temperature_c"]).sort_index()
    merged.index.name = timestamp

    # Do not bridge long gaps. Assign a new segment whenever adjacent rows
    # are separated by more than the configured continuity threshold.
    max_gap = pd.Timedelta(minutes=int(data_cfg["maximum_gap_minutes_for_continuous_windows"]))
    time_deltas = merged.index.to_series().diff()
    merged["segment_id"] = (time_deltas.isna() | (time_deltas > max_gap)).cumsum().astype(int).to_numpy()

    output_dir = ROOT / paths["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_dir / "building59_two_stream_10min.csv")
    co2_clean.to_csv(output_dir / "co2_sensor_panel_10min.csv")
    temp_clean.to_csv(output_dir / "temperature_sensor_panel_10min.csv")

    qc_report = {
        "source_files": ["zone_co2.csv", "zone_temp_interior.csv"],
        "humidity_included": False,
        "resampling_frequency": frequency,
        "duplicate_aggregation": "median",
        "interpolation": {
            "method": "time",
            "maximum_consecutive_bins": interpolation_limit,
            "only_internal_gaps": True
        },
        "co2_invalid_timestamp_rows": co2_bad_dates,
        "temperature_invalid_timestamp_rows": temp_bad_dates,
        "co2": co2_qc,
        "temperature": temp_qc,
        "aligned_dataset": {
            "rows": int(len(merged)),
            "columns": merged.columns.tolist(),
            "start": str(merged.index.min()) if len(merged) else None,
            "end": str(merged.index.max()) if len(merged) else None,
            "continuous_segments": int(merged["segment_id"].nunique()) if len(merged) else 0,
            "rows_per_segment": merged["segment_id"].value_counts().sort_index().to_dict() if len(merged) else {}
        },
        "important_interpretation": [
            "The two original sensor streams have different collection start dates.",
            "Only their timestamp overlap is retained.",
            "Long time gaps are represented as separate segments and must not be crossed by forecast windows.",
            "The modality summaries are medians across available sensors at each timestamp.",
            "Raw CSV files are never modified."
        ]
    }

    with (output_dir / "data_quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(qc_report, handle, indent=2)

    print("\nCO2 sensors:", len(co2_columns))
    print("Temperature sensors:", len(temp_columns))
    print("Aligned 10-minute rows:", f"{len(merged):,}")
    print("Continuous time segments:", merged["segment_id"].nunique())
    print("\nSaved:")
    print(output_dir / "building59_two_stream_10min.csv")
    print(output_dir / "co2_sensor_panel_10min.csv")
    print(output_dir / "temperature_sensor_panel_10min.csv")
    print(output_dir / "data_quality_report.json")
    print("\nNo Uganda synthetic data or model training was run in this step.")


if __name__ == "__main__":
    main()

"""
Research Pipeline V2 integrity and leakage audit.

Read-only audit: this script does not modify datasets, models, scalers,
partitions, or held-out test results. It checks data preparation outputs,
chronological partitions, timestamp/segment continuity, spatial feature
alignment, training-only scaler provenance, and valid window construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "processed"
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics"
CONFIG_PATH = ROOT / "config" / "windows.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_baselines import contiguous_window_starts  # noqa: E402


CHECKS: list[dict] = []


def record(name: str, passed: bool, details: str, severity: str = "error") -> None:
    CHECKS.append({
        "check": name,
        "status": "PASS" if passed else ("WARN" if severity == "warning" else "FAIL"),
        "details": details,
    })


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path, parse_dates: bool = True) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    return pd.read_csv(path, parse_dates=["date"] if parse_dates else None)


def check_partition(name: str, frame: pd.DataFrame, interval: int, sequence_length: int, horizon: int) -> dict:
    required = {"date", "co2_ppm", "temperature_c", "segment_id"}
    missing = required.difference(frame.columns)
    if missing:
        record(f"{name}: required columns", False, f"Missing {sorted(missing)}")
        return {"rows": len(frame), "windows": 0}

    dates = pd.to_datetime(frame["date"])
    sorted_unique = dates.is_monotonic_increasing and not dates.duplicated().any()
    record(
        f"{name}: timestamps sorted and unique",
        bool(sorted_unique),
        f"rows={len(frame)}, duplicate_timestamps={int(dates.duplicated().sum())}",
    )

    finite_targets = np.isfinite(
        frame[["co2_ppm", "temperature_c"]].to_numpy(dtype=float)
    ).all()
    record(
        f"{name}: finite target values",
        bool(finite_targets),
        "CO2 and temperature contain no NaN or infinite values."
        if finite_targets else "Non-finite target values detected.",
    )

    within_segment = True
    bad_segments = []
    expected = pd.Timedelta(minutes=interval)
    for segment_id, group in frame.groupby("segment_id", sort=False):
        group_dates = pd.to_datetime(group["date"])
        if len(group_dates) > 1 and not group_dates.diff().iloc[1:].eq(expected).all():
            within_segment = False
            bad_segments.append(int(segment_id))
    record(
        f"{name}: exact {interval}-minute continuity within segments",
        within_segment,
        "All segment intervals match the configured sampling interval."
        if within_segment else f"Segments with irregular intervals: {bad_segments}",
    )

    starts = contiguous_window_starts(
        frame["date"],
        frame["segment_id"],
        sequence_length,
        horizon,
        interval,
    )
    total = sequence_length + horizon
    windows_valid = True
    for start in starts:
        window = frame.iloc[start:start + total]
        if len(window) != total or window["segment_id"].nunique() != 1:
            windows_valid = False
            break
        if not pd.to_datetime(window["date"]).diff().iloc[1:].eq(expected).all():
            windows_valid = False
            break
    record(
        f"{name}: generated windows do not cross segments or time gaps",
        windows_valid,
        f"valid_windows={len(starts)}, sequence={sequence_length}, horizon={horizon}",
    )
    return {"rows": len(frame), "windows": len(starts)}


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG_PATH)
    interval = int(config["forecasting"]["sampling_interval_minutes"])
    sequence_length = int(config["forecasting"]["sequence_length"])
    horizon = int(config["forecasting"]["forecast_horizon_steps"])

    print("=" * 76)
    print("RESEARCH PIPELINE V2 - DATA INTEGRITY AND LEAKAGE AUDIT")
    print("=" * 76)
    print("Read-only audit. No model training or held-out test tuning.")

    quality = read_json(OUTPUT / "data_quality_report.json")
    source = read_csv(OUTPUT / "building59_two_stream_10min.csv")
    partitions = {
        name: read_csv(OUTPUT / f"building59_{name}.csv")
        for name in ("train", "validation", "test")
    }

    # Source preparation and quality-report consistency.
    qc_aligned = quality.get("aligned_dataset", {})
    qc_rows_match = int(qc_aligned.get("rows", -1)) == len(source)
    record(
        "prepared data row count matches quality report",
        qc_rows_match,
        f"source_rows={len(source)}, report_rows={qc_aligned.get('rows')}",
    )
    record(
        "prepared data excludes humidity",
        "humidity" not in " ".join(map(str, source.columns)).lower(),
        f"columns={source.columns.tolist()}",
    )
    record(
        "prepared targets contain no missing values",
        not source[["co2_ppm", "temperature_c"]].isna().any().any(),
        "Checks aligned CO2 and temperature target columns.",
    )

    # Partition sizes, order, and timestamp separation.
    expected_ratios = config["split"]
    total_rows = sum(len(frame) for frame in partitions.values())
    sizes_match = total_rows == len(source)
    record(
        "partition row counts sum to prepared dataset",
        sizes_match,
        f"source={len(source)}, train={len(partitions['train'])}, "
        f"validation={len(partitions['validation'])}, test={len(partitions['test'])}",
    )

    partition_dates = {}
    for name, frame in partitions.items():
        partition_dates[name] = pd.to_datetime(frame["date"])
        check_partition(name, frame, interval, sequence_length, horizon)

    no_overlap = (
        set(partition_dates["train"]).isdisjoint(set(partition_dates["validation"]))
        and set(partition_dates["train"]).isdisjoint(set(partition_dates["test"]))
        and set(partition_dates["validation"]).isdisjoint(set(partition_dates["test"]))
    )
    record("no timestamp overlap between partitions", no_overlap,
           "Train, validation and test timestamp sets are disjoint.")

    chronological = (
        partition_dates["train"].max() < partition_dates["validation"].min()
        and partition_dates["validation"].max() < partition_dates["test"].min()
    )
    record(
        "partitions are strictly chronological",
        bool(chronological),
        "Training precedes validation, which precedes held-out test.",
    )

    configured = [
        float(expected_ratios["train_ratio"]),
        float(expected_ratios["validation_ratio"]),
        float(expected_ratios["test_ratio"]),
    ]
    observed = [
        len(partitions["train"]) / len(source),
        len(partitions["validation"]) / len(source),
        len(partitions["test"]) / len(source),
    ]
    ratio_details = {
        key: {"configured": configured[i], "observed": observed[i]}
        for i, key in enumerate(("train", "validation", "test"))
    }
    record(
        "partition proportions match chronological split configuration",
        all(abs(a - b) <= 1.0 / len(source) + 1e-12
            for a, b in zip(configured, observed)),
        json.dumps(ratio_details),
    )

    # Original partitions must not contain synthetic scenario rows.
    synthetic_markers = [
        column for column in partitions["train"].columns
        if column in {"scenario_id", "data_origin", "occupants_assumed",
                      "air_changes_per_hour_assumed"}
    ]
    record(
        "original partitions are separate from synthetic Uganda scenarios",
        not synthetic_markers,
        f"Synthetic-only columns found in original partitions: {synthetic_markers}",
    )

    # Spatial feature files: exact row/timestamp alignment and finite values.
    spatial_frames = {}
    for name, base in partitions.items():
        spatial = read_csv(OUTPUT / f"building59_{name}_spatial_encoded.csv")
        spatial_frames[name] = spatial
        aligned = (
            len(spatial) == len(base)
            and pd.to_datetime(spatial["date"]).equals(pd.to_datetime(base["date"]))
            and spatial["segment_id"].equals(base["segment_id"])
        )
        record(
            f"{name}: spatial rows and metadata align with original partition",
            bool(aligned),
            f"base_rows={len(base)}, spatial_rows={len(spatial)}",
        )
        feature_cols = [
            c for c in spatial.columns
            if c.startswith("co2__") or c.startswith("temperature__")
        ]
        finite = bool(feature_cols) and np.isfinite(
            spatial[feature_cols].to_numpy(dtype=float)
        ).all()
        record(
            f"{name}: spatial features are finite",
            finite,
            f"spatial_feature_count={len(feature_cols)}",
        )

    layout = read_json(OUTPUT / "spatial_encoding_layout_and_scalers.json")
    record(
        "spatial layout correctly labels virtual coordinates",
        layout.get("physical_coordinates") is False
        and layout.get("floor_plan_available") is False,
        "Virtual grid is documented as assumed rather than measured.",
    )
    record(
        "spatial reading scaler provenance is training-only",
        layout.get("reading_scaler_fit_partition")
        == "original Building 59 training partition only"
        and layout.get("validation_and_test_used_for_scaler_fit") is False,
        "Checks spatial scaler provenance metadata.",
    )

    # Recompute sensor reading extrema from training timestamps and compare
    # them with the saved spatial encoding scaler values.
    for modality, filename, key in [
        ("co2", "co2_sensor_panel_10min.csv", "co2_training_scaler"),
        ("temperature", "temperature_sensor_panel_10min.csv",
         "temperature_training_scaler"),
    ]:
        panel = read_csv(OUTPUT / filename).set_index("date").sort_index()
        train_dates = pd.DatetimeIndex(partitions["train"]["date"])
        train_panel = panel.reindex(train_dates)
        saved = layout.get(key, {})
        scaler_matches = True
        compared = 0
        for sensor in train_panel.columns:
            values = pd.to_numeric(train_panel[sensor], errors="coerce")
            if sensor not in saved or not values.notna().any():
                scaler_matches = False
                continue
            params = saved[sensor]
            scaler_matches &= bool(
                np.isclose(float(values.min()), float(params["minimum"]))
                and np.isclose(float(values.max()), float(params["maximum"]))
            )
            compared += 1
        record(
            f"{modality}: saved sensor scaler matches training-only extrema",
            bool(scaler_matches and compared == len(train_panel.columns)),
            f"sensor_scalers_compared={compared}",
        )

    # Baseline target/input scaler provenance and train-only extrema.
    scaler_path = MODELS / "baseline_scalers.json"
    scaler_ok = False
    scaler_detail = "baseline_scalers.json not found."
    if scaler_path.is_file():
        saved = read_json(scaler_path)
        train = partitions["train"]
        train_targets = train[["co2_ppm", "temperature_c"]].to_numpy(dtype=float)
        target_min = np.asarray(saved.get("minimum", []), dtype=float)
        target_max = np.asarray(saved.get("maximum", []), dtype=float)
        scaler_ok = (
            target_min.shape == (2,)
            and target_max.shape == (2,)
            and np.allclose(target_min, np.nanmin(train_targets, axis=0))
            and np.allclose(target_max, np.nanmax(train_targets, axis=0))
        )
        scaler_detail = (
            f"saved_min={target_min.tolist()}, saved_max={target_max.tolist()}, "
            f"train_min={np.nanmin(train_targets, axis=0).tolist()}, "
            f"train_max={np.nanmax(train_targets, axis=0).tolist()}"
        )
    record("baseline target scaler matches training-only target extrema",
           bool(scaler_ok), scaler_detail)

    # Test and validation are inspected as data only; no models are loaded.
    record(
        "audit did not load trained models or alter held-out test",
        True,
        "This script reads partitions and preprocessing metadata only.",
    )

    report = {
        "audit": "research_pipeline_v2_data_integrity_and_leakage",
        "read_only": True,
        "test_used_for_tuning": False,
        "configuration": {
            "sequence_length": sequence_length,
            "forecast_horizon_steps": horizon,
            "sampling_interval_minutes": interval,
        },
        "partition_rows_and_windows": {
            name: {
                "rows": int(len(frame)),
                "valid_windows": int(contiguous_window_starts(
                    frame["date"], frame["segment_id"],
                    sequence_length, horizon, interval
                ).__len__()),
            }
            for name, frame in partitions.items()
        },
        "checks": CHECKS,
        "summary": {
            "passed": sum(c["status"] == "PASS" for c in CHECKS),
            "warnings": sum(c["status"] == "WARN" for c in CHECKS),
            "failed": sum(c["status"] == "FAIL" for c in CHECKS),
            "overall": "PASS" if all(c["status"] == "PASS" for c in CHECKS) else "REVIEW_REQUIRED",
        },
    }
    json_path = METRICS / "data_integrity_audit.json"
    csv_path = METRICS / "data_integrity_audit.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame(CHECKS).to_csv(csv_path, index=False)

    print("\nAUDIT CHECKS")
    print(pd.DataFrame(CHECKS).to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(report["summary"], indent=2))
    print("\nSAVED")
    print(json_path)
    print(csv_path)
    print("\nDATA INTEGRITY AUDIT COMPLETE.")


if __name__ == "__main__":
    main()

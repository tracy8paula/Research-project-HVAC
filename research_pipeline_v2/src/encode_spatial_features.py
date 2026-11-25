"""
Build training-only-scaled quantum-inspired spatial sine/cosine features.

Virtual coordinates are deterministic modelling assumptions because no floor
plan or verified sensor-location records are available. They must not be
described as measured physical positions.

The scaler is fitted on the original Building 59 training partition only.
Validation and test values are transformed with those frozen training values.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "processed"


def load_split(name: str) -> pd.DataFrame:
    path = OUTPUT / f"building59_{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run split_and_generate_uganda.py first.")
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def load_sensor_panel(filename: str) -> pd.DataFrame:
    path = OUTPUT / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing sensor panel: {path}. Run prepare_data.py first.")
    panel = pd.read_csv(path, parse_dates=["date"])
    if "date" not in panel.columns:
        raise ValueError(f"{filename} must contain a date column.")
    return panel.set_index("date").sort_index()


def virtual_grid(sensor_names: list[str]) -> dict[str, tuple[float, float]]:
    """Assign stable grid coordinates in [0, 1] using sorted sensor names."""
    names = sorted(sensor_names)
    count = len(names)
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    coordinates = {}

    for index, name in enumerate(names):
        row, column = divmod(index, columns)
        x = 0.5 if columns == 1 else column / (columns - 1)
        y = 0.5 if rows == 1 else row / (rows - 1)
        coordinates[name] = (float(x), float(y))

    return coordinates


def fit_reading_scaler(
    training_panel: pd.DataFrame,
    sensor_names: list[str],
) -> dict:
    scaler = {}
    for sensor in sensor_names:
        values = pd.to_numeric(training_panel[sensor], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        minimum = float(values.min()) if values.notna().any() else median
        maximum = float(values.max()) if values.notna().any() else median
        scaler[sensor] = {
            "minimum": minimum,
            "maximum": maximum,
            "training_median": median,
        }
    return scaler


def encode_panel(
    panel: pd.DataFrame,
    sensor_names: list[str],
    coordinates: dict[str, tuple[float, float]],
    scaler: dict,
    prefix: str,
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=panel.index)

    for sensor in sensor_names:
        x, y = coordinates[sensor]
        theta_x = 2.0 * np.pi * x
        theta_y = 2.0 * np.pi * y

        values = pd.to_numeric(panel[sensor], errors="coerce")
        params = scaler[sensor]
        values = values.fillna(params["training_median"])

        span = params["maximum"] - params["minimum"]
        if span <= 0:
            normalized = pd.Series(0.5, index=values.index)
        else:
            normalized = ((values - params["minimum"]) / span).clip(0.0, 1.0)

        theta_reading = normalized * np.pi
        safe_name = str(sensor).replace(" ", "_")

        encoded[f"{prefix}__{safe_name}__sin_x"] = np.sin(theta_x)
        encoded[f"{prefix}__{safe_name}__cos_x"] = np.cos(theta_x)
        encoded[f"{prefix}__{safe_name}__sin_y"] = np.sin(theta_y)
        encoded[f"{prefix}__{safe_name}__cos_y"] = np.cos(theta_y)
        encoded[f"{prefix}__{safe_name}__sin_reading"] = np.sin(theta_reading)
        encoded[f"{prefix}__{safe_name}__cos_reading"] = np.cos(theta_reading)

    return encoded


def main() -> None:
    print("=" * 72)
    print("RESEARCH PIPELINE V2 - SPATIAL SINE/COSINE FEATURE ENCODING")
    print("=" * 72)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    co2_panel = load_sensor_panel("co2_sensor_panel_10min.csv")
    temp_panel = load_sensor_panel("temperature_sensor_panel_10min.csv")

    co2_sensors = list(co2_panel.columns)
    temp_sensors = list(temp_panel.columns)

    if len(co2_sensors) != 11 or len(temp_sensors) != 16:
        print(
            "WARNING: Sensor counts differ from the observed preparation run:",
            len(co2_sensors), "CO2 and", len(temp_sensors), "temperature."
        )

    train_dates = pd.DatetimeIndex(train["date"])
    validation_dates = pd.DatetimeIndex(validation["date"])
    test_dates = pd.DatetimeIndex(test["date"])

    # Select sensor-panel rows using the exact timestamp membership of each
    # original partition. No validation/test row contributes to scaler fitting.
    train_co2 = co2_panel.reindex(train_dates)
    train_temp = temp_panel.reindex(train_dates)

    co2_scaler = fit_reading_scaler(train_co2, co2_sensors)
    temp_scaler = fit_reading_scaler(train_temp, temp_sensors)

    co2_coordinates = virtual_grid(co2_sensors)
    temp_coordinates = virtual_grid(temp_sensors)

    split_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    split_dates = {
        "train": train_dates,
        "validation": validation_dates,
        "test": test_dates,
    }

    for split_name, base in split_frames.items():
        dates = split_dates[split_name]
        co2_part = co2_panel.reindex(dates)
        temp_part = temp_panel.reindex(dates)

        co2_encoded = encode_panel(
            co2_part, co2_sensors, co2_coordinates, co2_scaler, "co2"
        )
        temp_encoded = encode_panel(
            temp_part, temp_sensors, temp_coordinates, temp_scaler, "temperature"
        )

        # Keep the original aggregate channels for direct baseline comparison.
        base_features = base.set_index("date")[
            ["co2_ppm", "temperature_c", "segment_id"]
        ]
        result = base_features.join(co2_encoded, how="left").join(
            temp_encoded, how="left"
        )
        result.index.name = "date"
        result = result.reset_index()

        expected_rows = len(base)
        if len(result) != expected_rows:
            raise RuntimeError(
                f"{split_name}: feature row count {len(result)} != {expected_rows}"
            )

        result.to_csv(
            OUTPUT / f"building59_{split_name}_spatial_encoded.csv",
            index=False
        )
        print(
            f"{split_name.title():12s}: {len(result):,} rows, "
            f"{result.shape[1] - 1} feature/metadata columns"
        )

    layout = {
        "layout_type": "deterministic_virtual_grid",
        "physical_coordinates": False,
        "floor_plan_available": False,
        "coordinate_range": [0.0, 1.0],
        "position_angle_mapping": "theta = 2*pi*coordinate",
        "reading_angle_mapping": "theta_reading = pi*training_minmax_scaled_reading",
        "reading_scaler_fit_partition": "original Building 59 training partition only",
        "validation_and_test_used_for_scaler_fit": False,
        "co2_sensor_coordinates": {
            name: {"x": xy[0], "y": xy[1]}
            for name, xy in co2_coordinates.items()
        },
        "temperature_sensor_coordinates": {
            name: {"x": xy[0], "y": xy[1]}
            for name, xy in temp_coordinates.items()
        },
        "co2_training_scaler": co2_scaler,
        "temperature_training_scaler": temp_scaler,
        "encoding_per_sensor": [
            "sin_x", "cos_x", "sin_y", "cos_y",
            "sin_reading", "cos_reading"
        ],
        "interpretation": (
            "Virtual coordinates are controlled experimental assumptions, "
            "not measured room positions. Results must be interpreted accordingly."
        ),
    }

    with (OUTPUT / "spatial_encoding_layout_and_scalers.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(layout, handle, indent=2)

    print("\nSaved:")
    for name in ["train", "validation", "test"]:
        print(OUTPUT / f"building59_{name}_spatial_encoded.csv")
    print(OUTPUT / "spatial_encoding_layout_and_scalers.json")
    print("\nScaler fitted on training data only. Validation and test were transform-only.")


if __name__ == "__main__":
    main()

"""
Chronological partitioning and Uganda-context synthetic scenarios.

IMPORTANT:
- Original Building 59 observations remain separate from synthetic scenarios.
- The 70/15/15 split is chronological, never random.
- Synthetic scenarios are generated from TRAINING rows only.
- CO2 scenarios use a well-mixed-room mass-balance model.
- Room layout, occupancy, room volume, outdoor CO2 and ventilation rates
  are explicit modelling assumptions, not measured Ugandan building data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "windows.json"
INPUT_PATH = ROOT / "outputs" / "processed" / "building59_two_stream_10min.csv"

# Explicit scenario assumptions. These are illustrative model parameters,
# not measurements from Building 59 or a Ugandan building.
SCENARIOS = {
    "baseline_naturally_ventilated": {
        "occupants": 2,
        "air_changes_per_hour": 0.70,
        "temperature_offset_c": 0.0,
    },
    "high_occupancy": {
        "occupants": 5,
        "air_changes_per_hour": 0.70,
        "temperature_offset_c": 0.0,
    },
    "low_ventilation": {
        "occupants": 2,
        "air_changes_per_hour": 0.25,
        "temperature_offset_c": 0.0,
    },
    "high_occupancy_low_ventilation": {
        "occupants": 5,
        "air_changes_per_hour": 0.25,
        "temperature_offset_c": 0.0,
    },
    "warm_indoor_conditions": {
        "occupants": 2,
        "air_changes_per_hour": 0.70,
        "temperature_offset_c": 2.0,
    },
}

# Published engineering-style assumptions used for the synthetic model.
ROOM_VOLUME_M3 = 300.0
OUTDOOR_CO2_PPM = 420.0
CO2_GENERATION_LITRES_PER_SECOND_PER_PERSON = 0.018
TIME_STEP_MINUTES = 10


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def chronological_split(frame: pd.DataFrame, config: dict):
    split_cfg = config["split"]
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation and test ratios must sum to 1.")

    frame = frame.sort_values("date").reset_index(drop=True)
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError(f"Not enough rows for chronological splitting: {n}")

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:val_end].copy()
    test = frame.iloc[val_end:].copy()
    return train, validation, test


def simulate_co2_segment(
    initial_co2_ppm: float,
    number_of_rows: int,
    occupants: int,
    air_changes_per_hour: float,
) -> np.ndarray:
    """Simulate indoor CO2 using a well-mixed single-zone mass balance."""
    if air_changes_per_hour <= 0:
        raise ValueError("Air changes per hour must be greater than zero.")

    dt_hours = TIME_STEP_MINUTES / 60.0
    decay = np.exp(-air_changes_per_hour * dt_hours)

    generation_m3_per_hour = (
        occupants
        * CO2_GENERATION_LITRES_PER_SECOND_PER_PERSON
        * 3.6
        / 1000.0
    )

    generation_term_ppm = (
        generation_m3_per_hour
        / (air_changes_per_hour * ROOM_VOLUME_M3)
        * (1.0 - decay)
        * 1_000_000.0
    )

    values = np.empty(number_of_rows, dtype=float)
    values[0] = float(initial_co2_ppm)

    for i in range(1, number_of_rows):
        values[i] = (
            OUTDOOR_CO2_PPM
            + (values[i - 1] - OUTDOOR_CO2_PPM) * decay
            + generation_term_ppm
        )

    return values


def generate_uganda_scenarios(training: pd.DataFrame) -> pd.DataFrame:
    records = []

    # Each original continuous segment is simulated independently.
    for scenario_name, assumptions in SCENARIOS.items():
        for source_segment_id, segment in training.groupby("segment_id", sort=False):
            segment = segment.sort_values("date").reset_index(drop=True)
            if segment.empty:
                continue

            simulated_co2 = simulate_co2_segment(
                initial_co2_ppm=float(segment.loc[0, "co2_ppm"]),
                number_of_rows=len(segment),
                occupants=int(assumptions["occupants"]),
                air_changes_per_hour=float(assumptions["air_changes_per_hour"]),
            )

            scenario = pd.DataFrame({
                "date": segment["date"],
                "co2_ppm": simulated_co2,
                "temperature_c": (
                    segment["temperature_c"].astype(float)
                    + float(assumptions["temperature_offset_c"])
                ),
                "scenario_id": scenario_name,
                "segment_id": int(source_segment_id),
                "data_origin": "synthetic_uganda_context_scenario",
                "source_data": "Building 59 training partition used as initial/context sequence",
                "occupants_assumed": int(assumptions["occupants"]),
                "air_changes_per_hour_assumed": float(
                    assumptions["air_changes_per_hour"]
                ),
            })
            records.append(scenario)

    if not records:
        raise ValueError("No synthetic scenarios were generated from training data.")

    return pd.concat(records, ignore_index=True)


def main() -> None:
    config = load_config()
    output_dir = ROOT / config["paths"]["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Prepared dataset not found: {INPUT_PATH}\n"
            "Run prepare_data.py first."
        )

    print("=" * 72)
    print("RESEARCH PIPELINE V2 - CHRONOLOGICAL SPLIT + UGANDA SCENARIOS")
    print("=" * 72)

    original = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    required = {"date", "co2_ppm", "temperature_c", "segment_id"}
    missing = required.difference(original.columns)
    if missing:
        raise ValueError(f"Prepared dataset is missing columns: {sorted(missing)}")

    original = original.dropna(
        subset=["date", "co2_ppm", "temperature_c", "segment_id"]
    ).sort_values("date").reset_index(drop=True)

    train, validation, test = chronological_split(original, config)

    # Save original observations separately. No synthetic rows enter these files.
    train.to_csv(output_dir / "building59_train.csv", index=False)
    validation.to_csv(output_dir / "building59_validation.csv", index=False)
    test.to_csv(output_dir / "building59_test.csv", index=False)

    # Generate scenarios from training observations only.
    synthetic = generate_uganda_scenarios(train)
    synthetic.to_csv(
        output_dir / "uganda_synthetic_training_scenarios.csv",
        index=False
    )

    partition_summary = {
        "source_dataset": "building59_two_stream_10min.csv",
        "source_observations": int(len(original)),
        "method": "chronological row-order split",
        "ratios_configured": {
            "train": config["split"]["train_ratio"],
            "validation": config["split"]["validation_ratio"],
            "test": config["split"]["test_ratio"],
        },
        "partitions": {
            "train": {
                "rows": int(len(train)),
                "start": str(train["date"].min()),
                "end": str(train["date"].max()),
                "segments": int(train["segment_id"].nunique()),
            },
            "validation": {
                "rows": int(len(validation)),
                "start": str(validation["date"].min()),
                "end": str(validation["date"].max()),
                "segments": int(validation["segment_id"].nunique()),
            },
            "test": {
                "rows": int(len(test)),
                "start": str(test["date"].min()),
                "end": str(test["date"].max()),
                "segments": int(test["segment_id"].nunique()),
            },
        },
        "synthetic_scenario_rows": int(len(synthetic)),
        "synthetic_scenarios": list(SCENARIOS.keys()),
        "synthetic_generation_source": "training partition only",
        "validation_and_test_augmented": False,
        "random_row_split": False,
        "note": (
            "Keep windows inside both their partition and original continuous "
            "segment. Synthetic scenarios are not measured Ugandan observations."
        ),
    }

    assumptions = {
        "dataset_label": "synthetic_uganda_context_scenarios",
        "measured_uganda_data": False,
        "generation_method": "well_mixed_single_zone_CO2_mass_balance",
        "generation_source": "Building 59 training partition only",
        "room_volume_m3_assumed": ROOM_VOLUME_M3,
        "outdoor_co2_ppm_assumed": OUTDOOR_CO2_PPM,
        "co2_generation_litres_per_second_per_person_assumed": (
            CO2_GENERATION_LITRES_PER_SECOND_PER_PERSON
        ),
        "time_step_minutes": TIME_STEP_MINUTES,
        "temperature_offsets_are_synthetic": True,
        "humidity_included": False,
        "pm25_included": False,
        "pm10_included": False,
        "scenarios": SCENARIOS,
        "limitations": [
            "No Ugandan building measurements are claimed or fabricated.",
            "No floor plan or measured sensor coordinates are available.",
            "Room volume, occupancy and ventilation rates are modelling assumptions.",
            "The warm-indoor scenario applies an explicit temperature offset.",
            "Synthetic scenarios are for controlled robustness experiments, not calibration."
        ],
    }

    with (output_dir / "partition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(partition_summary, f, indent=2)

    with (output_dir / "uganda_scenario_assumptions.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(assumptions, f, indent=2)

    print("\nORIGINAL BUILDING 59 PARTITIONS")
    print(f"Training:   {len(train):,} rows ({len(train) / len(original):.1%})")
    print(f"Validation: {len(validation):,} rows ({len(validation) / len(original):.1%})")
    print(f"Test:       {len(test):,} rows ({len(test) / len(original):.1%})")
    print("\nUGANDA-CONTEXT SYNTHETIC SCENARIOS")
    for name in SCENARIOS:
        count = int((synthetic["scenario_id"] == name).sum())
        print(f"{name}: {count:,} rows")
    print(f"Total synthetic rows: {len(synthetic):,}")

    print("\nSAVED FILES")
    for filename in [
        "building59_train.csv",
        "building59_validation.csv",
        "building59_test.csv",
        "uganda_synthetic_training_scenarios.csv",
        "partition_summary.json",
        "uganda_scenario_assumptions.json",
    ]:
        print(output_dir / filename)

    print("\nValidation and test data were not used to generate synthetic scenarios.")


if __name__ == "__main__":
    main()

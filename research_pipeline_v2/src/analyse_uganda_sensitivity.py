"""
One-at-a-time sensitivity analysis for Uganda-context synthetic CO2 scenarios.

This is a deterministic model-assumption sensitivity analysis, not a test-set
evaluation, predictive-accuracy assessment, or validation against Ugandan
measurements. All simulations use Building 59 TRAINING-partition segment
timestamps and initial CO2 states only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "outputs" / "processed"
METRICS = ROOT / "outputs" / "metrics"
TRAIN_FILE = PROCESSED / "building59_train.csv"
ASSUMPTIONS_FILE = PROCESSED / "uganda_scenario_assumptions.json"

DT_MINUTES = 10
# Exploratory ranges around the current illustrative baseline assumptions.
# They are sensitivity settings, not asserted Ugandan population parameters.
SENSITIVITY_VALUES = {
    "occupants": [1, 2, 5, 8],
    "air_changes_per_hour": [0.25, 0.50, 0.70, 1.00, 1.50],
    "room_volume_m3": [150.0, 225.0, 300.0, 450.0, 600.0],
    "outdoor_co2_ppm": [400.0, 420.0, 450.0, 500.0],
    "generation_lps_person": [0.012, 0.018, 0.027],
}
TEMPERATURE_OFFSETS_C = [-1.0, 0.0, 1.0, 2.0]


def simulate_segment(
    initial_co2: float,
    rows: int,
    occupants: int,
    ach: float,
    volume: float,
    outdoor_co2: float,
    generation_lps_person: float,
) -> np.ndarray:
    if rows < 1:
        raise ValueError("A segment must contain at least one row.")
    if occupants < 0 or ach <= 0 or volume <= 0 or generation_lps_person < 0:
        raise ValueError("Invalid physical assumption in sensitivity configuration.")

    dt_hours = DT_MINUTES / 60.0
    decay = np.exp(-ach * dt_hours)
    generation_m3_hour = occupants * generation_lps_person * 3.6
    generation_term = (
        generation_m3_hour / (ach * volume)
        * (1.0 - decay) * 1_000_000.0
    )

    values = np.empty(rows, dtype=float)
    values[0] = float(initial_co2)
    for i in range(1, rows):
        values[i] = (
            outdoor_co2
            + (values[i - 1] - outdoor_co2) * decay
            + generation_term
        )
    return values


def steady_state(
    occupants: int,
    ach: float,
    volume: float,
    outdoor_co2: float,
    generation_lps_person: float,
) -> float:
    generation_m3_hour = occupants * generation_lps_person * 3.6
    return float(
        outdoor_co2
        + generation_m3_hour / (ach * volume) * 1_000_000.0
    )


def main() -> None:
    if not TRAIN_FILE.is_file() or not ASSUMPTIONS_FILE.is_file():
        raise FileNotFoundError(
            "Required training partition or scenario assumptions are missing. "
            "Run split_and_generate_uganda.py first."
        )

    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    with ASSUMPTIONS_FILE.open("r", encoding="utf-8") as handle:
        assumptions = json.load(handle)

    required = {"date", "co2_ppm", "temperature_c", "segment_id"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training file is missing columns: {sorted(missing)}")
    if assumptions.get("measured_uganda_data") is not False:
        raise ValueError("Input assumptions must identify scenarios as synthetic.")

    train = train.dropna(subset=list(required)).copy()
    train = train.sort_values(["segment_id", "date"]).reset_index(drop=True)

    base = {
        "occupants": int(assumptions["scenarios"]["baseline_naturally_ventilated"]["occupants"]),
        "air_changes_per_hour": float(assumptions["scenarios"]["baseline_naturally_ventilated"]["air_changes_per_hour"]),
        "room_volume_m3": float(assumptions["room_volume_m3_assumed"]),
        "outdoor_co2_ppm": float(assumptions["outdoor_co2_ppm_assumed"]),
        "generation_lps_person": float(
            assumptions["co2_generation_litres_per_second_per_person_assumed"]
        ),
    }

    rows = []
    baseline_values_by_segment = {}
    for segment_id, segment in train.groupby("segment_id", sort=True):
        segment = segment.sort_values("date")
        baseline_values_by_segment[int(segment_id)] = simulate_segment(
            float(segment["co2_ppm"].iloc[0]),
            len(segment),
            base["occupants"],
            base["air_changes_per_hour"],
            base["room_volume_m3"],
            base["outdoor_co2_ppm"],
            base["generation_lps_person"],
        )

    # One-factor-at-a-time: all other parameters remain at their baseline.
    for parameter, values in SENSITIVITY_VALUES.items():
        for value in values:
            settings = dict(base)
            settings[parameter] = value
            all_co2 = []
            segment_rows = []

            for segment_id, segment in train.groupby("segment_id", sort=True):
                segment = segment.sort_values("date")
                simulated = simulate_segment(
                    float(segment["co2_ppm"].iloc[0]),
                    len(segment),
                    settings["occupants"],
                    settings["air_changes_per_hour"],
                    settings["room_volume_m3"],
                    settings["outdoor_co2_ppm"],
                    settings["generation_lps_person"],
                )
                all_co2.append(simulated)
                segment_rows.append({
                    "segment_id": int(segment_id),
                    "rows": int(len(segment)),
                    "co2_mean_ppm": float(np.mean(simulated)),
                    "co2_p95_ppm": float(np.quantile(simulated, 0.95)),
                    "co2_max_ppm": float(np.max(simulated)),
                    "co2_end_ppm": float(simulated[-1]),
                })

            combined = np.concatenate(all_co2)
            baseline_combined = np.concatenate(list(baseline_values_by_segment.values()))
            rows.append({
                "analysis_type": "co2_one_factor_at_a_time",
                "parameter_changed": parameter,
                "parameter_value": value,
                "baseline_parameter_value": base[parameter],
                "assumed_occupants": settings["occupants"],
                "assumed_air_changes_per_hour": settings["air_changes_per_hour"],
                "assumed_room_volume_m3": settings["room_volume_m3"],
                "assumed_outdoor_co2_ppm": settings["outdoor_co2_ppm"],
                "assumed_generation_lps_person": settings["generation_lps_person"],
                "theoretical_steady_state_co2_ppm": steady_state(
                    settings["occupants"],
                    settings["air_changes_per_hour"],
                    settings["room_volume_m3"],
                    settings["outdoor_co2_ppm"],
                    settings["generation_lps_person"],
                ),
                "co2_mean_ppm": float(np.mean(combined)),
                "co2_p95_ppm": float(np.quantile(combined, 0.95)),
                "co2_max_ppm": float(np.max(combined)),
                "mean_difference_from_baseline_ppm": float(
                    np.mean(combined - baseline_combined)
                ),
                "segment_count": int(len(segment_rows)),
            })

    sensitivity_df = pd.DataFrame(rows)

    # Temperature offset is an independent additive assumption and does not
    # alter the CO2 mass-balance calculation.
    temperature_rows = []
    base_temperature = train["temperature_c"].to_numpy(dtype=float)
    for offset in TEMPERATURE_OFFSETS_C:
        adjusted = base_temperature + offset
        temperature_rows.append({
            "temperature_offset_c": offset,
            "temperature_mean_c": float(np.mean(adjusted)),
            "temperature_p95_c": float(np.quantile(adjusted, 0.95)),
            "mean_difference_from_unadjusted_c": float(np.mean(adjusted - base_temperature)),
            "max_difference_from_unadjusted_c": float(np.max(np.abs(adjusted - base_temperature))),
        })
    temperature_df = pd.DataFrame(temperature_rows)

    METRICS.mkdir(parents=True, exist_ok=True)
    sensitivity_path = METRICS / "uganda_assumption_sensitivity.csv"
    temperature_path = METRICS / "uganda_temperature_offset_sensitivity.csv"
    report_path = METRICS / "uganda_sensitivity_analysis.json"

    sensitivity_df.to_csv(sensitivity_path, index=False)
    temperature_df.to_csv(temperature_path, index=False)

    report = {
        "analysis_type": "deterministic_one_factor_at_a_time_sensitivity_analysis",
        "data_label": "synthetic_uganda_context_scenarios",
        "measured_uganda_data": False,
        "source_context": "Building 59 training partition only",
        "training_rows_used": int(len(train)),
        "training_segments_used": int(train["segment_id"].nunique()),
        "held_out_validation_partition_loaded": False,
        "held_out_test_partition_loaded": False,
        "trained_models_loaded": False,
        "predictive_accuracy_claimed": False,
        "method": (
            "Each physical assumption is varied independently while all other "
            "assumptions remain at the illustrative baseline. Each segment is "
            "simulated independently from its original training CO2 initial state."
        ),
        "baseline_assumptions": base,
        "exploratory_parameter_values": SENSITIVITY_VALUES,
        "temperature_offsets_c": TEMPERATURE_OFFSETS_C,
        "co2_sensitivity_rows": int(len(sensitivity_df)),
        "temperature_sensitivity_rows": int(len(temperature_df)),
        "limitations": [
            "Sensitivity ranges are exploratory settings, not measured Ugandan building distributions.",
            "One-factor-at-a-time analysis does not estimate interactions between assumptions.",
            "The well-mixed single-zone model omits spatial concentration gradients and complex airflow.",
            "The temperature-offset experiment is an additive scenario, not a thermal model.",
            "No energy consumption, HVAC response, or predictive accuracy is evaluated.",
            "No validation or held-out test data are loaded.",
        ],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 76)
    print("UGANDA SYNTHETIC ASSUMPTION SENSITIVITY ANALYSIS")
    print("=" * 76)
    print("Descriptive model sensitivity only; not measured Ugandan data.")
    print("\nCO2 SENSITIVITY")
    print(sensitivity_df[[
        "parameter_changed", "parameter_value", "co2_mean_ppm",
        "co2_p95_ppm", "theoretical_steady_state_co2_ppm",
        "mean_difference_from_baseline_ppm",
    ]].to_string(index=False))
    print("\nTEMPERATURE OFFSET SENSITIVITY")
    print(temperature_df.to_string(index=False))
    print("\nINTEGRITY")
    print(f"Training rows used: {len(train):,}")
    print(f"Continuous segments: {train['segment_id'].nunique()}")
    print("Validation partition loaded: NO")
    print("Held-out test partition loaded: NO")
    print("Trained models loaded: NO")
    print("Measured Uganda data claimed: NO")
    print("\nSAVED")
    print(sensitivity_path)
    print(temperature_path)
    print(report_path)
    print("\nSENSITIVITY ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()

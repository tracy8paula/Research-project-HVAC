"""
Two-factor occupancy x ventilation interaction experiment.

Uses only Building 59 training-partition timestamps and initial CO2 states.
This is a synthetic model experiment, not measured Ugandan building data,
predictive evaluation, HVAC control evaluation, or energy assessment.
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

OCCUPANCY_LEVELS = [1, 2, 5, 8]
ACH_LEVELS = [0.25, 0.50, 0.70, 1.00, 1.50]
DT_HOURS = 10.0 / 60.0


def simulate(
    initial: float,
    rows: int,
    occupants: int,
    ach: float,
    volume: float,
    outdoor: float,
    generation_lps_person: float,
) -> np.ndarray:
    if rows < 1 or occupants < 0 or ach <= 0 or volume <= 0:
        raise ValueError("Invalid simulation input.")
    decay = np.exp(-ach * DT_HOURS)
    generation_m3_hour = occupants * generation_lps_person * 3.6
    generation_term = (
        generation_m3_hour / (ach * volume)
        * (1.0 - decay) * 1_000_000.0
    )
    result = np.empty(rows, dtype=float)
    result[0] = initial
    for index in range(1, rows):
        result[index] = (
            outdoor + (result[index - 1] - outdoor) * decay + generation_term
        )
    return result


def steady_state(
    occupants: int, ach: float, volume: float, outdoor: float, generation: float
) -> float:
    return float(
        outdoor
        + occupants * generation * 3.6 / (ach * volume) * 1_000_000.0
    )


def main() -> None:
    if not TRAIN_FILE.is_file() or not ASSUMPTIONS_FILE.is_file():
        raise FileNotFoundError(
            "Missing training partition or assumptions. "
            "Run split_and_generate_uganda.py first."
        )

    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    with ASSUMPTIONS_FILE.open("r", encoding="utf-8") as handle:
        assumptions = json.load(handle)

    required = {"date", "co2_ppm", "segment_id"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training file is missing columns: {sorted(missing)}")
    if assumptions.get("measured_uganda_data") is not False:
        raise ValueError("Scenario assumptions must be explicitly synthetic.")

    train = train.dropna(subset=list(required)).copy()
    train = train.sort_values(["segment_id", "date"]).reset_index(drop=True)

    volume = float(assumptions["room_volume_m3_assumed"])
    outdoor = float(assumptions["outdoor_co2_ppm_assumed"])
    generation = float(
        assumptions["co2_generation_litres_per_second_per_person_assumed"]
    )
    base_occ = int(
        assumptions["scenarios"]["baseline_naturally_ventilated"]["occupants"]
    )
    base_ach = float(
        assumptions["scenarios"]["baseline_naturally_ventilated"]["air_changes_per_hour"]
    )

    # Simulate each occupancy/ventilation combination independently for every
    # original continuous training segment; never bridge segment boundaries.
    results = []
    for occupants in OCCUPANCY_LEVELS:
        for ach in ACH_LEVELS:
            segment_arrays = []
            for _, segment in train.groupby("segment_id", sort=True):
                segment = segment.sort_values("date")
                values = simulate(
                    initial=float(segment["co2_ppm"].iloc[0]),
                    rows=len(segment),
                    occupants=occupants,
                    ach=ach,
                    volume=volume,
                    outdoor=outdoor,
                    generation_lps_person=generation,
                )
                segment_arrays.append(values)

            combined = np.concatenate(segment_arrays)
            results.append({
                "occupants": occupants,
                "air_changes_per_hour": ach,
                "room_volume_m3_assumed": volume,
                "outdoor_co2_ppm_assumed": outdoor,
                "generation_lps_person_assumed": generation,
                "co2_mean_ppm": float(np.mean(combined)),
                "co2_p95_ppm": float(np.quantile(combined, 0.95)),
                "co2_max_ppm": float(np.max(combined)),
                "theoretical_steady_state_co2_ppm": steady_state(
                    occupants, ach, volume, outdoor, generation
                ),
                "training_rows_simulated": int(len(combined)),
                "segments_simulated": int(train["segment_id"].nunique()),
            })

    result_df = pd.DataFrame(results)
    baseline_rows = result_df[
        (result_df["occupants"] == base_occ)
        & np.isclose(result_df["air_changes_per_hour"], base_ach)
    ]
    if len(baseline_rows) != 1:
        raise ValueError("Could not identify exactly one factorial baseline.")
    baseline_mean = float(baseline_rows.iloc[0]["co2_mean_ppm"])

    # Additive interaction: observed joint change minus the sum of the
    # separate occupancy and ventilation changes relative to baseline.
    occupancy_effects = (
        result_df[result_df["air_changes_per_hour"].apply(
            lambda x: np.isclose(x, base_ach)
        )][["occupants", "co2_mean_ppm"]]
        .set_index("occupants")["co2_mean_ppm"]
        .to_dict()
    )
    ventilation_effects = (
        result_df[result_df["occupants"] == base_occ]
        .set_index("air_changes_per_hour")["co2_mean_ppm"]
        .to_dict()
    )

    interaction_values = []
    for row in result_df.itertuples(index=False):
        occ_effect = occupancy_effects[int(row.occupants)] - baseline_mean
        ach_key = next(
            key for key in ventilation_effects
            if np.isclose(key, row.air_changes_per_hour)
        )
        ventilation_effect = ventilation_effects[ach_key] - baseline_mean
        joint_effect = row.co2_mean_ppm - baseline_mean
        interaction = joint_effect - occ_effect - ventilation_effect
        interaction_values.append(interaction)

    result_df["mean_difference_from_baseline_ppm"] = (
        result_df["co2_mean_ppm"] - baseline_mean
    )
    result_df["additive_interaction_ppm"] = interaction_values

    matrix = result_df.pivot(
        index="occupants",
        columns="air_changes_per_hour",
        values="co2_mean_ppm",
    ).reset_index()

    METRICS.mkdir(parents=True, exist_ok=True)
    result_path = METRICS / "uganda_occupancy_ventilation_factorial.csv"
    matrix_path = METRICS / "uganda_occupancy_ventilation_matrix.csv"
    report_path = METRICS / "uganda_occupancy_ventilation_interaction.json"

    result_df.to_csv(result_path, index=False)
    matrix.to_csv(matrix_path, index=False)

    report = {
        "analysis_type": "two_factor_occupancy_ventilation_factorial_simulation",
        "data_label": "synthetic_uganda_context_scenarios",
        "measured_uganda_data": False,
        "source_context": "Building 59 training partition only",
        "training_rows": int(len(train)),
        "training_segments": int(train["segment_id"].nunique()),
        "occupancy_levels": OCCUPANCY_LEVELS,
        "air_changes_per_hour_levels": ACH_LEVELS,
        "baseline": {
            "occupants": base_occ,
            "air_changes_per_hour": base_ach,
            "mean_co2_ppm": baseline_mean,
        },
        "fixed_assumptions": {
            "room_volume_m3": volume,
            "outdoor_co2_ppm": outdoor,
            "generation_lps_person": generation,
            "time_step_minutes": 10,
        },
        "interaction_definition": (
            "Additive interaction = joint mean change from baseline minus "
            "the sum of the separate occupancy and ventilation mean changes "
            "from baseline. This is a model-based contrast, not a causal "
            "effect estimated from observations."
        ),
        "held_out_validation_loaded": False,
        "held_out_test_loaded": False,
        "trained_models_loaded": False,
        "predictive_accuracy_claimed": False,
        "energy_or_hvac_control_evaluated": False,
        "limitations": [
            "All outputs are synthetic model simulations, not measured Ugandan data.",
            "Room volume, outdoor CO2 and per-person generation remain fixed assumptions.",
            "Only occupancy and ventilation are varied in this factorial experiment.",
            "The well-mixed single-zone model omits spatial gradients and complex airflow.",
            "The interaction contrast describes this mathematical model and its assumptions; it is not an empirical causal estimate.",
            "No forecasting accuracy, HVAC control performance or energy use is evaluated.",
        ],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 76)
    print("OCCUPANCY x VENTILATION FACTORIAL SENSITIVITY ANALYSIS")
    print("=" * 76)
    print("Synthetic model experiment only; not measured Ugandan building data.")
    print("\nMEAN CO2 MATRIX (ppm)")
    print(matrix.to_string(index=False))
    print("\nADDITIVE INTERACTION CONTRASTS (ppm)")
    print(result_df.pivot(
        index="occupants",
        columns="air_changes_per_hour",
        values="additive_interaction_ppm",
    ).round(3).to_string())
    print("\nINTEGRITY")
    print(f"Training rows used: {len(train):,}")
    print(f"Continuous segments: {train['segment_id'].nunique()}")
    print("Validation partition loaded: NO")
    print("Held-out test partition loaded: NO")
    print("Trained models loaded: NO")
    print("Measured Uganda data claimed: NO")
    print("\nSAVED")
    print(result_path)
    print(matrix_path)
    print(report_path)
    print("\nFACTORIAL ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()

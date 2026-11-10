"""
Audit Uganda-context synthetic scenarios for physical and data integrity.

This is a descriptive scenario audit, NOT an independent predictive test.
The synthetic scenarios are derived from Building 59 training-partition context.
No held-out validation/test data or trained model is loaded.

Outputs:
- outputs/metrics/uganda_scenario_summary.csv
- outputs/metrics/uganda_scenario_segment_summary.csv
- outputs/metrics/uganda_scenario_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "outputs" / "processed"
METRICS = ROOT / "outputs" / "metrics"

SCENARIO_FILE = PROCESSED / "uganda_synthetic_training_scenarios.csv"
ASSUMPTIONS_FILE = PROCESSED / "uganda_scenario_assumptions.json"
BASELINE_NAME = "baseline_naturally_ventilated"
EXPECTED_INTERVAL_MINUTES = 10


def load_inputs() -> tuple[pd.DataFrame, dict]:
    if not SCENARIO_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {SCENARIO_FILE}. Run split_and_generate_uganda.py first."
        )
    if not ASSUMPTIONS_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {ASSUMPTIONS_FILE}. Run split_and_generate_uganda.py first."
        )

    frame = pd.read_csv(SCENARIO_FILE, parse_dates=["date"])
    with ASSUMPTIONS_FILE.open("r", encoding="utf-8") as handle:
        assumptions = json.load(handle)

    required = {
        "date", "co2_ppm", "temperature_c", "scenario_id", "segment_id",
        "data_origin", "occupants_assumed", "air_changes_per_hour_assumed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Synthetic scenario file is missing columns: {sorted(missing)}")
    if assumptions.get("measured_uganda_data") is not False:
        raise ValueError("Scenario assumptions must explicitly identify data as synthetic.")

    frame = frame.sort_values(["scenario_id", "segment_id", "date"]).reset_index(drop=True)
    return frame, assumptions


def validate_scenarios(frame: pd.DataFrame, assumptions: dict) -> dict:
    expected_scenarios = set(assumptions["scenarios"])
    actual_scenarios = set(frame["scenario_id"].dropna().unique())
    if actual_scenarios != expected_scenarios:
        raise ValueError(
            f"Scenario names differ. Expected {sorted(expected_scenarios)}, "
            f"found {sorted(actual_scenarios)}."
        )

    if not (frame["data_origin"] == "synthetic_uganda_context_scenario").all():
        raise ValueError("Unexpected data_origin found in synthetic scenario file.")
    if frame[["co2_ppm", "temperature_c"]].isna().any().any():
        raise ValueError("Missing CO2 or temperature values found.")
    if not np.isfinite(frame[["co2_ppm", "temperature_c"]].to_numpy()).all():
        raise ValueError("Non-finite CO2 or temperature values found.")
    if (frame["co2_ppm"] <= 0).any():
        raise ValueError("Non-positive CO2 values found.")
    if (frame["air_changes_per_hour_assumed"] <= 0).any():
        raise ValueError("Non-positive ventilation assumptions found.")

    # Every scenario must preserve the same source timestamps per segment.
    reference = (
        frame[frame["scenario_id"] == BASELINE_NAME]
        .set_index(["segment_id", "date"])
        .sort_index()
    )
    if reference.empty:
        raise ValueError(f"Required reference scenario '{BASELINE_NAME}' is missing.")

    timestamp_alignment = {}
    for name in sorted(actual_scenarios):
        current = (
            frame[frame["scenario_id"] == name]
            .set_index(["segment_id", "date"])
            .sort_index()
        )
        aligned = current.index.equals(reference.index)
        timestamp_alignment[name] = bool(aligned)
        if not aligned:
            raise ValueError(f"Timestamp/segment alignment failed for scenario {name}.")

    # Confirm exact 10-minute intervals within each continuous source segment.
    interval_checks = []
    for (scenario, segment_id), group in frame.groupby(
        ["scenario_id", "segment_id"], sort=False
    ):
        dates = group.sort_values("date")["date"]
        deltas = dates.diff().dropna()
        interval_checks.append({
            "scenario_id": scenario,
            "segment_id": int(segment_id),
            "rows": int(len(group)),
            "all_intervals_10_minutes": bool(
                deltas.eq(pd.Timedelta(minutes=EXPECTED_INTERVAL_MINUTES)).all()
            ),
        })
    failed_intervals = [
        item for item in interval_checks if not item["all_intervals_10_minutes"]
    ]
    if failed_intervals:
        raise ValueError(
            "Unexpected timestamp gaps found inside scenario segments: "
            + json.dumps(failed_intervals[:5])
        )

    return {
        "scenario_names": sorted(actual_scenarios),
        "scenario_timestamp_alignment": timestamp_alignment,
        "segment_interval_checks": interval_checks,
        "all_scenarios_share_baseline_timestamps": True,
        "all_within_segment_intervals_are_10_minutes": True,
    }


def theoretical_steady_state_co2(
    occupants: int,
    ach: float,
    room_volume_m3: float,
    outdoor_co2_ppm: float,
    generation_lps_person: float,
) -> float:
    # Steady-state well-mixed single-zone mass-balance estimate.
    generation_m3_per_hour = occupants * generation_lps_person * 3.6
    return float(
        outdoor_co2_ppm
        + generation_m3_per_hour / (ach * room_volume_m3) * 1_000_000.0
    )


def main() -> None:
    print("=" * 76)
    print("RESEARCH PIPELINE V2 - UGANDA SYNTHETIC SCENARIO AUDIT")
    print("=" * 76)
    print("This is a descriptive simulation audit, not predictive test evaluation.")

    frame, assumptions = load_inputs()
    validation = validate_scenarios(frame, assumptions)

    room_volume = float(assumptions["room_volume_m3_assumed"])
    outdoor_co2 = float(assumptions["outdoor_co2_ppm_assumed"])
    generation = float(
        assumptions["co2_generation_litres_per_second_per_person_assumed"]
    )

    baseline = (
        frame[frame["scenario_id"] == BASELINE_NAME]
        .set_index(["segment_id", "date"])
        .sort_index()
    )

    overall_rows = []
    segment_rows = []
    scenario_diagnostics = {}

    for scenario_name, scenario in frame.groupby("scenario_id", sort=True):
        scenario = scenario.sort_values(["segment_id", "date"]).copy()
        matched = scenario.set_index(["segment_id", "date"]).sort_index()
        matched_baseline = baseline.reindex(matched.index)

        if matched_baseline[["co2_ppm", "temperature_c"]].isna().any().any():
            raise ValueError(f"Could not pair {scenario_name} with baseline rows.")

        assumptions_for_scenario = assumptions["scenarios"][scenario_name]
        occupants = int(assumptions_for_scenario["occupants"])
        ach = float(assumptions_for_scenario["air_changes_per_hour"])
        temperature_offset = float(
            assumptions_for_scenario["temperature_offset_c"]
        )
        equilibrium = theoretical_steady_state_co2(
            occupants, ach, room_volume, outdoor_co2, generation
        )

        co2_delta = (
            matched["co2_ppm"].to_numpy()
            - matched_baseline["co2_ppm"].to_numpy()
        )
        temp_delta = (
            matched["temperature_c"].to_numpy()
            - matched_baseline["temperature_c"].to_numpy()
        )

        row = {
            "scenario_id": scenario_name,
            "rows": int(len(scenario)),
            "segments": int(scenario["segment_id"].nunique()),
            "assumed_occupants": occupants,
            "assumed_air_changes_per_hour": ach,
            "assumed_temperature_offset_c": temperature_offset,
            "theoretical_steady_state_co2_ppm": equilibrium,
            "co2_mean_ppm": float(scenario["co2_ppm"].mean()),
            "co2_median_ppm": float(scenario["co2_ppm"].median()),
            "co2_p95_ppm": float(scenario["co2_ppm"].quantile(0.95)),
            "co2_max_ppm": float(scenario["co2_ppm"].max()),
            "temperature_mean_c": float(scenario["temperature_c"].mean()),
            "temperature_p95_c": float(scenario["temperature_c"].quantile(0.95)),
            "temperature_max_c": float(scenario["temperature_c"].max()),
            "mean_co2_difference_vs_baseline_ppm": float(np.mean(co2_delta)),
            "p95_co2_difference_vs_baseline_ppm": float(np.quantile(co2_delta, 0.95)),
            "mean_temperature_difference_vs_baseline_c": float(np.mean(temp_delta)),
            "max_absolute_temperature_difference_vs_assumed_offset_c": float(
                np.max(np.abs(temp_delta - temperature_offset))
            ),
        }
        overall_rows.append(row)

        segment_diagnostics = []
        for segment_id, segment in scenario.groupby("segment_id", sort=True):
            segment = segment.sort_values("date")
            paired = matched_baseline.loc[(int(segment_id), slice(None)), :]
            paired = paired.reindex(
                pd.MultiIndex.from_arrays(
                    [segment["segment_id"].astype(int), segment["date"]],
                    names=["segment_id", "date"],
                )
            )
            segment_co2_delta = (
                segment["co2_ppm"].to_numpy()
                - paired["co2_ppm"].to_numpy()
            )
            segment_temp_delta = (
                segment["temperature_c"].to_numpy()
                - paired["temperature_c"].to_numpy()
            )

            first_co2_delta = float(segment_co2_delta[0])
            segment_row = {
                "scenario_id": scenario_name,
                "segment_id": int(segment_id),
                "rows": int(len(segment)),
                "co2_start_ppm": float(segment["co2_ppm"].iloc[0]),
                "co2_end_ppm": float(segment["co2_ppm"].iloc[-1]),
                "co2_mean_ppm": float(segment["co2_ppm"].mean()),
                "co2_p95_ppm": float(segment["co2_ppm"].quantile(0.95)),
                "co2_max_ppm": float(segment["co2_ppm"].max()),
                "co2_mean_difference_vs_baseline_ppm": float(
                    np.mean(segment_co2_delta)
                ),
                "temperature_mean_c": float(segment["temperature_c"].mean()),
                "temperature_mean_difference_vs_baseline_c": float(
                    np.mean(segment_temp_delta)
                ),
                "initial_co2_difference_vs_baseline_ppm": first_co2_delta,
            }
            segment_rows.append(segment_row)
            segment_diagnostics.append(segment_row)

        scenario_diagnostics[scenario_name] = {
            "assumptions": assumptions_for_scenario,
            "theoretical_steady_state_co2_ppm": equilibrium,
            "observed_max_co2_ppm": row["co2_max_ppm"],
            "observed_p95_co2_ppm": row["co2_p95_ppm"],
            "mean_co2_difference_vs_baseline_ppm": row[
                "mean_co2_difference_vs_baseline_ppm"
            ],
            "mean_temperature_difference_vs_baseline_c": row[
                "mean_temperature_difference_vs_baseline_c"
            ],
            "segments": segment_diagnostics,
        }

    overall_df = pd.DataFrame(overall_rows).sort_values("scenario_id")
    segment_df = pd.DataFrame(segment_rows).sort_values(
        ["scenario_id", "segment_id"]
    )

    # Verify that each scenario begins from the same CO2 state as its matched
    # baseline segment. This checks the intended controlled comparison.
    start_differences = segment_df["initial_co2_difference_vs_baseline_ppm"].abs()
    initial_state_match = bool(np.allclose(start_differences, 0.0, atol=1e-8))
    if not initial_state_match:
        raise ValueError("Scenario initial CO2 states do not match the baseline.")

    # Verify that the temperature offset is applied exactly relative to baseline.
    offset_errors = overall_df[
        "max_absolute_temperature_difference_vs_assumed_offset_c"
    ]
    temperature_offset_match = bool(np.allclose(offset_errors, 0.0, atol=1e-8))
    if not temperature_offset_match:
        raise ValueError(
            "One or more scenario temperature offsets do not match their assumptions."
        )

    METRICS.mkdir(parents=True, exist_ok=True)
    summary_csv = METRICS / "uganda_scenario_summary.csv"
    segment_csv = METRICS / "uganda_scenario_segment_summary.csv"
    audit_json = METRICS / "uganda_scenario_audit.json"

    overall_df.to_csv(summary_csv, index=False)
    segment_df.to_csv(segment_csv, index=False)

    audit = {
        "evaluation_type": "descriptive_synthetic_scenario_audit",
        "dataset_label": assumptions["dataset_label"],
        "measured_uganda_data": False,
        "source": assumptions["generation_source"],
        "held_out_test_partition_loaded": False,
        "validation_partition_loaded": False,
        "trained_models_loaded": False,
        "predictive_accuracy_claimed": False,
        "scenario_row_count": int(len(frame)),
        "scenario_count": int(frame["scenario_id"].nunique()),
        "assumed_room_volume_m3": room_volume,
        "assumed_outdoor_co2_ppm": outdoor_co2,
        "assumed_co2_generation_lps_person": generation,
        "validation_checks": validation,
        "initial_co2_matches_baseline_per_segment": initial_state_match,
        "temperature_offsets_match_assumptions": temperature_offset_match,
        "scenario_results": scenario_diagnostics,
        "limitations": [
            "These are synthetic scenarios, not measured Ugandan building data.",
            "The scenario seed/context comes from the Building 59 training partition.",
            "Room volume, outdoor CO2, occupancy and air-change rates are assumptions.",
            "Theoretical steady-state CO2 is a model-derived reference, not a measured value.",
            "This audit does not establish forecasting generalisation or HVAC energy savings.",
            "No humidity, PM2.5 or PM10 values are included.",
        ],
    }
    with audit_json.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print("\nSCENARIO SUMMARY")
    columns = [
        "scenario_id", "assumed_occupants", "assumed_air_changes_per_hour",
        "co2_mean_ppm", "co2_p95_ppm", "co2_max_ppm",
        "temperature_mean_c", "mean_co2_difference_vs_baseline_ppm",
        "mean_temperature_difference_vs_baseline_c",
    ]
    print(overall_df[columns].to_string(index=False))

    print("\nINTEGRITY CHECKS")
    print("All scenarios share the baseline timestamps: PASS")
    print("All within-segment intervals are exactly 10 minutes: PASS")
    print("All scenarios start from the baseline CO2 state per segment:",
          "PASS" if initial_state_match else "FAIL")
    print("Temperature offsets match configured assumptions:",
          "PASS" if temperature_offset_match else "FAIL")
    print("Held-out test partition loaded: NO")
    print("Predictive accuracy claimed: NO")

    print("\nSAVED")
    print(summary_csv)
    print(segment_csv)
    print(audit_json)
    print("\nSYNTHETIC SCENARIO AUDIT COMPLETE.")


if __name__ == "__main__":
    main()

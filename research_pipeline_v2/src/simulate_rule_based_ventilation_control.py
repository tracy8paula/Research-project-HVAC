"""
Offline rule-based ventilation-control simulation.

This is a preliminary control-policy experiment, not an AI/RL controller,
HVAC commissioning test, measured energy evaluation, or measured Ugandan
building study. It uses only Building 59 training-segment initial CO2 states
and explicit synthetic occupancy/ventilation assumptions.

The fan-energy quantity is a relative cubic proxy (ACH / high_ACH)^3.
It is dimensionless and must not be reported as kWh or actual energy savings.
The 1000 ppm upper and 900 ppm lower thresholds are illustrative simulation
switching settings, not health limits or universal regulatory thresholds.
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

DT_HOURS = 10.0 / 60.0
OCCUPANCY_LEVELS = [2, 5, 8]
LOW_ACH = 0.25
REFERENCE_ACH = 0.70
HIGH_ACH = 1.00
UPPER_TRIGGER_PPM = 1000.0
LOWER_RELEASE_PPM = 900.0
MIN_DWELL_MINUTES = 30
MIN_DWELL_STEPS = MIN_DWELL_MINUTES // 10


def co2_next(
    current_ppm: float,
    occupants: int,
    ach: float,
    volume_m3: float,
    outdoor_ppm: float,
    generation_lps_person: float,
) -> float:
    decay = np.exp(-ach * DT_HOURS)
    generation_m3_hour = occupants * generation_lps_person * 3.6
    generation_term = (
        generation_m3_hour / (ach * volume_m3)
        * (1.0 - decay) * 1_000_000.0
    )
    return float(
        outdoor_ppm + (current_ppm - outdoor_ppm) * decay + generation_term
    )


def run_policy(
    initial_values: list[tuple[int, np.ndarray]],
    policy: str,
    occupants: int,
    volume: float,
    outdoor: float,
    generation: float,
) -> dict:
    all_co2 = []
    all_ach = []
    total_switches = 0
    segment_summaries = []

    for segment_id, initial_series in initial_values:
        current = float(initial_series[0])
        simulated = np.empty(len(initial_series), dtype=float)
        actions = np.empty(len(initial_series), dtype=float)
        simulated[0] = current

        active_ach = LOW_ACH
        switches = 0
        steps_since_switch = MIN_DWELL_STEPS

        for index in range(len(initial_series)):
            if policy in {
                "rule_based_hysteresis",
                "rule_based_hysteresis_30min_dwell",
            }:
                dwell_required = (
                    MIN_DWELL_STEPS
                    if policy == "rule_based_hysteresis_30min_dwell"
                    else 0
                )
                if (
                    current >= UPPER_TRIGGER_PPM
                    and active_ach != HIGH_ACH
                    and steps_since_switch >= dwell_required
                ):
                    active_ach = HIGH_ACH
                    switches += 1
                    steps_since_switch = 0
                elif (
                    current <= LOWER_RELEASE_PPM
                    and active_ach != LOW_ACH
                    and steps_since_switch >= dwell_required
                ):
                    active_ach = LOW_ACH
                    switches += 1
                    steps_since_switch = 0
                action = active_ach
            elif policy == "fixed_low":
                action = LOW_ACH
            elif policy == "fixed_reference":
                action = REFERENCE_ACH
            elif policy == "fixed_high":
                action = HIGH_ACH
            else:
                raise ValueError(f"Unknown policy: {policy}")

            actions[index] = action
            steps_since_switch += 1
            if index + 1 < len(initial_series):
                current = co2_next(
                    current, occupants, action, volume, outdoor, generation
                )
                simulated[index + 1] = current

        proxy = (actions / HIGH_ACH) ** 3
        above = simulated > UPPER_TRIGGER_PPM
        segment_summaries.append({
            "segment_id": int(segment_id),
            "rows": int(len(simulated)),
            "co2_mean_ppm": float(np.mean(simulated)),
            "co2_p95_ppm": float(np.quantile(simulated, 0.95)),
            "co2_max_ppm": float(np.max(simulated)),
            "fraction_time_above_upper_trigger": float(np.mean(above)),
            "mean_relative_fan_energy_proxy": float(np.mean(proxy)),
            "relative_fan_energy_proxy_sum": float(np.sum(proxy)),
            "low_ach_fraction": float(np.mean(np.isclose(actions, LOW_ACH))),
            "high_ach_fraction": float(np.mean(np.isclose(actions, HIGH_ACH))),
            "control_switches": int(switches),
        })
        all_co2.append(simulated)
        all_ach.append(actions)
        total_switches += switches

    co2 = np.concatenate(all_co2)
    ach_values = np.concatenate(all_ach)
    proxy = (ach_values / HIGH_ACH) ** 3
    return {
        "policy": policy,
        "occupants_assumed": occupants,
        "rows_simulated": int(len(co2)),
        "segments_simulated": int(len(segment_summaries)),
        "co2_mean_ppm": float(np.mean(co2)),
        "co2_median_ppm": float(np.median(co2)),
        "co2_p95_ppm": float(np.quantile(co2, 0.95)),
        "co2_max_ppm": float(np.max(co2)),
        "fraction_time_above_upper_trigger": float(np.mean(co2 > UPPER_TRIGGER_PPM)),
        "mean_relative_fan_energy_proxy": float(np.mean(proxy)),
        "relative_fan_energy_proxy_sum": float(np.sum(proxy)),
        "low_ach_fraction": float(np.mean(np.isclose(ach_values, LOW_ACH))),
        "reference_ach_fraction": float(np.mean(np.isclose(ach_values, REFERENCE_ACH))),
        "high_ach_fraction": float(np.mean(np.isclose(ach_values, HIGH_ACH))),
        "control_switches": int(total_switches),
        "segment_results": segment_summaries,
    }


def main() -> None:
    if not TRAIN_FILE.is_file() or not ASSUMPTIONS_FILE.is_file():
        raise FileNotFoundError(
            "Missing training partition or scenario assumptions. "
            "Run split_and_generate_uganda.py first."
        )

    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    with ASSUMPTIONS_FILE.open("r", encoding="utf-8") as handle:
        assumptions = json.load(handle)

    required = {"date", "co2_ppm", "segment_id"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training partition is missing columns: {sorted(missing)}")
    if assumptions.get("measured_uganda_data") is not False:
        raise ValueError("Input scenario assumptions must be labelled synthetic.")

    train = train.dropna(subset=list(required)).copy()
    train = train.sort_values(["segment_id", "date"]).reset_index(drop=True)

    volume = float(assumptions["room_volume_m3_assumed"])
    outdoor = float(assumptions["outdoor_co2_ppm_assumed"])
    generation = float(
        assumptions["co2_generation_litres_per_second_per_person_assumed"]
    )

    # Use each training segment's first CO2 value only as the initial
    # condition; retain the observed segment length and simulate all later
    # CO2 values from the mass-balance equation.
    segment_inputs = []
    for segment_id, segment in train.groupby("segment_id", sort=True):
        segment = segment.sort_values("date")
        timestamps = segment["date"].diff().dropna()
        if not timestamps.eq(pd.Timedelta(minutes=10)).all():
            raise ValueError(
                f"Training segment {segment_id} contains non-10-minute intervals."
            )
        segment_inputs.append((
            int(segment_id),
            np.full(
                len(segment),
                float(segment["co2_ppm"].iloc[0]),
                dtype=float,
            ),
        ))

    results = []
    detailed = {}
    for occupants in OCCUPANCY_LEVELS:
        for policy in [
            "fixed_low",
            "fixed_reference",
            "fixed_high",
            "rule_based_hysteresis",
            "rule_based_hysteresis_30min_dwell",
        ]:
            outcome = run_policy(
                segment_inputs, policy, occupants, volume, outdoor, generation
            )
            detailed[f"{occupants}_{policy}"] = outcome
            summary = {key: value for key, value in outcome.items() if key != "segment_results"}
            results.append(summary)

    summary_df = pd.DataFrame(results)
    METRICS.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS / "uganda_rule_based_control_summary.csv"
    json_path = METRICS / "uganda_rule_based_control_audit.json"
    summary_df.to_csv(csv_path, index=False)

    report = {
        "analysis_type": "offline_rule_based_ventilation_control_simulation",
        "data_label": "synthetic_uganda_context_control_experiment",
        "measured_uganda_data": False,
        "source_context": "Building 59 training partition initial CO2 values and segment durations only",
        "training_rows_used_for_simulation_duration": int(len(train)),
        "training_segments": int(train["segment_id"].nunique()),
        "occupancy_levels_assumed": OCCUPANCY_LEVELS,
        "policies": [
            "fixed_low",
            "fixed_reference",
            "fixed_high",
            "rule_based_hysteresis",
            "rule_based_hysteresis_30min_dwell",
        ],
        "control_settings": {
            "low_ach": LOW_ACH,
            "reference_ach": REFERENCE_ACH,
            "high_ach": HIGH_ACH,
            "upper_switching_trigger_ppm": UPPER_TRIGGER_PPM,
            "lower_release_trigger_ppm": LOWER_RELEASE_PPM,
            "time_step_minutes": 10,
            "minimum_dwell_minutes_for_dwell_policy": MIN_DWELL_MINUTES,
        },
        "fixed_physical_assumptions": {
            "room_volume_m3": volume,
            "outdoor_co2_ppm": outdoor,
            "co2_generation_lps_person": generation,
        },
        "energy_proxy": {
            "formula": "(selected_ACH / high_ACH) ** 3",
            "unit": "dimensionless relative proxy",
            "actual_kwh_measured": False,
            "physical_hvac_power_model": False,
            "interpretation": (
                "Illustrative cubic fan-energy proxy only. It is not measured "
                "electricity use, actual HVAC energy, or a validated equipment model."
            ),
        },
        "control_trigger_note": (
            "The 1000 ppm upper and 900 ppm lower thresholds are illustrative "
            "simulation switching settings, not health limits or universal standards."
        ),
        "held_out_validation_loaded": False,
        "held_out_test_loaded": False,
        "trained_ai_model_loaded": False,
        "reinforcement_learning_implemented": False,
        "temperature_dynamics_modelled": False,
        "cooling_or_heating_energy_modelled": False,
        "predictive_accuracy_claimed": False,
        "results": detailed,
        "limitations": [
            "This is a rule-based offline simulation, not the proposed learned AI controller.",
            "Occupancy is constant within each simulated segment.",
            "Room volume, outdoor CO2, generation rate, ACH action levels and thresholds are assumptions.",
            "The well-mixed single-zone model omits spatial concentration gradients and complex airflow.",
            "The cubic fan-energy proxy is not kWh and is not calibrated to an actual fan or HVAC unit.",
            "Temperature, thermal comfort, cooling/heating load and total building energy are not simulated.",
            "No physical HVAC device or embedded edge hardware is controlled.",
            "No validation or held-out test partition is loaded.",
            "The 30-minute minimum dwell is an illustrative control constraint, not tuned or validated against physical equipment.",
        ],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 76)
    print("OFFLINE RULE-BASED VENTILATION CONTROL SIMULATION")
    print("=" * 76)
    print("Synthetic model experiment only; not measured Ugandan building data.")
    print("\nPOLICY COMPARISON")
    print(summary_df[[
        "occupants_assumed", "policy", "co2_mean_ppm", "co2_p95_ppm",
        "co2_max_ppm", "fraction_time_above_upper_trigger",
        "mean_relative_fan_energy_proxy", "low_ach_fraction",
        "high_ach_fraction", "control_switches",
    ]].to_string(index=False))
    print("\nINTEGRITY")
    print(f"Training rows used for segment durations: {len(train):,}")
    print(f"Continuous segments: {train['segment_id'].nunique()}")
    print("Validation partition loaded: NO")
    print("Held-out test partition loaded: NO")
    print("Trained AI model loaded: NO")
    print("Actual kWh measured: NO")
    print("Physical HVAC controlled: NO")
    print("\nSAVED")
    print(csv_path)
    print(json_path)
    print("\nThe 30-minute dwell policy is a switching-stability sensitivity test.")
    print("It is not a validated HVAC equipment setting.")
    print("\nRULE-BASED CONTROL SIMULATION COMPLETE.")


if __name__ == "__main__":
    main()

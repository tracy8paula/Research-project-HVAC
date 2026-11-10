"""Run the reproducible Research Pipeline V2 stages in dependency order.

The runner is CPU-compatible and fails immediately if a stage fails.
It does not evaluate or tune against the held-out test partition.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

STAGES = [
    ("01 - Prepare and align original sensor streams", "prepare_data.py"),
    ("02 - Chronological split and synthetic Uganda scenarios", "split_and_generate_uganda.py"),
    ("03 - Quantum-inspired spatial feature encoding", "encode_spatial_features.py"),
    ("04 - Train and compare forecasting baselines", "train_baselines.py"),
    ("05 - QPSO optimisation using validation fitness", "qpso_optimise_spatial_lstm.py"),
    ("06 - Audit synthetic Uganda scenarios", "evaluate_uganda_scenarios.py"),
    ("07 - One-factor sensitivity analysis", "analyse_uganda_sensitivity.py"),
    ("08 - Occupancy and ventilation factorial analysis", "analyse_uganda_factor_interactions.py"),
    ("09 - Rule-based ventilation policy simulation", "simulate_rule_based_ventilation_control.py"),
    ("10 - Data integrity and leakage audit", "audit_data_integrity.py"),
]


def main() -> None:
    raw_files = [
        ROOT / "data" / "raw" / "zone_co2.csv",
        ROOT / "data" / "raw" / "zone_temp_interior.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in raw_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required original CSV files are missing: "
            + ", ".join(missing)
            + ". Mount both files into data/raw before running the pipeline."
        )

    print("=" * 78, flush=True)
    print("RESEARCH PIPELINE V2 - END-TO-END EXECUTION", flush=True)
    print("Execution mode: CPU-compatible; fail-fast", flush=True)
    print("Held-out test policy: no test evaluation or tuning in this runner", flush=True)
    print("=" * 78, flush=True)

    for label, filename in STAGES:
        script = SRC / filename
        if not script.is_file():
            raise FileNotFoundError(f"Pipeline stage script not found: {script}")

        print(f"\n\n{'=' * 78}", flush=True)
        print(label, flush=True)
        print(f"Running: {filename}", flush=True)
        print("=" * 78, flush=True)

        subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=True,
        )

    print("\n" + "=" * 78, flush=True)
    print("PIPELINE COMPLETED: ALL 10 STAGES EXITED SUCCESSFULLY", flush=True)
    print("Review outputs/metrics/data_integrity_audit.json before presenting.", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()

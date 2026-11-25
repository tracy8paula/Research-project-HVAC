"""
PROTECTED FINAL-TEST EVALUATION RECORD.

The chronological held-out test partition was already opened and evaluated
before the validation objective was corrected to equal-weight normalized MAE.

Those earlier results are historical, pre-correction evidence. Re-evaluating
the test partition with newly trained checkpoints would be a post-hoc analysis,
not an independent blind test, because the original test results have already
been inspected. This script therefore intentionally DOES NOT load the test
partition, load model checkpoints, or write/overwrite evaluation results.

For the corrected experiments, report validation-selected model results and
retain the original held-out test outputs with an explicit pre-correction
qualification. Do not use the historical test ranking for model selection.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"

HISTORICAL_JSON = METRICS / "final_held_out_test_results.json"
HISTORICAL_CSV = METRICS / "final_held_out_test_comparison.csv"


def main() -> int:
    print("=" * 76)
    print("PROTECTED HISTORICAL HELD-OUT TEST RECORD")
    print("=" * 76)
    print("The test partition was evaluated before the validation objective correction.")
    print("Those results are historical pre-correction evidence, not a fresh blind test.")
    print("This script will not load test data, load checkpoints, or overwrite results.")

    missing = [
        str(path)
        for path in (HISTORICAL_JSON, HISTORICAL_CSV)
        if not path.is_file()
    ]
    if missing:
        print("\nHistorical result artifact(s) missing:")
        for path in missing:
            print("-", path)
        print("Do not recreate them by rerunning the held-out test.")
        return 1

    try:
        with HISTORICAL_JSON.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print("\nCould not read the historical JSON record:", error)
        return 1

    print("\nHistorical JSON:", HISTORICAL_JSON)
    print("Historical CSV:", HISTORICAL_CSV)
    print("Recorded partition:", record.get("evaluation_partition", "not recorded"))
    print(
        "Recorded test-selection flag:",
        record.get("test_used_for_model_selection_or_tuning", "not recorded"),
    )
    print("\nPROTECTION CHECK PASSED. No test data was opened or modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

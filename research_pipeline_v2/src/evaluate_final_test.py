"""
One-time held-out test evaluation for Research Pipeline V2.

Loads the chronological held-out test partition only in this final stage.
Compares persistence, non-spatial LSTM, spatial LSTM, and QPSO spatial LSTM.
No model selection or tuning is performed here.

Also reports a simulated local CPU inference benchmark for the QPSO model.
This is not an embedded-device measurement.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_baselines import (
    ForecastLSTM,
    TARGETS,
    build_dataset,
    calculate_metrics,
    load_config,
    load_partition,
    predict_model,
    fit_minmax,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "processed"
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics"


def load_saved_scalers() -> tuple[dict, dict, dict]:
    path = MODELS / "baseline_scalers.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing training-fitted scalers: {path}")
    with path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    target_scaler = {
        "minimum": np.asarray(saved["minimum"], dtype=np.float32),
        "maximum": np.asarray(saved["maximum"], dtype=np.float32),
        "span": np.asarray(saved["span"], dtype=np.float32),
    }
    nonspatial_scaler = {
        "minimum": np.asarray(saved["nonspatial_input_minimum"], dtype=np.float32),
        "maximum": np.asarray(saved["nonspatial_input_maximum"], dtype=np.float32),
        "span": np.asarray(saved["nonspatial_input_span"], dtype=np.float32),
    }
    return saved, target_scaler, nonspatial_scaler


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Required model checkpoint is missing: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


def create_model(checkpoint: dict, device: torch.device) -> ForecastLSTM:
    model = ForecastLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_dimension=int(checkpoint["hidden_dimension"]),
        layers=int(checkpoint["lstm_layers"]),
        dropout=float(checkpoint["dropout"]),
        horizon=int(checkpoint["forecast_horizon_steps"]),
        output_size=len(TARGETS),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def metrics_by_horizon(
    prediction_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    target_scaler: dict,
) -> dict:
    minimum = target_scaler["minimum"]
    span = target_scaler["span"]
    prediction = prediction_scaled * span + minimum
    truth = truth_scaled * span + minimum
    result = {}
    for step in range(prediction.shape[1]):
        step_result = {}
        for index, target in enumerate(TARGETS):
            error = prediction[:, step, index] - truth[:, step, index]
            step_result[target] = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
            }
        step_result["macro_average_mae"] = float(np.mean([
            step_result[target]["mae"] for target in TARGETS
        ]))
        result[f"step_{step + 1}"] = step_result
    return result


def evaluate_model(model, dataset, target_scaler, batch_size, device):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    prediction, truth = predict_model(model, loader, device)
    overall = calculate_metrics(prediction, truth, target_scaler)
    by_horizon = metrics_by_horizon(prediction, truth, target_scaler)
    return overall, by_horizon, prediction, truth


def evaluate_persistence(dataset, target_scaler, batch_size):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    predictions, truths = [], []
    for x, y in loader:
        last_observation = x[:, -1, :2].numpy()
        predictions.append(
            np.repeat(last_observation[:, None, :], y.shape[1], axis=1)
        )
        truths.append(y.numpy())
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    return (
        calculate_metrics(prediction, truth, target_scaler),
        metrics_by_horizon(prediction, truth, target_scaler),
    )


def benchmark_cpu(model, input_size, sequence_length, warmup_runs, measured_runs):
    """Benchmark batch-size-one inference on a local CPU using a zero input."""
    cpu_model = model.to(torch.device("cpu")).eval()
    sample = torch.zeros(1, sequence_length, input_size, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup_runs):
            cpu_model(sample)

        durations_ms = []
        for _ in range(measured_runs):
            start = time.perf_counter()
            cpu_model(sample)
            durations_ms.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(durations_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    return {
        "benchmark_type": "simulated_local_cpu_batch_size_one_inference",
        "device": "CPU",
        "input_tensor_shape": list(sample.shape),
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
        "mean_latency_ms": mean_ms,
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)),
        "throughput_inferences_per_second": float(1000.0 / mean_ms) if mean_ms else None,
        "embedded_device_tested": False,
    }


def main():
    print("=" * 76)
    print("RESEARCH PIPELINE V2 - FINAL HELD-OUT TEST EVALUATION")
    print("=" * 76)

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["forecasting"]["batch_size"])
    sequence_length = int(config["forecasting"]["sequence_length"])
    horizon = int(config["forecasting"]["forecast_horizon_steps"])
    interval = int(config["forecasting"]["sampling_interval_minutes"])

    saved_scalers, target_scaler, nonspatial_scaler = load_saved_scalers()
    spatial_feature_order = saved_scalers["spatial_input_order"]
    nonspatial_feature_order = saved_scalers["nonspatial_input_order"]

    # Confirm the persistence input scale and target scale are identical.
    if not (
        np.allclose(nonspatial_scaler["minimum"], target_scaler["minimum"])
        and np.allclose(nonspatial_scaler["span"], target_scaler["span"])
    ):
        raise ValueError(
            "Persistence input scaler differs from target scaler; "
            "the persistence baseline cannot be evaluated safely."
        )

    # Confirm every selected checkpoint exists BEFORE opening the test files.
    checkpoint_paths = {
        "lstm_without_spatial": MODELS / "lstm_without_spatial.pt",
        "lstm_with_spatial": MODELS / "lstm_with_spatial.pt",
        "qpso_spatial_lstm": MODELS / "qpso_spatial_lstm.pt",
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected checkpoint: {path}")

    print("Device:", device)
    print("Sequence length:", sequence_length)
    print("Forecast horizon:", horizon, "steps")
    print("Sampling interval:", interval, "minutes")
    print("Target scaler: original training partition only")
    print("Model selection: completed before this test evaluation")
    print("Opening held-out test partition now. Do not use these results to retune.")

    # Each held-out partition file is read once. No training or validation data
    # is used to fit or modify any scaler in this script.
    test_base = load_partition("test")
    test_spatial = load_partition("test", spatial=True)

    if len(test_base) != len(test_spatial):
        raise ValueError("Base and spatial test partitions have different row counts.")
    if not pd.to_datetime(test_base["date"]).equals(
        pd.to_datetime(test_spatial["date"])
    ):
        raise ValueError("Base and spatial test timestamps are not aligned.")

    test_nonspatial, _, _ = build_dataset(
        test_base,
        nonspatial_feature_order,
        nonspatial_scaler,
        target_scaler,
        sequence_length,
        horizon,
        interval,
    )
    test_spatial_dataset, _, _ = build_dataset(
        test_spatial,
        spatial_feature_order,
        None,
        target_scaler,
        sequence_length,
        horizon,
        interval,
    )

    if len(test_nonspatial) == 0 or len(test_spatial_dataset) == 0:
        raise ValueError("No valid contiguous held-out test windows were generated.")
    if len(test_nonspatial) != len(test_spatial_dataset):
        raise ValueError("Non-spatial and spatial test window counts do not match.")

    print("Held-out test rows:", len(test_base))
    print("Valid held-out test windows:", len(test_nonspatial))

    results = {
        "evaluation_partition": "chronological_held_out_test",
        "test_partition_opened_in_this_final_stage": True,
        "test_used_for_model_selection_or_tuning": False,
        "test_scaler_fitting": False,
        "sequence_length": sequence_length,
        "forecast_horizon_steps": horizon,
        "sampling_interval_minutes": interval,
        "valid_test_windows": len(test_nonspatial),
        "target_order": TARGETS,
        "target_scaler_fit_partition": "training_only",
        "window_integrity": (
            "All input and target rows share one segment_id and exact "
            "10-minute timestamp differences."
        ),
        "models": {},
    }

    persistence_overall, persistence_steps = evaluate_persistence(
        test_nonspatial, target_scaler, batch_size
    )
    results["models"]["persistence"] = {
        "overall": persistence_overall,
        "by_forecast_step": persistence_steps,
    }

    loaded_models = {}
    for model_name, path in checkpoint_paths.items():
        checkpoint = load_checkpoint(path, device)
        model = create_model(checkpoint, device)
        loaded_models[model_name] = model

        dataset = (
            test_nonspatial
            if model_name == "lstm_without_spatial"
            else test_spatial_dataset
        )
        overall, by_step, _, _ = evaluate_model(
            model, dataset, target_scaler, batch_size, device
        )
        results["models"][model_name] = {
            "overall": overall,
            "by_forecast_step": by_step,
            "checkpoint": path.name,
            "selected_by": checkpoint.get("selected_by", "validation"),
        }

    # Validation results are included for transparent comparison only.
    validation_path = METRICS / "baseline_validation_results.json"
    qpso_validation_path = METRICS / "qpso_optimisation_results.json"
    if validation_path.is_file() and qpso_validation_path.is_file():
        with validation_path.open("r", encoding="utf-8") as handle:
            baseline_validation = json.load(handle)
        with qpso_validation_path.open("r", encoding="utf-8") as handle:
            qpso_validation = json.load(handle)
        results["validation_reference_macro_mae"] = {
            "persistence": baseline_validation["persistence"]["validation_metrics"]["macro_average_mae"],
            "lstm_without_spatial": baseline_validation["lstm_without_spatial"]["validation_metrics"]["macro_average_mae"],
            "lstm_with_spatial": baseline_validation["lstm_with_spatial"]["validation_metrics"]["macro_average_mae"],
            "qpso_spatial_lstm": qpso_validation["full_training_refit_validation_mae"],
        }

    qpso_model = loaded_models["qpso_spatial_lstm"]
    edge_cfg = config.get("edge_benchmark", {})
    results["qpso_local_cpu_benchmark"] = benchmark_cpu(
        qpso_model,
        int(qpso_model.lstm.input_size),
        sequence_length,
        int(edge_cfg.get("warmup_runs", 50)),
        int(edge_cfg.get("measured_runs", 500)),
    )

    # Test winner is descriptive only. It must not trigger another model change.
    ranked = sorted(
        (
            (name, values["overall"]["macro_average_mae"])
            for name, values in results["models"].items()
        ),
        key=lambda item: item[1],
    )
    results["test_ranking_descriptive_only"] = [
        {"rank": index + 1, "model": name, "macro_average_mae": score}
        for index, (name, score) in enumerate(ranked)
    ]
    results["warning"] = (
        "Held-out test results are final evaluation evidence. Do not retune, "
        "reselect, or repeatedly modify models based on this partition."
    )

    output_path = METRICS / "final_held_out_test_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    rows = []
    for model_name, values in results["models"].items():
        overall = values["overall"]
        rows.append({
            "model": model_name,
            "co2_mae_ppm": overall["co2_ppm"]["mae"],
            "co2_rmse_ppm": overall["co2_ppm"]["rmse"],
            "temperature_mae_c": overall["temperature_c"]["mae"],
            "temperature_rmse_c": overall["temperature_c"]["rmse"],
            "macro_average_mae": overall["macro_average_mae"],
        })
    table_path = METRICS / "final_held_out_test_comparison.csv"
    pd.DataFrame(rows).sort_values("macro_average_mae").to_csv(
        table_path, index=False
    )

    print("\nFINAL HELD-OUT TEST RESULTS")
    print(pd.DataFrame(rows).sort_values("macro_average_mae").to_string(index=False))
    print("\nQPSO SPATIAL LSTM - LOCAL CPU BENCHMARK")
    print(json.dumps(results["qpso_local_cpu_benchmark"], indent=2))
    print("\nSaved:", output_path)
    print("Saved:", table_path)
    print("\nTEST EVALUATION COMPLETE. DO NOT USE TEST RESULTS TO RETUNE MODELS.")


if __name__ == "__main__":
    main()

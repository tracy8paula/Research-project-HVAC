
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Constrained CPU benchmark for the IAQ LSTM ONNX model"
    )

    parser.add_argument(
        "--model",
        default="/app/project/lstm_iaq_model.onnx"
    )

    parser.add_argument(
        "--data",
        default="/app/project/building59_uganda_augmented.csv"
    )

    parser.add_argument(
        "--scaler",
        default="/app/project/scaler.pkl"
    )

    parser.add_argument(
        "--config",
        default="/app/project/model_config.json"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=500
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30
    )

    parser.add_argument(
        "--output",
        default="/app/project/edge_benchmark_results.json"
    )

    return parser.parse_args()


def load_configuration(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    required = ["sequence_length", "feature_columns"]

    for key in required:
        if key not in config:
            raise ValueError(
                f"Missing '{key}' in model_config.json"
            )

    return config


def load_input_sequences(
    data_path,
    scaler_path,
    config,
    sequence_length
):
    feature_columns = config["feature_columns"]

    print("\nLoading dataset...")

    data = pd.read_csv(
        data_path,
        usecols=feature_columns,
        low_memory=False
    )

    for column in feature_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(data) <= sequence_length:
        raise ValueError(
            "Not enough valid rows to create input sequences."
        )

    scaler = joblib.load(scaler_path)

    # The scaler must have been fitted on the same
    # feature columns in the same order.
    scaled_data = scaler.transform(
        data[feature_columns].to_numpy(dtype=np.float32)
    )

    scaled_data = np.asarray(
        scaled_data,
        dtype=np.float32
    )

    sequences = []

    for index in range(
        sequence_length,
        len(scaled_data)
    ):
        sequence = scaled_data[
            index - sequence_length:index
        ]

        sequences.append(sequence)

    sequences = np.asarray(
        sequences,
        dtype=np.float32
    )

    print("Valid dataset rows:", len(data))
    print("Input sequences:", len(sequences))
    print("Sequence shape:", sequences.shape)

    return sequences


def main():
    args = parse_arguments()

    model_path = Path(args.model)
    data_path = Path(args.data)
    scaler_path = Path(args.scaler)
    config_path = Path(args.config)

    # --------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------

    required_files = [
        model_path,
        data_path,
        scaler_path,
        config_path
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError(
            "Iterations must be positive and warmup cannot be negative."
        )

    # --------------------------------------------------
    # LOAD CONFIGURATION
    # --------------------------------------------------

    config = load_configuration(str(config_path))

    sequence_length = int(
        config["sequence_length"]
    )

    feature_columns = config["feature_columns"]

    # --------------------------------------------------
    # LOAD ONNX MODEL
    # --------------------------------------------------

    print("\nLoading ONNX model...")

    session_options = ort.SessionOptions()

    # Force CPU execution.
    session_options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("Execution provider:", session.get_providers())
    print("Input name:", input_name)
    print("Output name:", output_name)
    print("Model input shape:", session.get_inputs()[0].shape)

    # --------------------------------------------------
    # LOAD REAL DATA SEQUENCES
    # --------------------------------------------------

    sequences = load_input_sequences(
        str(data_path),
        str(scaler_path),
        config,
        sequence_length
    )

    # Select one representative sequence.
    # Batch size 1 simulates individual edge inference.
    sample = sequences[0:1]

    expected_features = len(feature_columns)

    if sample.shape[2] != expected_features:
        raise ValueError(
            "Input feature count does not match model configuration."
        )

    # --------------------------------------------------
    # WARM-UP
    # --------------------------------------------------

    print("\nRunning warm-up...")

    for _ in range(args.warmup):
        session.run(
            [output_name],
            {input_name: sample}
        )

    # --------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------

    print("\nStarting constrained CPU benchmark...")

    process = psutil.Process(os.getpid())

    latencies_ms = []

    memory_before = (
        process.memory_info().rss / (1024 * 1024)
    )

    cpu_before = psutil.cpu_percent(
        interval=0.2
    )

    for iteration in range(args.iterations):

        start_time = time.perf_counter()

        prediction = session.run(
            [output_name],
            {input_name: sample}
        )

        end_time = time.perf_counter()

        latency = (
            end_time - start_time
        ) * 1000

        latencies_ms.append(latency)

        if (iteration + 1) % 100 == 0:
            print(
                f"Completed {iteration + 1}/"
                f"{args.iterations} inferences"
            )

    cpu_after = psutil.cpu_percent(
        interval=0.2
    )

    memory_after = (
        process.memory_info().rss / (1024 * 1024)
    )

    # --------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------

    latency_array = np.asarray(
        latencies_ms,
        dtype=float
    )

    mean_latency = float(
        np.mean(latency_array)
    )

    median_latency = float(
        np.median(latency_array)
    )

    p95_latency = float(
        np.percentile(latency_array, 95)
    )

    p99_latency = float(
        np.percentile(latency_array, 99)
    )

    throughput = (
        1000 / mean_latency
        if mean_latency > 0 else 0
    )

    model_size_mb = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    results = {
        "benchmark_type": "Constrained Docker CPU simulation",
        "execution_provider": "CPUExecutionProvider",
        "batch_size": 1,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "sequence_length": sequence_length,
        "feature_columns": feature_columns,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_inferences_per_second": throughput,
        "cpu_percent_before": cpu_before,
        "cpu_percent_after": cpu_after,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "model_size_mb": model_size_mb,
        "sample_output_shape": list(
            np.asarray(prediction[0]).shape
        )
    }

    print("\n" + "=" * 55)
    print("IAQ EDGE BENCHMARK RESULTS")
    print("=" * 55)

    for key, value in results.items():
        print(f"{key}: {value}")

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    output_path = Path(args.output)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print("\nResults saved to:", output_path)
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()

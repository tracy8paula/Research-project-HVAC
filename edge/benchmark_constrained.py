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
        description="Resource-constrained ONNX CPU benchmark for the IAQ LSTM"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_cgroup_limit(path):
    try:
        value = Path(path).read_text().strip()
        if value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def resource_limits():
    limits = {}

    # cgroup v2 (common in Docker Desktop/Linux containers)
    cpu_max_path = "/sys/fs/cgroup/cpu.max"
    memory_max_path = "/sys/fs/cgroup/memory.max"

    try:
        cpu_parts = Path(cpu_max_path).read_text().strip().split()
        if len(cpu_parts) == 2 and cpu_parts[0] != "max":
            quota, period = map(int, cpu_parts)
            limits["cpu_limit_cores"] = round(quota / period, 3)
    except (OSError, ValueError, ZeroDivisionError):
        pass

    memory_bytes = read_cgroup_limit(memory_max_path)
    if memory_bytes is not None:
        limits["memory_limit_mb"] = round(memory_bytes / (1024 ** 2), 2)

    # cgroup v1 fallback
    if "cpu_limit_cores" not in limits:
        quota = read_cgroup_limit("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = read_cgroup_limit("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period and quota > 0 and period > 0:
            limits["cpu_limit_cores"] = round(quota / period, 3)

    if "memory_limit_mb" not in limits:
        memory_bytes = read_cgroup_limit(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        )
        if memory_bytes and memory_bytes < 2**60:
            limits["memory_limit_mb"] = round(
                memory_bytes / (1024 ** 2), 2
            )

    return limits


def load_sequences(data_path, scaler_path, config):
    features = config["feature_columns"]
    sequence_length = int(config["sequence_length"])

    frame = pd.read_csv(
        data_path,
        usecols=features,
        low_memory=False
    )
    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) <= sequence_length:
        raise ValueError("Not enough valid data rows for an inference sequence.")

    scaler = joblib.load(scaler_path)

    if getattr(scaler, "n_features_in_", None) != len(features):
        raise ValueError(
            "Scaler feature count does not match model_config.json."
        )

    scaled = scaler.transform(
        frame[features].to_numpy(dtype=np.float32)
    ).astype(np.float32)

    # One real chronological sequence is sufficient for repeated latency tests.
    sample = scaled[-sequence_length:][None, :, :]
    return sample, len(frame)


def main():
    args = parse_arguments()

    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("Iterations must be >= 1 and warmup must be >= 0.")

    model_path = Path(args.model)
    data_path = Path(args.data)
    scaler_path = Path(args.scaler)
    config_path = Path(args.config)
    output_path = Path(args.output)

    for path in (model_path, data_path, scaler_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_config_keys = ("sequence_length", "feature_columns")
    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"Missing '{key}' in model configuration.")

    sample, valid_rows = load_sequences(data_path, scaler_path, config)

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if sample.shape[1:] != (int(config["sequence_length"]),
                            len(config["feature_columns"])):
        raise ValueError("Prepared input shape does not match model configuration.")

    if len(input_meta.shape) != 3:
        raise ValueError(f"Expected a 3D ONNX input, got {input_meta.shape}")

    process = psutil.Process(os.getpid())
    limits = resource_limits()

    print("ONNX provider:", session.get_providers())
    print("Input:", input_meta.name, input_meta.shape)
    print("Output:", output_meta.name, output_meta.shape)
    print("Sample input shape:", sample.shape)
    print("Valid source rows:", valid_rows)
    print("Container limits detected:", limits or "Not exposed by this Docker runtime")

    # Warm-up is excluded from the measured interval.
    for _ in range(args.warmup):
        session.run([output_meta.name], {input_meta.name: sample})

    latencies = []
    memory_before = process.memory_info().rss / (1024 ** 2)
    cpu_times_before = process.cpu_times()
    wall_start = time.perf_counter()

    for i in range(args.iterations):
        start = time.perf_counter()
        output = session.run([output_meta.name], {input_meta.name: sample})
        latencies.append((time.perf_counter() - start) * 1000)

        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{args.iterations} inferences")

    wall_elapsed = time.perf_counter() - wall_start
    cpu_times_after = process.cpu_times()
    memory_after = process.memory_info().rss / (1024 ** 2)

    cpu_seconds = (
        (cpu_times_after.user - cpu_times_before.user)
        + (cpu_times_after.system - cpu_times_before.system)
    )
    cpu_percent_one_core = (
        (cpu_seconds / wall_elapsed) * 100 if wall_elapsed > 0 else 0.0
    )

    values = np.asarray(latencies, dtype=np.float64)
    mean_latency = float(np.mean(values))

    results = {
        "benchmark_type": "Docker resource-constrained ONNX CPU inference",
        "execution_provider": session.get_providers()[0],
        "batch_size": 1,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "sequence_length": int(config["sequence_length"]),
        "feature_columns": config["feature_columns"],
        "mean_latency_ms": mean_latency,
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)),
        "throughput_inferences_per_second": (
            float(args.iterations / wall_elapsed) if wall_elapsed > 0 else 0.0
        ),
        "process_cpu_percent_of_one_core": float(cpu_percent_one_core),
        "process_memory_before_mb": float(memory_before),
        "process_memory_after_mb": float(memory_after),
        "model_size_mb": float(model_path.stat().st_size / (1024 ** 2)),
        "sample_output_shape": list(np.asarray(output[0]).shape),
        "container_limits": limits,
        "wall_time_seconds": float(wall_elapsed)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=4), encoding="utf-8")

    print("\n" + "=" * 60)
    print("CONSTRAINED IAQ EDGE BENCHMARK RESULTS")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"\nResults saved to: {output_path}")
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()

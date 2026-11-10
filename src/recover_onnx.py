"""
Recover the last complete, previously exported IAQ ONNX model.

This script does NOT train anything.
It searches models/archive_previous_run for a matching checkpoint, ONNX model,
scaler and config created by the earlier successful run. It restores those
files to models/ and validates ONNX Runtime inference against the dataset.

Run from the repository root:
    python src/recover_previous_onnx.py

The recovered model is the previous model (the earlier 80/20 workflow), not
the unsaved weights from the interrupted 70/15/15 training run.
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
ARCHIVE_DIR = MODEL_DIR / "archive_previous_run"
DATA_PATH = ROOT / "building59_uganda_augmented.csv"

FILES = {
    "checkpoint": "lstm_iaq_model",
    "onnx": "lstm_iaq_model",
    "scaler": "scaler",
    "config": "model_config",
}
EXTENSIONS = {
    "checkpoint": ".pth",
    "onnx": ".onnx",
    "scaler": ".pkl",
    "config": ".json",
}


def find_latest_complete_archive():
    if not ARCHIVE_DIR.is_dir():
        raise FileNotFoundError(
            f"Archive folder not found: {ARCHIVE_DIR}\n"
            "Do not train again. Check whether the old model files still exist "
            "in models/ or in another backup folder."
        )

    candidates = []
    for folder in ARCHIVE_DIR.iterdir():
        if not folder.is_dir():
            continue

        matched = {}
        for kind, stem in FILES.items():
            suffix = EXTENSIONS[kind]
            matches = sorted(folder.glob(f"{stem}_*{suffix}"))
            if len(matches) == 1:
                matched[kind] = matches[0]

        if len(matched) == len(FILES):
            candidates.append((folder.stat().st_mtime, folder, matched))

    if not candidates:
        # Support the earlier archive layout where timestamped files were
        # copied directly into archive_previous_run/.
        matched = {}
        for kind, stem in FILES.items():
            matches = sorted(ARCHIVE_DIR.glob(f"{stem}_*{EXTENSIONS[kind]}"))
            if matches:
                matched[kind] = max(matches, key=lambda p: p.stat().st_mtime)

        if len(matched) == len(FILES):
            return ARCHIVE_DIR, matched

        raise FileNotFoundError(
            "Could not find a complete archived set containing .pth, .onnx, "
            ".pkl and .json files. No files have been changed."
        )

    _, folder, matched = max(candidates, key=lambda item: item[0])
    return folder, matched


def restore_files(matched):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    destinations = {
        "checkpoint": MODEL_DIR / "lstm_iaq_model.pth",
        "onnx": MODEL_DIR / "lstm_iaq_model.onnx",
        "scaler": MODEL_DIR / "scaler.pkl",
        "config": MODEL_DIR / "model_config.json",
    }

    # Keep any current files before replacing them.
    backup_dir = MODEL_DIR / "recovery_backup"
    existing = [p for p in destinations.values() if p.exists()]
    if existing:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in existing:
            backup = backup_dir / path.name
            if backup.exists():
                backup = backup_dir / f"{path.stem}_backup{path.suffix}"
            shutil.copy2(path, backup)
        print("Existing model files backed up in:", backup_dir)

    for key, source in matched.items():
        shutil.copy2(source, destinations[key])
        print(f"[RESTORED] {source.name} -> {destinations[key].name}")

    return destinations


def validate_restored_set(paths):
    for key, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Restored {key} file is missing or empty: {path}")

    with open(paths["config"], "r", encoding="utf-8") as f:
        config = json.load(f)

    required = ("sequence_length", "feature_columns")
    for key in required:
        if key not in config:
            raise ValueError(f"model_config.json is missing '{key}'.")

    features = config["feature_columns"]
    sequence_length = int(config["sequence_length"])

    scaler = joblib.load(paths["scaler"])
    if getattr(scaler, "n_features_in_", None) != len(features):
        raise ValueError(
            "Archived scaler feature count does not match archived config."
        )

    model = onnx.load(str(paths["onnx"]))
    onnx.checker.check_model(model)

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(paths["onnx"]),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    if "CPUExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX Runtime did not load CPUExecutionProvider.")

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH, usecols=features, low_memory=False)
    frame.columns = frame.columns.str.strip().str.lower()
    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    if len(frame) < sequence_length:
        raise ValueError(
            f"Only {len(frame)} valid rows; need at least {sequence_length}."
        )

    sample_rows = frame[features].tail(sequence_length).to_numpy(dtype=np.float32)
    sample_scaled = scaler.transform(sample_rows).astype(np.float32)
    sample = sample_scaled[np.newaxis, :, :]

    prediction = session.run(
        [output_meta.name],
        {input_meta.name: sample},
    )[0]

    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("ONNX inference returned NaN or infinite values.")

    print("\nVALIDATION PASSED")
    print("ONNX structural check: passed")
    print("ONNX Runtime provider:", session.get_providers())
    print("Input tensor:", input_meta.name, input_meta.shape)
    print("Output tensor:", output_meta.name, output_meta.shape)
    print("Feature columns:", features)
    print("Sequence length:", sequence_length)
    print("Test input shape:", sample.shape)
    print("Test output shape:", prediction.shape)
    print("Model size (MB):", round(paths["onnx"].stat().st_size / (1024 * 1024), 6))
    print("\nRecovered previous ONNX model is ready for the existing Docker benchmark.")
    print("Reminder: this is the archived previous model, not the unsaved 70/15/15 run.")


def main():
    folder, matched = find_latest_complete_archive()
    print("Using archived model set:", folder)
    paths = restore_files(matched)
    validate_restored_set(paths)


if __name__ == "__main__":
    main()

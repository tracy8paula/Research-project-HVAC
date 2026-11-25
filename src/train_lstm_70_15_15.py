"""
70/15/15 chronological training pipeline for the IAQ LSTM.

Important:
- 70% chronological training data
- 15% chronological validation data (early stopping/model selection)
- 15% chronological held-out test data (used only after training)
- MinMaxScaler is fitted on training data only.
- Checkpoint, scaler and config are saved BEFORE ONNX export.
- Uses the legacy torch.onnx.export API compatible with PyTorch 2.2.2:
  do NOT pass dynamo=False.
"""

import copy
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "building59_uganda_augmented.csv"
MODEL_DIR = PROJECT_ROOT / "models"

CHECKPOINT_PATH = MODEL_DIR / "lstm_iaq_model.pth"
ONNX_PATH = MODEL_DIR / "lstm_iaq_model.onnx"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.json"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
PLOT_PATH = MODEL_DIR / "forecast_evaluation.png"

# Keep the feature order consistent across training, scaler, ONNX and Docker.
FEATURE_COLUMNS = ["co2", "temperature", "humidity"]
TARGET_COLUMNS = ["co2", "temperature"]

SEQ_LENGTH = 12
PRED_HORIZON = 3

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. WINDOWED DATASET
# ============================================================

class BuildingDataset(Dataset):
    def __init__(self, scaled_data, seq_len, pred_horizon):
        self.data = torch.from_numpy(
            np.ascontiguousarray(scaled_data, dtype=np.float32)
        )
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.length = len(self.data) - seq_len - pred_horizon + 1

        if self.length <= 0:
            raise ValueError(
                f"Not enough rows for windows: need at least "
                f"{seq_len + pred_horizon}, got {len(self.data)}."
            )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        target_index = index + self.seq_len + self.pred_horizon - 1
        y = self.data[target_index, :len(TARGET_COLUMNS)]
        return x, y


# ============================================================
# 4. MODEL ARCHITECTURE
# ============================================================

class IAQForecastLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(TARGET_COLUMNS))
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


# ============================================================
# 5. LOAD AND SPLIT CHRONOLOGICALLY
# ============================================================

def load_and_prepare_data():
    print("=" * 68)
    print("LOADING DATA AND CREATING CHRONOLOGICAL 70/15/15 SPLIT")
    print("=" * 68)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required model features are missing: {missing}. "
            f"Expected exactly: {FEATURE_COLUMNS}"
        )

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS)

    raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    if not np.isfinite(raw).all():
        raise ValueError("Non-finite values remain after data cleaning.")

    n = len(raw)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train_raw = raw[:train_end]
    validation_raw = raw[train_end:validation_end]
    test_raw = raw[validation_end:]

    # Fit ONLY on the training partition. Never fit on validation or test.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    validation_scaled = scaler.transform(validation_raw).astype(np.float32)
    test_scaled = scaler.transform(test_raw).astype(np.float32)

    train_ds = BuildingDataset(train_scaled, SEQ_LENGTH, PRED_HORIZON)
    validation_ds = BuildingDataset(
        validation_scaled, SEQ_LENGTH, PRED_HORIZON
    )
    test_ds = BuildingDataset(test_scaled, SEQ_LENGTH, PRED_HORIZON)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"Dataset: {DATA_PATH.name}")
    print(f"Valid chronological rows: {n:,}")
    print(f"Training rows (70%): {len(train_raw):,}")
    print(f"Validation rows (15%): {len(validation_raw):,}")
    print(f"Held-out test rows (15%): {len(test_raw):,}")
    print(
        "Window counts: "
        f"train={len(train_ds):,}, "
        f"validation={len(validation_ds):,}, "
        f"test={len(test_ds):,}"
    )
    print("Feature order:", FEATURE_COLUMNS)
    print("Target columns:", TARGET_COLUMNS)
    print("Scaler fitted on training partition only.")

    split_info = {
        "valid_rows": n,
        "train_rows": len(train_raw),
        "validation_rows": len(validation_raw),
        "test_rows": len(test_raw),
        "train_end_index": train_end,
        "validation_end_index": validation_end,
        "train_windows": len(train_ds),
        "validation_windows": len(validation_ds),
        "test_windows": len(test_ds)
    }

    return (
        train_loader,
        validation_loader,
        test_loader,
        scaler,
        split_info
    )


# ============================================================
# 6. TRAIN USING VALIDATION ONLY FOR MODEL SELECTION
# ============================================================

def train_model(train_loader, validation_loader, input_dim):
    model = IAQForecastLSTM(input_dim=input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("\n" + "=" * 68)
    print("STARTING TRAINING")
    print("=" * 68)
    print("Training device:", DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += loss.item() * x.size(0)

        train_loss = train_total / len(train_loader.dataset)

        # Validation is used for early stopping and checkpoint selection.
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for x, y in validation_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                prediction = model(x)
                loss = criterion(prediction, y)
                validation_total += loss.item() * x.size(0)

        validation_loss = validation_total / len(validation_loader.dataset)
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.7f} | "
            f"validation_loss={validation_loss:.7f} | "
            f"lr={current_lr:.7g}",
            flush=True
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print("Best validation checkpoint updated.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping: validation loss did not improve.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    print(f"\nBest validation loss: {best_validation_loss:.7f}")
    return model, best_validation_loss, epoch


# ============================================================
# 7. SAVE A MATCHED ARTIFACT SET BEFORE ONNX EXPORT
# ============================================================

def preserve_existing_artifacts():
    existing = [
        CHECKPOINT_PATH, ONNX_PATH, SCALER_PATH,
        CONFIG_PATH, METRICS_PATH, PLOT_PATH
    ]
    present = [p for p in existing if p.exists()]
    if not present:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = MODEL_DIR / "archive_before_70_15_15" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    for path in present:
        shutil.copy2(path, backup_dir / path.name)

    print(f"\nPrevious model artifacts backed up to: {backup_dir}")
    return str(backup_dir.relative_to(PROJECT_ROOT))


def save_checkpoint_and_scaler(model, scaler):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save to temporary files then atomically replace final paths.
    checkpoint_tmp = CHECKPOINT_PATH.with_suffix(".pth.tmp")
    torch.save(model.state_dict(), checkpoint_tmp)
    checkpoint_tmp.replace(CHECKPOINT_PATH)

    scaler_tmp = SCALER_PATH.with_suffix(".pkl.tmp")
    joblib.dump(scaler, scaler_tmp)
    scaler_tmp.replace(SCALER_PATH)

    print("\n[OK] New 70/15/15 PyTorch checkpoint saved.")
    print("[OK] New training-only scaler saved.")


# ============================================================
# 8. EXPORT ONNX (PyTorch 2.2.2 COMPATIBLE)
# ============================================================

def export_onnx_and_compare(model, test_loader, input_dim):
    print("\n" + "=" * 68)
    print("EXPORTING AND CHECKING ONNX")
    print("=" * 68)

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x[:1].to("cpu")

    with torch.no_grad():
        pytorch_output = model_cpu(sample_x).cpu().numpy()

    temporary_onnx = ONNX_PATH.with_name("lstm_iaq_model.tmp.onnx")

    # PyTorch 2.2.2 uses the legacy exporter by default.
    # Do not pass dynamo=False: that argument is unsupported in this version.
    torch.onnx.export(
        model_cpu,
        sample_x,
        str(temporary_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_sequence"],
        output_names=["forecast_output"],
        dynamic_axes={
            "input_sequence": {0: "batch_size"},
            "forecast_output": {0: "batch_size"}
        }
    )

    onnx.checker.check_model(str(temporary_onnx))
    session = ort.InferenceSession(
        str(temporary_onnx),
        providers=["CPUExecutionProvider"]
    )
    onnx_output = session.run(
        ["forecast_output"],
        {"input_sequence": sample_x.numpy().astype(np.float32)}
    )[0]

    max_abs_difference = float(
        np.max(np.abs(pytorch_output - onnx_output))
    )
    print("ONNX structural check: passed")
    print("ONNX Runtime providers:", session.get_providers())
    print("PyTorch output shape:", tuple(pytorch_output.shape))
    print("ONNX output shape:", tuple(onnx_output.shape))
    print(f"Maximum absolute output difference: {max_abs_difference:.10g}")

    if not np.allclose(
        pytorch_output, onnx_output, rtol=1e-3, atol=1e-5
    ):
        temporary_onnx.unlink(missing_ok=True)
        raise RuntimeError(
            "ONNX output does not sufficiently match PyTorch output. "
            "The checkpoint and scaler have already been saved safely."
        )

    temporary_onnx.replace(ONNX_PATH)
    print("[OK] ONNX export and numerical comparison passed.")
    return max_abs_difference


# ============================================================
# 9. EVALUATE ON THE HELD-OUT TEST PARTITION ONLY
# ============================================================

def evaluate_test_set(model, test_loader, scaler):
    model = model.to("cpu").eval()
    predictions_scaled = []
    actuals_scaled = []

    with torch.no_grad():
        for x, y in test_loader:
            predictions_scaled.append(model(x).numpy())
            actuals_scaled.append(y.numpy())

    predictions_scaled = np.vstack(predictions_scaled)
    actuals_scaled = np.vstack(actuals_scaled)

    # Inverse-transform the two target columns using the full scaler shape.
    input_dim = len(FEATURE_COLUMNS)
    pred_full = np.zeros((len(predictions_scaled), input_dim), dtype=np.float32)
    actual_full = np.zeros((len(actuals_scaled), input_dim), dtype=np.float32)
    pred_full[:, :2] = predictions_scaled
    actual_full[:, :2] = actuals_scaled

    predictions = scaler.inverse_transform(pred_full)[:, :2]
    actuals = scaler.inverse_transform(actual_full)[:, :2]

    errors = predictions - actuals
    metrics = {
        "test_co2_mae_ppm": float(np.mean(np.abs(errors[:, 0]))),
        "test_temperature_mae_c": float(np.mean(np.abs(errors[:, 1]))),
        "test_co2_rmse_ppm": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "test_temperature_rmse_c": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "test_samples": int(len(actuals))
    }

    print("\n" + "=" * 68)
    print("FINAL HELD-OUT TEST RESULTS (15%)")
    print("=" * 68)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    plot_length = min(200, len(actuals))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    axes[0].plot(actuals[:plot_length, 0], label="Actual CO2")
    axes[0].plot(
        predictions[:plot_length, 0],
        label="Predicted CO2",
        linestyle="--"
    )
    axes[0].set_ylabel("CO2 (ppm)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(actuals[:plot_length, 1], label="Actual temperature")
    axes[1].plot(
        predictions[:plot_length, 1],
        label="Predicted temperature",
        linestyle="--"
    )
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    return metrics


# ============================================================
# 10. SAVE CONFIGURATION AND METRICS
# ============================================================

def save_configuration(split_info, best_validation_loss, best_epoch, backup_path):
    config = {
        "model_name": "IAQForecastLSTM",
        "model_format": "ONNX",
        "model_file": "models/lstm_iaq_model.onnx",
        "checkpoint_file": "models/lstm_iaq_model.pth",
        "scaler_file": "models/scaler.pkl",
        "sequence_length": SEQ_LENGTH,
        "forecast_horizon": PRED_HORIZON,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "input_dimension": len(FEATURE_COLUMNS),
        "output_dimension": len(TARGET_COLUMNS),
        "input_name": "input_sequence",
        "output_name": "forecast_output",
        "architecture": {
            "type": "LSTM",
            "hidden_dimension": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "batch_first": True,
            "fully_connected": [HIDDEN_DIM, 32, len(TARGET_COLUMNS)]
        },
        "preprocessing": {
            "scaler": "MinMaxScaler",
            "scaler_fitted_on": "training_partition_only",
            "feature_order": FEATURE_COLUMNS,
            "split_method": "chronological",
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO
        },
        "training": {
            "epochs_requested": EPOCHS,
            "best_epoch": int(best_epoch),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "early_stopping_patience": PATIENCE,
            "random_seed": RANDOM_SEED,
            "best_validation_loss": float(best_validation_loss)
        },
        "dataset": {
            "file": DATA_PATH.name,
            **{key: int(value) for key, value in split_info.items()}
        },
        "previous_artifacts_backup": backup_path
    }

    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")
    print("[OK] Model configuration saved.")


def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the recovered/previous artifact set before replacing it.
    backup_path = preserve_existing_artifacts()

    (
        train_loader,
        validation_loader,
        test_loader,
        scaler,
        split_info
    ) = load_and_prepare_data()

    model, best_validation_loss, best_epoch = train_model(
        train_loader,
        validation_loader,
        input_dim=len(FEATURE_COLUMNS)
    )

    # Critical: save the newly trained checkpoint and scaler BEFORE ONNX export.
    save_checkpoint_and_scaler(model, scaler)

    # Save config now too, so the training split and feature metadata survive
    # even if ONNX export fails.
    save_configuration(
        split_info, best_validation_loss, best_epoch, backup_path
    )

    # ONNX export is attempted only after durable checkpoint/scaler/config saves.
    max_onnx_difference = export_onnx_and_compare(model, test_loader, len(FEATURE_COLUMNS))

    # The held-out test partition is evaluated only after model selection is over.
    test_metrics = evaluate_test_set(model, test_loader, scaler)
    test_metrics["best_validation_loss"] = float(best_validation_loss)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["onnx_max_absolute_difference"] = max_onnx_difference
    test_metrics["split"] = "chronological_70_15_15"

    METRICS_PATH.write_text(
        json.dumps(test_metrics, indent=4), encoding="utf-8"
    )

    required = [
        CHECKPOINT_PATH, SCALER_PATH, CONFIG_PATH,
        ONNX_PATH, METRICS_PATH, PLOT_PATH
    ]
    print("\n" + "=" * 68)
    print("FINAL ARTIFACT VERIFICATION")
    print("=" * 68)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected artifact: {path}")
        print(f"[OK] {path.relative_to(PROJECT_ROOT)}")

    print("\n70/15/15 TRAINING, EXPORT AND TEST EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()

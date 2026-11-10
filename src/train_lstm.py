
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
PLOT_PATH = MODEL_DIR / "forecast_evaluation.png"

# Time-series configuration
SEQ_LENGTH = 12
PRED_HORIZON = 3

# LSTM architecture
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Training configuration
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# 3. DATASET CLASS
# ============================================================

class BuildingDataset(Dataset):

    def __init__(self, data, seq_len, pred_horizon):

        self.data = torch.from_numpy(
            np.ascontiguousarray(data)
        ).float()

        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        self.length = (
            len(data) - seq_len - pred_horizon + 1
        )

        if self.length <= 0:
            raise ValueError(
                f"Insufficient observations for windowing. "
                f"Need at least {seq_len + pred_horizon} rows."
            )

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        X = self.data[
            index:index + self.seq_len
        ]

        target_index = (
            index + self.seq_len + self.pred_horizon - 1
        )

        # First two features are CO2 and temperature.
        Y = self.data[target_index, :2]

        return X, Y


# ============================================================
# 4. LSTM MODEL
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
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_timestep = output[:, -1, :]

        return self.fc(last_timestep)


# ============================================================
# 5. LOAD AND PREPARE DATA
# ============================================================

def load_and_prepare_data():

    print("=" * 65)
    print("LOADING IAQ DATASET")
    print("=" * 65)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    df.columns = df.columns.str.strip().str.lower()

    required_features = [
        "co2",
        "temperature"
    ]

    missing = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required dataset columns are missing: {missing}"
        )

    # Preserve the exact feature order used by the model.
    feature_columns = [
        "co2",
        "temperature"
    ]

    # Humidity is included when available.
    if "humidity" in df.columns:
        feature_columns.append("humidity")

    print("Dataset:", DATA_PATH)
    print("Available rows:", len(df))
    print("Input features:", feature_columns)

    # Convert selected features to numeric.
    df[feature_columns] = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove missing and infinite observations.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=feature_columns
    )

    raw_data = df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    minimum_rows = (
        SEQ_LENGTH + PRED_HORIZON
    ) * 2

    if len(raw_data) < minimum_rows:
        raise ValueError(
            f"Only {len(raw_data)} valid rows remain. "
            f"At least {minimum_rows} are required."
        )

    # Chronological 80/20 split.
    split_index = int(
        len(raw_data) * 0.8
    )

    train_raw = raw_data[:split_index]
    test_raw = raw_data[split_index:]

    # Fit scaler ONLY on training data.
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_raw
    ).astype(np.float32)

    test_scaled = scaler.transform(
        test_raw
    ).astype(np.float32)

    train_dataset = BuildingDataset(
        train_scaled,
        SEQ_LENGTH,
        PRED_HORIZON
    )

    test_dataset = BuildingDataset(
        test_scaled,
        SEQ_LENGTH,
        PRED_HORIZON
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Valid observations:", len(raw_data))
    print("Training observations:", len(train_raw))
    print("Testing observations:", len(test_raw))
    print("Training windows:", len(train_dataset))
    print("Testing windows:", len(test_dataset))
    print("Input dimension:", len(feature_columns))

    return (
        train_loader,
        test_loader,
        scaler,
        feature_columns,
        len(raw_data),
        split_index
    )


# ============================================================
# 6. TRAINING
# ============================================================

def train_model(train_loader, test_loader, input_dim):

    model = IAQForecastLSTM(
        input_dim=input_dim
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("\n" + "=" * 65)
    print("STARTING LSTM TRAINING")
    print("=" * 65)

    print("Device:", DEVICE)

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------
        # Training phase
        # ----------------------------

        model.train()

        train_loss = 0.0

        for X, Y in train_loader:

            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(X)

            loss = criterion(
                predictions,
                Y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss += (
                loss.item() * X.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        # ----------------------------
        # Validation phase
        # ----------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for X, Y in test_loader:

                X = X.to(DEVICE)
                Y = Y.to(DEVICE)

                predictions = model(X)

                loss = criterion(
                    predictions,
                    Y
                )

                validation_loss += (
                    loss.item() * X.size(0)
                )

        validation_loss /= len(
            test_loader.dataset
        )

        scheduler.step(validation_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f} | "
            f"Learning Rate: {current_lr:.7f}",
            flush=True
        )

        # ----------------------------
        # Save best model in memory
        # ----------------------------

        if validation_loss < best_loss:

            best_loss = validation_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print("Best model updated.")

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:

                print(
                    "Early stopping triggered."
                )

                break

    if best_state is None:
        raise RuntimeError(
            "Training failed to produce a valid model."
        )

    # Restore the best model.
    model.load_state_dict(best_state)

    model.eval()

    print("\nBest validation loss:", best_loss)

    return model, best_loss


# ============================================================
# 7. SAVE TRAINED MODEL
# ============================================================

def save_checkpoint(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        CHECKPOINT_PATH
    )

    print("\nPyTorch checkpoint saved:")
    print(CHECKPOINT_PATH)


# ============================================================
# 8. SAVE SCALER
# ============================================================

def save_scaler(scaler):

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("\nFitted scaler saved:")
    print(SCALER_PATH)


# ============================================================
# 9. EXPORT ONNX MODEL
# ============================================================

def export_onnx(model, input_dim):

    print("\n" + "=" * 65)
    print("EXPORTING ONNX MODEL")
    print("=" * 65)

    model = model.to("cpu")
    model.eval()

    dummy_input = torch.randn(
        1,
        SEQ_LENGTH,
        input_dim,
        dtype=torch.float32
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input_sequence"],
        output_names=["forecast_output"],
        dynamic_axes={
            "input_sequence": {
                0: "batch_size"
            },
            "forecast_output": {
                0: "batch_size"
            }
        },
        opset_version=17,
        dynamo=False
    )

    if not ONNX_PATH.is_file():
        raise RuntimeError(
            "ONNX export did not produce the expected file."
        )

    print("ONNX model exported:")
    print(ONNX_PATH)

    print("Input name: input_sequence")
    print("Output name: forecast_output")
    print(
        "Input shape:",
        [1, SEQ_LENGTH, input_dim]
    )
    print("Output shape:", [1, 2])


# ============================================================
# 10. GENERATE MODEL CONFIGURATION
# ============================================================

def save_model_config(
    feature_columns,
    total_rows,
    split_index,
    best_loss
):

    config = {

        "model_name": "IAQForecastLSTM",

        "model_format": "ONNX",

        "model_file": "models/lstm_iaq_model.onnx",

        "checkpoint_file": "models/lstm_iaq_model.pth",

        "scaler_file": "models/scaler.pkl",

        "sequence_length": SEQ_LENGTH,

        "forecast_horizon": PRED_HORIZON,

        "feature_columns": feature_columns,

        "target_columns": [
            "co2",
            "temperature"
        ],

        "input_dimension": len(feature_columns),

        "output_dimension": 2,

        "input_name": "input_sequence",

        "output_name": "forecast_output",

        "architecture": {

            "type": "LSTM",

            "hidden_dimension": HIDDEN_DIM,

            "num_layers": NUM_LAYERS,

            "dropout": DROPOUT,

            "batch_first": True

        },

        "preprocessing": {

            "scaler": "MinMaxScaler",

            "scaler_file": "models/scaler.pkl",

            "scaler_fitted_on": "training_data_only",

            "feature_order": feature_columns,

            "train_test_split": 0.8,

            "split_method": "chronological"

        },

        "training": {

            "epochs_requested": EPOCHS,

            "batch_size": BATCH_SIZE,

            "learning_rate": LEARNING_RATE,

            "early_stopping_patience": PATIENCE,

            "random_seed": RANDOM_SEED,

            "best_validation_loss": float(best_loss)

        },

        "dataset": {

            "file": "building59_uganda_augmented.csv",

            "valid_rows": int(total_rows),

            "training_split_index": int(split_index)

        }

    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    print("\nModel configuration saved:")
    print(CONFIG_PATH)


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_loader,
    scaler,
    input_dim
):

    model.eval()

    predictions_list = []
    actuals_list = []

    with torch.no_grad():

        for X, Y in test_loader:

            predictions = model(
                X.to("cpu")
            )

            predictions_list.append(
                predictions.numpy()
            )

            actuals_list.append(
                Y.numpy()
            )

    predictions = np.vstack(
        predictions_list
    )

    actuals = np.vstack(
        actuals_list
    )

    # Restore original measurement units.
    dummy_predictions = np.zeros(
        (len(predictions), input_dim),
        dtype=np.float32
    )

    dummy_actuals = np.zeros(
        (len(actuals), input_dim),
        dtype=np.float32
    )

    dummy_predictions[:, :2] = predictions
    dummy_actuals[:, :2] = actuals

    predictions_real = scaler.inverse_transform(
        dummy_predictions
    )[:, :2]

    actuals_real = scaler.inverse_transform(
        dummy_actuals
    )[:, :2]

    mae_co2 = np.mean(
        np.abs(
            predictions_real[:, 0]
            - actuals_real[:, 0]
        )
    )

    mae_temperature = np.mean(
        np.abs(
            predictions_real[:, 1]
            - actuals_real[:, 1]
        )
    )

    print("\n" + "=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    print(
        f"CO2 MAE: {mae_co2:.4f} ppm"
    )

    print(
        f"Temperature MAE: {mae_temperature:.4f} °C"
    )

    # Evaluation plot.
    plot_length = min(
        200,
        len(actuals_real)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8)
    )

    axes[0].plot(
        actuals_real[:plot_length, 0],
        label="Actual CO2"
    )

    axes[0].plot(
        predictions_real[:plot_length, 0],
        label="Predicted CO2",
        linestyle="--"
    )

    axes[0].set_ylabel("CO2 (ppm)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        actuals_real[:plot_length, 1],
        label="Actual Temperature"
    )

    axes[1].plot(
        predictions_real[:plot_length, 1],
        label="Predicted Temperature",
        linestyle="--"
    )

    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150
    )

    plt.close(fig)

    print("\nEvaluation plot saved:")
    print(PLOT_PATH)

    return {
        "co2_mae": float(mae_co2),
        "temperature_mae": float(mae_temperature)
    }


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def main():

    set_seed()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        train_loader,
        test_loader,
        scaler,
        feature_columns,
        total_rows,
        split_index
    ) = load_and_prepare_data()

    input_dim = len(
        feature_columns
    )

    model, best_loss = train_model(
        train_loader,
        test_loader,
        input_dim
    )

    # Save all artifacts from the same training run.
    save_checkpoint(model)

    save_scaler(scaler)

    export_onnx(
        model,
        input_dim
    )

    evaluation_results = evaluate_model(
        model,
        test_loader,
        scaler,
        input_dim
    )

    save_model_config(
        feature_columns,
        total_rows,
        split_index,
        best_loss
    )

    # --------------------------------------------------------
    # FINAL ARTIFACT CHECK
    # --------------------------------------------------------

    required_artifacts = [
        CHECKPOINT_PATH,
        ONNX_PATH,
        SCALER_PATH,
        CONFIG_PATH,
        PLOT_PATH
    ]

    print("\n" + "=" * 65)
    print("FINAL ARTIFACT VERIFICATION")
    print("=" * 65)

    for artifact in required_artifacts:

        if not artifact.is_file():

            raise FileNotFoundError(
                f"Expected artifact was not created: {artifact}"
            )

        print(
            f"[OK] {artifact.relative_to(PROJECT_ROOT)}"
        )

    print("\nEvaluation summary:")

    for key, value in evaluation_results.items():
        print(f"{key}: {value:.4f}")

    print("\nLSTM TRAINING PIPELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

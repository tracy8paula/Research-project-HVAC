
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION (FLAT DIRECTORY STRUCTURE) ---
INPUT_CSV = "building59_uganda_augmented.csv"

# Saving models and plots directly to the current directory
MODEL_SAVE_PATH = "models/lstm_iaq_model.pth"
ONNX_SAVE_PATH = "models/lstm_iaq_model.onnx"
PLOT_SAVE_PATH = "forecast_evaluation.png"

SEQ_LENGTH = 12  # 60 mins lookback (assuming 5-min intervals)
PRED_HORIZON = 3  # 15 mins ahead forecast
BATCH_SIZE = 64
HIDDEN_DIM = 64
EPOCHS = 20
PATIENCE = 5  # Early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BuildingDataset(Dataset):
    """PyTorch Dataset for Time-Series Windowing."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_horizon: int):
        # Store the original data instead of creating all sliding windows.
        # Windows are extracted only when requested by the DataLoader.
        self.data = torch.from_numpy(
            np.ascontiguousarray(data)
        ).float()

        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        self.length = len(data) - seq_len - pred_horizon + 1

        if self.length <= 0:
            raise ValueError(
                f"Insufficient data for windowing. "
                f"Received {len(data)} rows, but need at least "
                f"{seq_len + pred_horizon} rows."
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):

        # Input sequence
        X = self.data[idx: idx + self.seq_len]

        # Target: CO2 & Temperature (first 2 cols)
        target_idx = idx + self.seq_len + self.pred_horizon - 1

        Y = self.data[target_idx, :2]

        return X, Y


class IAQForecastLSTM(nn.Module):
    """Stacked LSTM Architecture for CO2 & Thermal Forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(IAQForecastLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        # Take the output from the last time step
        return self.fc(out[:, -1, :])


def load_and_prepare_data(csv_path: str):
    """Loads CSV, scales train/test separately, and creates DataLoaders."""

    print(f"Loading dataset from {csv_path}...", flush=True)

    # Resolve CSV location
    if not os.path.exists(csv_path):

        current_dir = os.path.dirname(os.path.abspath(__file__))

        possible_path = os.path.join(current_dir, csv_path)
        parent_path = os.path.join(current_dir, "..", csv_path)

        if os.path.exists(possible_path):
            csv_path = possible_path

        elif os.path.exists(parent_path):
            csv_path = parent_path

        else:
            raise FileNotFoundError(
                f"Missing {csv_path}. "
                "Make sure the CSV is in the same folder."
            )

    # Load dataset
    df = pd.read_csv(csv_path)

    # Required forecasting variables
    required_features = ['co2', 'temperature']

    missing_features = [
        f for f in required_features if f not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required CSV columns: {missing_features}"
        )

    # Preserve original feature names and order.
    # Humidity remains an optional input feature.
    features = ['co2', 'temperature', 'humidity']

    features = [f for f in features if f in df.columns]

    # Convert selected features to numeric values
    df[features] = df[features].apply(
        pd.to_numeric, errors='coerce'
    )

    # Remove missing and invalid observations
    df = df.dropna(subset=features)

    df = df[
        np.isfinite(df[features].values).all(axis=1)
    ]

    raw_data = df[features].values.astype(np.float32)

    # Check dataset size
    minimum_rows = SEQ_LENGTH + PRED_HORIZON

    if len(raw_data) < minimum_rows * 2:
        raise ValueError(
            f"Insufficient observations: {len(raw_data)} rows. "
            f"At least {minimum_rows * 2} are needed for "
            "the training and testing datasets."
        )

    print(f"Valid observations: {len(raw_data)}")
    print(f"Input features: {features}")

    # Chronological Train/Test Split (80/20)
    # Split BEFORE scaling to prevent data leakage.
    split = int(len(raw_data) * 0.8)

    train_raw = raw_data[:split]
    test_raw = raw_data[split:]

    # Fit scaler ONLY on training data
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    # Create datasets
    train_dataset = BuildingDataset(
        train_scaled, SEQ_LENGTH, PRED_HORIZON
    )

    test_dataset = BuildingDataset(
        test_scaled, SEQ_LENGTH, PRED_HORIZON
    )

    # Create DataLoaders
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

    print(f"Training windows: {len(train_dataset)}")
    print(f"Testing windows: {len(test_dataset)}")

    return train_loader, test_loader, len(features), scaler


def train_and_export():
    """Main training loop with early stopping, schedulers, and ONNX export."""

    train_loader, test_loader, input_dim, scaler = (
        load_and_prepare_data(INPUT_CSV)
    )

    model = IAQForecastLSTM(
        input_dim, HIDDEN_DIM
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Removed deprecated verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    print(
        f"\n--- Starting LSTM Training on {DEVICE} ---",
        flush=True
    )

    best_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0

        total_batches = len(train_loader)

        # Display approximately 10 progress updates per epoch
        progress_interval = max(1, total_batches // 10)

        for batch_index, (bx, by) in enumerate(
            train_loader, start=1
        ):

            bx = bx.to(DEVICE)
            by = by.to(DEVICE)

            # Reset gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            predictions = model(bx)

            loss = criterion(predictions, by)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update model parameters
            optimizer.step()

            train_loss += loss.item() * bx.size(0)

            # Training progress
            if (
                batch_index % progress_interval == 0
                or batch_index == total_batches
            ):
                print(
                    f"Epoch {epoch:02d}/{EPOCHS} | "
                    f"Batch {batch_index}/{total_batches}",
                    flush=True
                )

        # Evaluation
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for bx, by in test_loader:

                bx = bx.to(DEVICE)
                by = by.to(DEVICE)

                predictions = model(bx)

                loss = criterion(predictions, by)

                val_loss += loss.item() * bx.size(0)

        # Calculate average losses
        train_loss /= len(train_loader.dataset)

        val_loss /= len(test_loader.dataset)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Learning Rate: {current_lr:.7f}",
            flush=True
        )

        # Early Stopping & Model Saving Check
        if val_loss < best_loss:

            best_loss = val_loss

            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print(
                f"Best model saved. Validation Loss: {best_loss:.5f}",
                flush=True
            )

        else:

            epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:

                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    "Restoring best weights.",
                    flush=True
                )

                break

    # Evaluation
    print("\n--- Evaluating Best Model ---", flush=True)

    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            "No trained model checkpoint was saved."
        )

    model.load_state_dict(
        torch.load(
            MODEL_SAVE_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    preds_list = []
    actuals_list = []

    with torch.no_grad():

        for bx, by in test_loader:

            predictions = model(
                bx.to(DEVICE)
            )

            preds_list.append(
                predictions.cpu().numpy()
            )

            actuals_list.append(
                by.numpy()
            )

    preds = np.vstack(preds_list)
    actuals = np.vstack(actuals_list)

    # Denormalize predictions
    dummy_preds = np.zeros(
        (len(preds), input_dim)
    )

    dummy_actuals = np.zeros(
        (len(actuals), input_dim)
    )

    dummy_preds[:, :2] = preds
    dummy_actuals[:, :2] = actuals

    preds_real = scaler.inverse_transform(
        dummy_preds
    )[:, :2]

    actuals_real = scaler.inverse_transform(
        dummy_actuals
    )[:, :2]

    # Calculate forecasting errors
    mae_co2 = float(
        np.mean(
            np.abs(
                preds_real[:, 0] - actuals_real[:, 0]
            )
        )
    )

    mae_temp = float(
        np.mean(
            np.abs(
                preds_real[:, 1] - actuals_real[:, 1]
            )
        )
    )

    print(
        f"15-Min Forecast CO2 MAE: {mae_co2:.2f} ppm"
    )

    print(
        f"15-Min Forecast Temp MAE: {mae_temp:.2f} °C"
    )

    # Plot evaluation results
    try:

        fig, ax = plt.subplots(
            2, 1, figsize=(10, 8)
        )

        plot_length = min(200, len(actuals_real))

        ax[0].plot(
            actuals_real[:plot_length, 0],
            label="Actual CO2",
            color="navy"
        )

        ax[0].plot(
            preds_real[:plot_length, 0],
            label="Predicted CO2",
            color="red",
            linestyle="--"
        )

        ax[0].legend()
        ax[0].set_ylabel("CO2 (ppm)")
        ax[0].grid(True)

        ax[1].plot(
            actuals_real[:plot_length, 1],
            label="Actual Temp",
            color="green"
        )

        ax[1].plot(
            preds_real[:plot_length, 1],
            label="Predicted Temp",
            color="orange",
            linestyle="--"
        )

        ax[1].legend()
        ax[1].set_ylabel("Temperature (°C)")
        ax[1].grid(True)

        plt.tight_layout()

        plt.savefig(PLOT_SAVE_PATH)

        plt.close(fig)

        print(
            f"Evaluation plot saved to '{PLOT_SAVE_PATH}'"
        )

    except Exception as e:

        print(f"Plotting failed: {e}")

    # Export to ONNX with dynamic batch sizing
    print("\n--- Exporting ONNX Model ---", flush=True)

    try:

        dummy_input = torch.randn(
            1, SEQ_LENGTH, input_dim
        ).to(DEVICE)

        torch.onnx.export(
            model,
            dummy_input,
            ONNX_SAVE_PATH,
            input_names=['input_sequence'],
            output_names=['forecast_output'],
            dynamic_axes={
                'input_sequence': {0: 'batch_size'},
                'forecast_output': {0: 'batch_size'}
            }
        )

        print(
            f"ONNX Model saved to {ONNX_SAVE_PATH} "
            "(Ready for INT8 Edge Quantization)"
        )

    except Exception as e:

        print(
            f"ONNX export failed: {e}\n"
            "The trained PyTorch model is still saved."
        )

    print("\n--- LSTM TRAINING PIPELINE COMPLETED ---")


if __name__ == "__main__":
    train_and_export()

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
# Assuming the CSV is in the same main folder as your script
INPUT_CSV = "building59_uganda_augmented.csv"

# Saving models and plots directly to the current directory
MODEL_SAVE_PATH = "lstm_iaq_model.pth"
ONNX_SAVE_PATH = "lstm_iaq_model.onnx"
PLOT_SAVE_PATH = "forecast_evaluation.png"

SEQ_LENGTH = 12  # 60 mins lookback (assuming 5-min intervals)
PRED_HORIZON = 3  # 15 mins ahead forecast
BATCH_SIZE = 64
HIDDEN_DIM = 64
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BuildingDataset(Dataset):
    """PyTorch Dataset for Time-Series Windowing."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_horizon: int):
        self.X, self.Y = [], []
        # Create sliding windows
        for i in range(len(data) - seq_len - pred_horizon + 1):
            self.X.append(data[i: i + seq_len])
            self.Y.append(data[i + seq_len + pred_horizon - 1, :2])  # Target: CO2 & Temp (first 2 cols)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class IAQForecastLSTM(nn.Module):
    """Stacked LSTM Architecture for CO2 & Thermal Forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(IAQForecastLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Predicting CO2 and Temperature
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output from the last time step in the sequence
        return self.fc(out[:, -1, :])


def load_and_prepare_data(csv_path: str):
    """Loads CSV, scales data, and creates DataLoaders."""
    print(f"Loading dataset from {csv_path}...")

    # Fallback to search recursively if not found in exact path
    if not os.path.exists(csv_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_path = os.path.join(current_dir, csv_path)

        # If it's a subfolder issue, let's try one directory up
        parent_path = os.path.join(current_dir, "..", csv_path)

        if os.path.exists(possible_path):
            csv_path = possible_path
        elif os.path.exists(parent_path):
            csv_path = parent_path
        else:
            raise FileNotFoundError(
                f"Missing {csv_path}. Make sure the CSV is in the same folder you are running this script from.")

    df = pd.read_csv(csv_path)
    features = ['co2', 'temperature', 'humidity']

    # Ensure columns exist before filtering
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features)

    raw_data = df[features].values

    # Use robust MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)

    # Chronological Train/Test Split (80/20)
    split = int(len(scaled_data) * 0.8)

    train_dataset = BuildingDataset(scaled_data[:split], SEQ_LENGTH, PRED_HORIZON)
    test_dataset = BuildingDataset(scaled_data[split:], SEQ_LENGTH, PRED_HORIZON)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, len(features), scaler


def train_and_export():
    """Main training loop and ONNX export for Edge AI."""
    train_loader, test_loader, input_dim, scaler = load_and_prepare_data(INPUT_CSV)
    model = IAQForecastLSTM(input_dim, HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n--- Starting LSTM Training on {DEVICE} ---")
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                val_loss += criterion(model(bx.to(DEVICE)), by.to(DEVICE)).item() * bx.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(test_loader.dataset)

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n--- Evaluating Best Model ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    preds_list, actuals_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds_list.append(model(bx.to(DEVICE)).cpu().numpy())
            actuals_list.append(by.numpy())

    preds = np.vstack(preds_list)
    actuals = np.vstack(actuals_list)

    # Denormalize predictions using the scaler (targets are cols 0 and 1)
    dummy_preds = np.zeros((len(preds), input_dim))
    dummy_actuals = np.zeros((len(actuals), input_dim))

    dummy_preds[:, :2] = preds
    dummy_actuals[:, :2] = actuals

    preds_real = scaler.inverse_transform(dummy_preds)[:, :2]
    actuals_real = scaler.inverse_transform(dummy_actuals)[:, :2]

    mae_co2 = np.mean(np.abs(preds_real[:, 0] - actuals_real[:, 0]))
    mae_temp = np.mean(np.abs(preds_real[:, 1] - actuals_real[:, 1]))

    print(f"15-Min Forecast CO2 MAE:  {mae_co2:.2f} ppm")
    print(f"15-Min Forecast Temp MAE: {mae_temp:.2f} °C")

    # Safe Plotting
    try:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(actuals_real[:200, 0], label="Actual CO2", color="navy")
        ax[0].plot(preds_real[:200, 0], label="Predicted CO2", color="red", linestyle="--")
        ax[0].legend()
        ax[0].set_ylabel("CO2 (ppm)")
        ax[0].grid(True)

        ax[1].plot(actuals_real[:200, 1], label="Actual Temp", color="green")
        ax[1].plot(preds_real[:200, 1], label="Predicted Temp", color="orange", linestyle="--")
        ax[1].legend()
        ax[1].set_ylabel("Temperature (°C)")
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH)
        print(f"Evaluation plot saved to '{PLOT_SAVE_PATH}'")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Export to ONNX for Edge deployment
    dummy_input = torch.randn(1, SEQ_LENGTH, input_dim).to(DEVICE)
    torch.onnx.export(
        model, dummy_input, ONNX_SAVE_PATH,
        input_names=['input_sequence'], output_names=['forecast_output']
    )
    print(f"ONNX Model saved to {ONNX_SAVE_PATH} (Ready for INT8 Edge Quantization)")


if __name__ == "__main__":
    train_and_export()
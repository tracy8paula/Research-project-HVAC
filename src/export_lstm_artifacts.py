
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import onnx
import onnxruntime as ort


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

CHECKPOINT = MODEL_DIR / "lstm_iaq_model.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
DATASET = ROOT / "building59_uganda_augmented.csv"

ONNX_PATH = MODEL_DIR / "lstm_iaq_model.onnx"
TEMP_ONNX_PATH = MODEL_DIR / "lstm_iaq_model_temp.onnx"
CONFIG_PATH = MODEL_DIR / "model_config.json"

# --------------------------------------------------
# MODEL CONFIGURATION
# Confirmed against the existing checkpoint
# --------------------------------------------------

SEQUENCE_LENGTH = 12
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 2

FEATURE_COLUMNS = [
    "co2",
    "temperature",
    "humidity"
]

TARGET_COLUMNS = [
    "co2",
    "temperature"
]


# --------------------------------------------------
# MODEL ARCHITECTURE
# Must match the saved checkpoint
# --------------------------------------------------

class IAQForecastLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

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

        return self.fc(out[:, -1, :])


# --------------------------------------------------
# VERIFY EXISTING FILES
# --------------------------------------------------

for path in [CHECKPOINT, SCALER_PATH, DATASET]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file is missing: {path}"
        )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Existing checkpoint and scaler found.")


# --------------------------------------------------
# LOAD AND VERIFY SCALER
# --------------------------------------------------

scaler = joblib.load(SCALER_PATH)

if scaler.n_features_in_ != len(FEATURE_COLUMNS):
    raise ValueError(
        f"Scaler expects {scaler.n_features_in_} features, "
        f"but configuration specifies {len(FEATURE_COLUMNS)}."
    )

print("Scaler verified: 3 features.")


# --------------------------------------------------
# VERIFY DATASET FEATURES
# --------------------------------------------------

df = pd.read_csv(DATASET)

missing = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Dataset is missing required features: {missing}"
    )

print("Dataset features verified.")


# --------------------------------------------------
# LOAD TRAINED CHECKPOINT
# --------------------------------------------------

model = IAQForecastLSTM(
    input_dim=len(FEATURE_COLUMNS),
    hidden_dim=HIDDEN_DIM
)

state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

print("Trained LSTM checkpoint loaded successfully.")


# --------------------------------------------------
# EXPORT ONNX
# --------------------------------------------------

dummy_input = torch.randn(
    1,
    SEQUENCE_LENGTH,
    len(FEATURE_COLUMNS),
    dtype=torch.float32
)

print("\nExporting ONNX model...")

torch.onnx.export(
    model,
    dummy_input,
    str(TEMP_ONNX_PATH),
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

print("ONNX export completed.")


# --------------------------------------------------
# VALIDATE EXPORTED ONNX FILE
# --------------------------------------------------

onnx_model = onnx.load(str(TEMP_ONNX_PATH))

onnx.checker.check_model(onnx_model)

print("ONNX structural validation passed.")


# --------------------------------------------------
# TEST ONNX RUNTIME INFERENCE
# --------------------------------------------------

session = ort.InferenceSession(
    str(TEMP_ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

test_input = np.zeros(
    (1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)),
    dtype=np.float32
)

result = session.run(
    [output_name],
    {input_name: test_input}
)[0]

if result.shape != (1, OUTPUT_DIM):
    raise ValueError(
        f"Unexpected ONNX output shape: {result.shape}"
    )

if not np.isfinite(result).all():
    raise ValueError("ONNX returned invalid numerical values.")

print("ONNX Runtime inference passed.")
print("Output shape:", result.shape)


# --------------------------------------------------
# ONLY REPLACE ONNX AFTER SUCCESSFUL VALIDATION
# --------------------------------------------------

os.replace(TEMP_ONNX_PATH, ONNX_PATH)

print("Validated ONNX model saved:", ONNX_PATH)


# --------------------------------------------------
# GENERATE MATCHING JSON CONFIGURATION
# --------------------------------------------------

config = {
    "model_name": "lstm_iaq_model",
    "model_type": "LSTM",
    "sequence_length": SEQUENCE_LENGTH,
    "forecast_horizon": 3,
    "feature_columns": FEATURE_COLUMNS,
    "target_columns": TARGET_COLUMNS,
    "input_dim": len(FEATURE_COLUMNS),
    "output_dim": OUTPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "dropout": 0.2,
    "input_name": "input_sequence",
    "output_name": "forecast_output",
    "onnx_opset": 17,
    "scaler": "scaler.pkl",
    "dataset": "building59_uganda_augmented.csv",
    "preprocessing": "MinMaxScaler"
}

with open(CONFIG_PATH, "w") as file:

    json.dump(
        config,
        file,
        indent=4
    )

print("Configuration saved:", CONFIG_PATH)


# --------------------------------------------------
# FINAL ARTIFACT CHECK
# --------------------------------------------------

print("\n--- FINAL ARTIFACT VERIFICATION ---")

for path in [
    CHECKPOINT,
    ONNX_PATH,
    SCALER_PATH,
    CONFIG_PATH
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact verification failed: {path}"
        )

    print(
        f"{path.name}: "
        f"{path.stat().st_size:,} bytes"
    )

print("\nALL EXPORT AND VALIDATION CHECKS PASSED.")

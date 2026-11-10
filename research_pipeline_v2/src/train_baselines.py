"""
Train validation-selected forecasting baselines for Research Pipeline V2.

This stage deliberately does NOT load or evaluate the held-out test partition.
The test set is reserved for one final evaluation after QPSO/model selection.

Experiments:
1. Persistence baseline.
2. LSTM using aggregate CO2 and temperature (no spatial encoding).
3. LSTM using sensor-level quantum-inspired sine/cosine features.

Windows must remain within one original continuous segment and use exact
10-minute steps. Scalers are fitted on original training rows only.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "processed"
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics"
CONFIG_PATH = ROOT / "config" / "windows.json"

TARGETS = ["co2_ppm", "temperature_c"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_partition(name: str, spatial: bool = False) -> pd.DataFrame:
    suffix = "_spatial_encoded" if spatial else ""
    path = OUTPUT / f"building59_{name}{suffix}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}. Run the preceding pipeline stages."
        )
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def fit_minmax(train_values: np.ndarray) -> dict:
    minimum = np.nanmin(train_values, axis=0)
    maximum = np.nanmax(train_values, axis=0)
    span = maximum - minimum
    span[span == 0] = 1.0
    return {"minimum": minimum, "maximum": maximum, "span": span}


def transform_minmax(values: np.ndarray, scaler: dict) -> np.ndarray:
    transformed = (values - scaler["minimum"]) / scaler["span"]
    return np.clip(transformed, 0.0, 1.0).astype(np.float32)


def inverse_minmax(values: np.ndarray, scaler: dict) -> np.ndarray:
    return values * scaler["span"] + scaler["minimum"]


def contiguous_window_starts(
    dates: pd.Series,
    segment_ids: pd.Series,
    sequence_length: int,
    horizon: int,
    interval_minutes: int,
) -> list[int]:
    """Return global start positions for fully contiguous input+target windows."""
    metadata = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "segment_id": segment_ids.astype(int),
        "position": np.arange(len(dates)),
    })
    starts: list[int] = []
    expected_delta = pd.Timedelta(minutes=interval_minutes)
    total = sequence_length + horizon

    for _, group in metadata.groupby("segment_id", sort=False):
        positions = group["position"].to_numpy()
        group_dates = group["date"].reset_index(drop=True)
        if len(positions) < total:
            continue
        deltas_ok = group_dates.diff().iloc[1:].eq(expected_delta).to_numpy()
        run_start = 0
        for j in range(1, len(group_dates)):
            if not deltas_ok[j - 1]:
                run_positions = positions[run_start:j]
                if len(run_positions) >= total:
                    starts.extend(run_positions[k] for k in range(len(run_positions) - total + 1))
                run_start = j
        run_positions = positions[run_start:]
        if len(run_positions) >= total:
            starts.extend(run_positions[k] for k in range(len(run_positions) - total + 1))

    return starts


class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        starts: list[int],
        sequence_length: int,
        horizon: int,
    ):
        self.features = features
        self.targets = targets
        self.starts = starts
        self.sequence_length = sequence_length
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        middle = start + self.sequence_length
        end = middle + self.horizon
        x = self.features[start:middle]
        y = self.targets[middle:end]
        return torch.from_numpy(x), torch.from_numpy(y)


class ForecastLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dimension: int,
        layers: int,
        dropout: float,
        horizon: int,
        output_size: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dimension,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, horizon * output_size),
        )
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        final_state = sequence_output[:, -1, :]
        prediction = self.head(final_state)
        return prediction.view(-1, self.horizon, self.output_size)


def build_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    feature_scaler: dict | None,
    target_scaler: dict,
    sequence_length: int,
    horizon: int,
    interval_minutes: int,
) -> tuple[WindowDataset, np.ndarray, np.ndarray]:
    raw_x = frame[feature_columns].to_numpy(dtype=np.float32)
    raw_y = frame[TARGETS].to_numpy(dtype=np.float32)

    if feature_scaler is not None:
        x = transform_minmax(raw_x, feature_scaler)
    else:
        # Spatial sensor channels are already bounded sine/cosine features.
        x = np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y = transform_minmax(raw_y, target_scaler)
    starts = contiguous_window_starts(
        frame["date"],
        frame["segment_id"],
        sequence_length,
        horizon,
        interval_minutes,
    )
    dataset = WindowDataset(x, y, starts, sequence_length, horizon)
    return dataset, x, y


def calculate_metrics(
    prediction_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    target_scaler: dict,
) -> dict:
    prediction = inverse_minmax(prediction_scaled, target_scaler)
    truth = inverse_minmax(truth_scaled, target_scaler)
    result = {}
    for index, name in enumerate(TARGETS):
        error = prediction[:, :, index] - truth[:, :, index]
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        }
    result["macro_average_mae"] = float(
        np.mean([result[name]["mae"] for name in TARGETS])
    )
    return result


@torch.no_grad()
def predict_model(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, truths = [], []
    for x, y in loader:
        x = x.to(device)
        prediction = model(x).cpu().numpy()
        predictions.append(prediction)
        truths.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(truths)


def evaluate_persistence(
    dataset: WindowDataset,
    target_scaler: dict,
    batch_size: int,
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, truths = [], []
    for x, y in loader:
        # Persistence repeats the final observed target for all forecast steps.
        last_value = x[:, -1, :2].numpy()
        predictions.append(np.repeat(last_value[:, None, :], y.shape[1], axis=1))
        truths.append(y.numpy())
    return calculate_metrics(
        np.concatenate(predictions),
        np.concatenate(truths),
        target_scaler,
    )


def train_one_model(
    model_name: str,
    train_dataset: WindowDataset,
    validation_dataset: WindowDataset,
    input_size: int,
    config: dict,
    target_scaler: dict,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    forecast_cfg = config["forecasting"]
    seed_everything(int(config["project"]["random_seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(forecast_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(forecast_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = ForecastLSTM(
        input_size=input_size,
        hidden_dimension=int(forecast_cfg["hidden_dimension"]),
        layers=int(forecast_cfg["lstm_layers"]),
        dropout=float(forecast_cfg["dropout"]),
        horizon=int(forecast_cfg["forecast_horizon_steps"]),
        output_size=len(TARGETS),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(forecast_cfg["learning_rate"]),
    )
    loss_function = nn.MSELoss()
    best_validation = float("inf")
    best_state = None
    patience = int(forecast_cfg["early_stopping_patience"])
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(forecast_cfg["epochs"]) + 1):
        model.train()
        training_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(forecast_cfg["gradient_clip_norm"]),
            )
            optimizer.step()
            training_losses.append(float(loss.item()))

        validation_prediction, validation_truth = predict_model(
            model, validation_loader, device
        )
        validation_metrics = calculate_metrics(
            validation_prediction, validation_truth, target_scaler
        )
        validation_loss = validation_metrics["macro_average_mae"]
        epoch_record = {
            "epoch": epoch,
            "training_mse_scaled": float(np.mean(training_losses)),
            "validation_macro_mae_original_units": validation_loss,
            "validation_co2_mae_ppm": validation_metrics["co2_ppm"]["mae"],
            "validation_temperature_mae_c": validation_metrics["temperature_c"]["mae"],
        }
        history.append(epoch_record)

        print(
            f"{model_name} | epoch {epoch:02d} | "
            f"train MSE={epoch_record['training_mse_scaled']:.6f} | "
            f"validation MAE={validation_loss:.4f}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"{model_name}: early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    final_prediction, final_truth = predict_model(model, validation_loader, device)
    metrics = calculate_metrics(final_prediction, final_truth, target_scaler)

    model_path = MODELS / f"{model_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_dimension": int(forecast_cfg["hidden_dimension"]),
        "lstm_layers": int(forecast_cfg["lstm_layers"]),
        "dropout": float(forecast_cfg["dropout"]),
        "sequence_length": int(forecast_cfg["sequence_length"]),
        "forecast_horizon_steps": int(forecast_cfg["forecast_horizon_steps"]),
        "target_order": TARGETS,
        "selected_by": "validation_macro_average_MAE",
    }, model_path)

    return {
        "validation_metrics": metrics,
        "best_validation_macro_mae": best_validation,
        "checkpoint": str(model_path),
        "epochs_completed": len(history),
    }, history


def main() -> None:
    config = load_config()
    MODELS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    seed_everything(int(config["project"]["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequence_length = int(config["forecasting"]["sequence_length"])
    horizon = int(config["forecasting"]["forecast_horizon_steps"])
    interval = int(config["forecasting"]["sampling_interval_minutes"])
    batch_size = int(config["forecasting"]["batch_size"])

    print("=" * 72)
    print("RESEARCH PIPELINE V2 - PERSISTENCE AND LSTM BASELINES")
    print("=" * 72)
    print("Device:", device)
    print("Sequence length:", sequence_length)
    print("Forecast horizon:", horizon, "steps")
    print("Target order:", TARGETS)
    print("IMPORTANT: held-out test partition will NOT be loaded or evaluated.")

    train = load_partition("train")
    validation = load_partition("validation")

    target_scaler = fit_minmax(train[TARGETS].to_numpy(dtype=np.float32))

    # Non-spatial baseline: only the two aggregate measurements.
    nonspatial_features = TARGETS.copy()
    train_nonspatial, train_x_nonspatial, _ = build_dataset(
        train, nonspatial_features,
        fit_minmax(train[nonspatial_features].to_numpy(dtype=np.float32)),
        target_scaler, sequence_length, horizon, interval
    )
    nonspatial_scaler = fit_minmax(
        train[nonspatial_features].to_numpy(dtype=np.float32)
    )
    # Rebuild using the single persisted scaler object.
    train_nonspatial, _, _ = build_dataset(
        train, nonspatial_features, nonspatial_scaler,
        target_scaler, sequence_length, horizon, interval
    )
    validation_nonspatial, _, _ = build_dataset(
        validation, nonspatial_features, nonspatial_scaler,
        target_scaler, sequence_length, horizon, interval
    )

    # Spatial model: sensor-level encoded channels only; no raw aggregate
    # columns are included in its input tensor.
    train_spatial_frame = load_partition("train", spatial=True)
    validation_spatial_frame = load_partition("validation", spatial=True)
    spatial_features = [
        column for column in train_spatial_frame.columns
        if column.startswith("co2__") or column.startswith("temperature__")
    ]
    if not spatial_features:
        raise ValueError("No spatial sine/cosine feature columns were found.")

    train_spatial, _, _ = build_dataset(
        train_spatial_frame, spatial_features, None,
        target_scaler, sequence_length, horizon, interval
    )
    validation_spatial, _, _ = build_dataset(
        validation_spatial_frame, spatial_features, None,
        target_scaler, sequence_length, horizon, interval
    )

    for label, dataset in [
        ("training non-spatial", train_nonspatial),
        ("validation non-spatial", validation_nonspatial),
        ("training spatial", train_spatial),
        ("validation spatial", validation_spatial),
    ]:
        if len(dataset) == 0:
            raise ValueError(
                f"No valid {label} windows. Check continuous segments and 10-minute timestamps."
            )

    print("\nVALID WINDOW COUNTS")
    print("Training:", len(train_nonspatial))
    print("Validation:", len(validation_nonspatial))
    print("Spatial training:", len(train_spatial))
    print("Spatial validation:", len(validation_spatial))

    # Persistence baseline is computed on validation only.
    persistence_metrics = evaluate_persistence(
        validation_nonspatial, target_scaler, batch_size
    )
    print("\nPERSISTENCE VALIDATION METRICS")
    print(json.dumps(persistence_metrics, indent=2))

    nonspatial_result, nonspatial_history = train_one_model(
        "lstm_without_spatial",
        train_nonspatial,
        validation_nonspatial,
        len(nonspatial_features),
        config,
        target_scaler,
        device,
    )

    spatial_result, spatial_history = train_one_model(
        "lstm_with_spatial",
        train_spatial,
        validation_spatial,
        len(spatial_features),
        config,
        target_scaler,
        device,
    )

    summary = {
        "evaluation_partition": "validation",
        "held_out_test_loaded": False,
        "held_out_test_evaluated": False,
        "target_order": TARGETS,
        "sequence_length": sequence_length,
        "forecast_horizon_steps": horizon,
        "target_scaler_fit_partition": "training_only",
        "input_scaler_fit_partition": "training_only_for_nonspatial",
        "spatial_features_pre_scaled_by": "encode_spatial_features.py_training_only_scalers",
        "window_rule": (
            "All input and target rows must share a segment_id and have exact "
            "10-minute timestamp differences."
        ),
        "persistence": {"validation_metrics": persistence_metrics},
        "lstm_without_spatial": nonspatial_result,
        "lstm_with_spatial": spatial_result,
        "selected_candidate": min(
            [
                ("lstm_without_spatial", nonspatial_result["best_validation_macro_mae"]),
                ("lstm_with_spatial", spatial_result["best_validation_macro_mae"]),
            ],
            key=lambda item: item[1],
        )[0],
    }

    with (METRICS / "baseline_validation_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    pd.DataFrame(nonspatial_history).to_csv(
        METRICS / "lstm_without_spatial_training_history.csv", index=False
    )
    pd.DataFrame(spatial_history).to_csv(
        METRICS / "lstm_with_spatial_training_history.csv", index=False
    )

    scaler_export = {
        "target_order": TARGETS,
        "minimum": target_scaler["minimum"].tolist(),
        "maximum": target_scaler["maximum"].tolist(),
        "span": target_scaler["span"].tolist(),
        "fit_partition": "training_only",
        "nonspatial_input_order": nonspatial_features,
        "nonspatial_input_minimum": nonspatial_scaler["minimum"].tolist(),
        "nonspatial_input_maximum": nonspatial_scaler["maximum"].tolist(),
        "nonspatial_input_span": nonspatial_scaler["span"].tolist(),
        "spatial_input_order": spatial_features,
    }
    with (MODELS / "baseline_scalers.json").open("w", encoding="utf-8") as handle:
        json.dump(scaler_export, handle, indent=2)

    print("\nVALIDATION MODEL COMPARISON")
    print("Persistence MAE:", persistence_metrics["macro_average_mae"])
    print(
        "LSTM without spatial MAE:",
        nonspatial_result["validation_metrics"]["macro_average_mae"]
    )
    print(
        "LSTM with spatial MAE:",
        spatial_result["validation_metrics"]["macro_average_mae"]
    )
    print("\nSelected baseline candidate:", summary["selected_candidate"])
    print("\nSaved model checkpoints:", MODELS)
    print("Saved validation metrics:", METRICS / "baseline_validation_results.json")
    print("\nHELD-OUT TEST DATA WAS NOT USED.")


if __name__ == "__main__":
    main()

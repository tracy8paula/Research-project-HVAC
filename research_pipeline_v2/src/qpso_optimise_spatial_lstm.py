"""
Quantum-behaved Particle Swarm Optimisation (QPSO) for spatial LSTM tuning.

Fitness is measured on the validation partition only. A deterministic subset
of training windows is used for inexpensive particle fitness training; the
full training set is used to refit the selected configuration afterwards.
The held-out test partition is never loaded in this script.

Search variables:
- hidden_dimension: integer [16,  64]
- learning_rate: log-scaled [0.0001, 0.005]
- dropout: [0.0, 0.4]

This is a classical QPSO optimisation algorithm, not execution on quantum
hardware and not evidence of quantum advantage.
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from train_baselines import (
    ForecastLSTM,
    TARGETS,
    build_dataset,
    fit_minmax,
    inverse_minmax,
    load_config,
    load_partition,
    predict_model,
    seed_everything,
    transform_minmax,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "processed"
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics"


def clamp_particle(position: np.ndarray) -> np.ndarray:
    bounds = np.array([
        [16.0, 64.0],
        [math.log10(0.0001), math.log10(0.005)],
        [0.0, 0.4],
    ])
    return np.clip(position, bounds[:, 0], bounds[:, 1])


def decode_particle(position: np.ndarray) -> dict:
    position = clamp_particle(position)
    return {
        "hidden_dimension": int(np.clip(round(position[0]), 16, 64)),
        "learning_rate": float(10 ** position[1]),
        "dropout": float(position[2]),
    }


def metric_from_scaled(
    prediction_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    target_scaler: dict,
) -> float:
    prediction = inverse_minmax(prediction_scaled, target_scaler)
    truth = inverse_minmax(truth_scaled, target_scaler)
    errors = []
    for i in range(len(TARGETS)):
        errors.append(float(np.mean(np.abs(prediction[:, :, i] - truth[:, :, i]))))
    return float(np.mean(errors))


def make_model(input_size, params, config, device):
    forecast = config["forecasting"]
    return ForecastLSTM(
        input_size=input_size,
        hidden_dimension=params["hidden_dimension"],
        layers=int(forecast["lstm_layers"]),
        dropout=params["dropout"],
        horizon=int(forecast["forecast_horizon_steps"]),
        output_size=len(TARGETS),
    ).to(device)


def train_candidate(
    params,
    train_loader,
    validation_loader,
    input_size,
    config,
    target_scaler,
    device,
    epochs,
    seed,
):
    seed_everything(seed)
    model = make_model(input_size, params, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_function = nn.MSELoss()
    best_score = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["forecasting"]["gradient_clip_norm"]),
            )
            optimizer.step()

        prediction_scaled, truth_scaled = predict_model(
            model, validation_loader, device
        )
        score = metric_from_scaled(prediction_scaled, truth_scaled, target_scaler)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("QPSO candidate failed to produce a checkpoint.")
    return best_score, best_state


def make_loaders(train_dataset, validation_dataset, config, seed):
    batch_size = int(config["forecasting"]["batch_size"])
    # Fitness uses a deterministic subset to keep the 25-particle search
    # practical on a laptop. Refit below uses every valid training window.
    rng = np.random.default_rng(seed)
    maximum_fitness_windows = min(6000, len(train_dataset))
    selected = np.sort(
        rng.choice(len(train_dataset), size=maximum_fitness_windows, replace=False)
    ).tolist()
    fitness_train = Subset(train_dataset, selected)

    train_loader = DataLoader(
        fitness_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    full_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, full_train_loader, validation_loader, maximum_fitness_windows


def qpso_search(
    train_loader,
    validation_loader,
    input_size,
    config,
    target_scaler,
    device,
):
    qpso_cfg = config["qpso"]
    particles = int(qpso_cfg["particles"])
    iterations = int(qpso_cfg["iterations"])
    fitness_epochs = int(qpso_cfg["fitness_epochs"])
    seed = int(qpso_cfg["seed"])
    rng = np.random.default_rng(seed)
    random.seed(seed)

    lower = np.array([16.0, math.log10(0.0001), 0.0])
    upper = np.array([64.0, math.log10(0.005), 0.4])

    positions = rng.uniform(lower, upper, size=(particles, 3))
    velocities = np.zeros_like(positions)
    personal_positions = positions.copy()
    personal_scores = np.full(particles, np.inf)
    global_position = None
    global_score = np.inf
    history = []

    for iteration in range(iterations):
        print(f"\nQPSO iteration {iteration + 1}/{iterations}")
        # Evaluate every particle with the same deterministic seed schedule.
        for particle_index in range(particles):
            params = decode_particle(positions[particle_index])
            score, _ = train_candidate(
                params,
                train_loader,
                validation_loader,
                input_size,
                config,
                target_scaler,
                device,
                fitness_epochs,
                seed + iteration * particles + particle_index,
            )

            if score < personal_scores[particle_index]:
                personal_scores[particle_index] = score
                personal_positions[particle_index] = positions[particle_index].copy()

            if score < global_score:
                global_score = score
                global_position = positions[particle_index].copy()

            print(
                f" particle {particle_index + 1}: "
                f"hidden={params['hidden_dimension']}, "
                f"lr={params['learning_rate']:.6f}, "
                f"dropout={params['dropout']:.3f}, "
                f"validation MAE={score:.5f}"
            )

            history.append({
                "iteration": iteration + 1,
                "particle": particle_index + 1,
                **params,
                "validation_macro_mae": score,
                "global_best_after_evaluation": global_score,
            })

        # Quantum-behaved update. Alpha decreases linearly from 1.0 to 0.5.
        mbest = personal_positions.mean(axis=0)
        alpha = 1.0 - 0.5 * (iteration / max(iterations - 1, 1))

        for particle_index in range(particles):
            phi = rng.random(3)
            attractor = (
                phi * personal_positions[particle_index]
                + (1.0 - phi) * global_position
            )
            u = np.maximum(rng.random(3), 1e-12)
            direction = np.where(rng.random(3) < 0.5, -1.0, 1.0)
            positions[particle_index] = (
                attractor
                + direction
                * alpha
                * np.abs(mbest - positions[particle_index])
                * np.log(1.0 / u)
            )
            positions[particle_index] = clamp_particle(positions[particle_index])

        print(f" Global best validation MAE: {global_score:.5f}")

    return decode_particle(global_position), global_score, history


def main() -> None:
    config = load_config()
    if not config["qpso"].get("enabled", False):
        raise RuntimeError("QPSO is disabled in windows.json.")

    MODELS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["random_seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequence_length = int(config["forecasting"]["sequence_length"])
    horizon = int(config["forecasting"]["forecast_horizon_steps"])
    interval = int(config["forecasting"]["sampling_interval_minutes"])

    print("=" * 72)
    print("RESEARCH PIPELINE V2 - QPSO SPATIAL LSTM OPTIMISATION")
    print("=" * 72)
    print("Device:", device)
    print("Particles:", config["qpso"]["particles"])
    print("Iterations:", config["qpso"]["iterations"])
    print("Fitness epochs per candidate:", config["qpso"]["fitness_epochs"])
    print("Validation is the fitness partition.")
    print("Held-out test partition will NOT be loaded.")

    train_base = load_partition("train")
    validation_base = load_partition("validation")
    train_spatial = load_partition("train", spatial=True)
    validation_spatial = load_partition("validation", spatial=True)

    feature_columns = [
        column for column in train_spatial.columns
        if column.startswith("co2__") or column.startswith("temperature__")
    ]
    if not feature_columns:
        raise ValueError("Spatial feature columns not found.")

    target_scaler = fit_minmax(train_base[TARGETS].to_numpy(dtype=np.float32))
    train_dataset, _, _ = build_dataset(
        train_spatial, feature_columns, None, target_scaler,
        sequence_length, horizon, interval
    )
    validation_dataset, _, _ = build_dataset(
        validation_spatial, feature_columns, None, target_scaler,
        sequence_length, horizon, interval
    )
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("No valid contiguous windows available for QPSO.")

    fitness_loader, full_train_loader, validation_loader, fitness_count = make_loaders(
        train_dataset, validation_dataset, config, seed
    )
    print(f"Fitness training windows per candidate: {fitness_count:,}")
    print(f"Full training windows for final refit: {len(train_dataset):,}")
    print(f"Validation windows: {len(validation_dataset):,}")

    best_params, best_fitness, history = qpso_search(
        fitness_loader,
        validation_loader,
        len(feature_columns),
        config,
        target_scaler,
        device,
    )

    # Refit the selected hyperparameters on ALL valid training windows.
    print("\nREFITTING SELECTED QPSO MODEL ON ALL TRAINING WINDOWS")
    full_epochs = int(config["forecasting"]["epochs"])
    best_score, best_state = train_candidate(
        best_params,
        full_train_loader,
        validation_loader,
        len(feature_columns),
        config,
        target_scaler,
        device,
        full_epochs,
        seed + 10000,
    )

    final_model = make_model(len(feature_columns), best_params, config, device)
    final_model.load_state_dict(best_state)
    checkpoint_path = MODELS / "qpso_spatial_lstm.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "input_size": len(feature_columns),
        "hidden_dimension": best_params["hidden_dimension"],
        "lstm_layers": int(config["forecasting"]["lstm_layers"]),
        "dropout": best_params["dropout"],
        "learning_rate": best_params["learning_rate"],
        "sequence_length": sequence_length,
        "forecast_horizon_steps": horizon,
        "target_order": TARGETS,
        "selected_by": "QPSO_validation_macro_average_MAE",
        "held_out_test_used": False,
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(METRICS / "qpso_search_history.csv", index=False)
    summary = {
        "algorithm": "Quantum-behaved Particle Swarm Optimisation (QPSO)",
        "implementation_hardware": "classical CPU/GPU via PyTorch",
        "quantum_hardware_used": False,
        "quantum_advantage_claimed": False,
        "fitness_partition": "validation",
        "fitness_training_subset_windows": fitness_count,
        "full_refit_training_windows": len(train_dataset),
        "held_out_test_loaded": False,
        "held_out_test_evaluated": False,
        "particles": int(config["qpso"]["particles"]),
        "iterations": int(config["qpso"]["iterations"]),
        "fitness_epochs_per_particle": int(config["qpso"]["fitness_epochs"]),
        "selected_hyperparameters": best_params,
        "best_particle_search_validation_mae": best_fitness,
        "full_training_refit_validation_mae": best_score,
        "checkpoint": str(checkpoint_path),
        "search_history": str(METRICS / "qpso_search_history.csv"),
    }
    with (METRICS / "qpso_optimisation_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\nQPSO SELECTED HYPERPARAMETERS")
    print(json.dumps(best_params, indent=2))
    print("Best particle fitness MAE:", best_fitness)
    print("Full-refit validation MAE:", best_score)
    print("\nSaved:", checkpoint_path)
    print("Saved:", METRICS / "qpso_optimisation_results.json")
    print("Saved:", METRICS / "qpso_search_history.csv")
    print("\nHELD-OUT TEST DATA WAS NOT USED.")


if __name__ == "__main__":
    main()

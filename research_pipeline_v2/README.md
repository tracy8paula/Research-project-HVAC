# Short-Horizon Indoor CO2 and Temperature Forecasting for HVAC Decision Support

Research code for an experimental forecasting study using Building 59 indoor-environment observations. It explores methods that may inform predictive ventilation/HVAC decisions in Ugandan building contexts.

> **Status:** experimental prototype, not a validated Ugandan deployment. Synthetic scenarios are not Ugandan measurements. The control simulation is not a commissioned HVAC controller or proof of energy savings.

## Research question
Can short-horizon models forecast indoor CO2 and temperature, and can these forecasts provide useful evidence for future ventilation/HVAC decision support in building contexts relevant to Uganda?

## Implemented
- Two-stream CO2 and temperature data preparation, QC and timestamp alignment.
- Chronological 70/15/15 split and continuous-segment-aware forecast windows.
- Training-only scaling and validation-based model selection.
- Persistence and aggregate LSTM baselines.
- Sensor-level sine/cosine spatial feature encoding using a **virtual assumed grid**.
- QPSO hyperparameter search (a classical quantum-behaved optimisation heuristic; no quantum computer or quantum advantage).
- Five synthetic Uganda-context scenarios generated from training-partition information.
- Scenario sensitivity and occupancy/ventilation interaction analyses.
- Offline rule-based ventilation simulation with a dimensionless fan-energy proxy.
- Read-only data integrity/leakage audit and CPU-compatible Docker packaging.
- Static dashboard of saved outputs (not live model inference).

## Current experiment definition
- Inputs: 12 ten-minute observations (120 minutes of history).
- Targets: indoor CO2 (ppm) and temperature (°C).
- Forecast: three ten-minute steps = **30 minutes ahead**, not five minutes.
- Chronological split: 70/15/15.
- Scalers fit on training only; QPSO fitness uses validation only.
- The historical test partition was inspected before a later validation-objective correction. It must not be represented as a fresh blind test.

Recorded preparation run: 11 CO2 and 16 temperature sensors, 54,893 aligned ten-minute rows, four continuous segments; partitions 38,425/8,234/8,234 rows. The recorded audit reported 32/32 checks passing. These historical values must be rechecked by rerunning against the exact authorised inputs; generated outputs are not committed.

## Run locally
Place the authorised source files at `data/raw/zone_co2.csv` and `data/raw/zone_temp_interior.csv`. Do not commit raw data without permission.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1
python src/check_environment.py
python src/run_pipeline.py
```

Run fast invariant tests:
```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

Docker:
```bash
docker compose build
docker compose run --rm research-pipeline
```

## Scientific limitations
1. Source measurements are Building 59 data, not Ugandan building measurements.
2. Synthetic Uganda-context scenarios use assumed occupancy, room volume, outdoor CO2 and air-change rates.
3. Spatial coordinates are virtual assumptions, not measured sensor locations or floor-plan geometry.
4. Fan energy is represented by a dimensionless cubic proxy, not kWh or a calibrated HVAC power model.
5. Current ventilation simulation is rule-based, not learned predictive control or physical equipment testing.
6. The control simulation does not model thermal dynamics, comfort, heating/cooling loads or whole-building energy.
7. Docker CPU execution is packaging, not embedded-device deployment or edge hardware benchmarking.
8. Historical test outputs are pre-correction exploratory evidence, not a fresh blind test.

See [Research Status](docs/RESEARCH_STATUS_AND_LIMITATIONS.md), [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md) and [Reproducibility Checklist](docs/REPRODUCIBILITY_CHECKLIST.md).

## Defensible current claim
A leakage-aware experimental pipeline has been implemented to compare short-horizon indoor CO2/temperature forecasting methods on Building 59 observations and explore explicitly synthetic Uganda-context scenarios. Local Ugandan building performance, real energy savings, health outcomes, quantum advantage and physical HVAC deployment have **not** been established.

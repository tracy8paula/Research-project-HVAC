# Research Pipeline V2

This is a separate research implementation for:

**A Quantum-Inspired Edge AI Framework for Energy-Aware Indoor Air Quality and HVAC Control in Ugandan Buildings.**

It is isolated under `research_pipeline_v2/` on branch `research-pipeline-v2`. Existing project files and earlier commits are not replaced.

## Source data

The pipeline uses only:

1. `data/raw/zone_co2.csv`
2. `data/raw/zone_temp_interior.csv`

Humidity is excluded. No other Building 59 folder or CSV is required.

Do not upload the raw research CSVs to GitHub. Keep them locally on the Windows PC.

## Windows preparation

1. Clone or switch to the `research-pipeline-v2` branch.
2. Copy the two CSV files into `research_pipeline_v2/data/raw/`.
3. Open PowerShell in the `research_pipeline_v2` directory.
4. Activate the existing Conda environment:

```powershell
conda activate ML_CLASS
```

5. Run the non-mutating environment check:

```powershell
python .\src\check_environment.py
```

6. Run the Windows folder and input-file check:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

7. If the environment check passes and both CSVs are found, prepare the two streams:

```powershell
python .\src\prepare_data.py
```

The setup and environment-check scripts do not install, remove or upgrade packages. Do not install dependencies blindly; first inspect the environment-check output.

## Data preparation

The current preparation script:
- validates timestamp columns and discovers sensor columns from the actual naming patterns;
- converts invalid timestamps and out-of-range sensor readings into missing values;
- uses per-sensor median aggregation for duplicate timestamps;
- resamples the CO₂ and temperature streams independently to 10-minute bins;
- interpolates only short internal gaps (up to two bins);
- aligns the two modalities by timestamp;
- retains sensor-level panels for the later spatial feature experiment;
- marks long discontinuities as separate time segments so forecast windows cannot bridge them;
- saves a JSON quality report.

The broad QC bounds currently configured are 250–2000 ppm for CO₂ and 10–40 °C for indoor temperature. These are data-quality filters, not health or comfort thresholds. The report records how many values are removed by these bounds so the decision can be reviewed.

## Methodological safeguards for the complete study

- Chronological 70/15/15 train/validation/test partitions.
- Scalers and learned transformations fitted on training data only.
- Uganda-context scenarios are synthetic modelling scenarios, never measured Ugandan observations.
- Uganda scenarios are generated from training-partition information and evaluated separately.
- No humidity, PM2.5 or PM10 values are invented.
- The virtual spatial layout is a documented assumption, not a measured floor plan or verified sensor map.
- QPSO uses validation performance only; the held-out test partition is reserved for final evaluation.
- Edge latency results from the Windows PC are labelled as local CPU benchmarks, not physical embedded-device measurements.

## Current implementation status

Available now:
- Windows configuration;
- input/output folder setup;
- non-mutating environment diagnostic;
- two-file timestamp-safe data preparation and quality reporting.

Still to be implemented and validated before the full research run:
- training-only spatial sine/cosine feature construction using a documented virtual sensor layout;
- Uganda-context mass-balance scenario generation;
- chronological split and leakage-safe scaling integration;
- persistence and LSTM baselines;
- QPSO hyperparameter search;
- ablation experiments, final held-out test evaluation and edge benchmark.

Do not treat this branch as the complete research pipeline until those remaining modules and their integration tests are in place.

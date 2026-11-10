# Research Pipeline V2

This is a separate, self-contained research implementation for the project:

**A Quantum-Inspired Edge AI Framework for Energy-Aware Indoor Air Quality and HVAC Control in Ugandan Buildings.**

It is deliberately isolated under `research_pipeline_v2/`. Existing project files and earlier commits are not replaced.

## Source data

The pipeline is designed to use only:

1. `data/raw/zone_co2.csv`
2. `data/raw/zone_temp_interior.csv`

Humidity is excluded. No other Building 59 folder or CSV is required.

Do not upload the raw research CSVs to GitHub. Keep them locally on the Windows PC.

## Windows preparation

1. Clone or pull the `research-pipeline-v2` branch of the repository.
2. Copy the two CSV files into `research_pipeline_v2/data/raw/`.
3. Open PowerShell in the `research_pipeline_v2` directory.
4. Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

The setup script only checks the configuration, creates output directories and verifies the two input files. It does not install or upgrade packages.

The intended existing Conda environment is `ML_CLASS`. Package installation and environment checks will be specified separately so the existing working environment is not unexpectedly changed.

## Methodological safeguards

- CO₂ and temperature are aligned by timestamp at a common 10-minute interval.
- Duplicate timestamps are handled explicitly; data tables are never vertically concatenated.
- Missing and implausible sensor values are recorded in quality-control outputs.
- Chronological 70/15/15 train/validation/test partitions are used.
- Scalers and learned transformations are fitted on training data only.
- Uganda-context scenarios are synthetic modelling scenarios, never described as measured Ugandan observations.
- Uganda scenarios are generated from training-partition information and evaluated separately.
- No humidity, PM2.5 or PM10 values are invented.
- The virtual spatial layout is a documented assumption, not a measured floor plan or verified sensor map.
- QPSO uses validation performance only; the held-out test partition is reserved for final evaluation.
- Edge latency results from the Windows PC are labelled as local CPU benchmarks, not physical embedded-device measurements.

## Current status

This branch currently contains the Windows configuration and project layout. The processing, feature engineering, QPSO, model training, evaluation and benchmark scripts will be added as the implementation is completed and validated against the two source files.

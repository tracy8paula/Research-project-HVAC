# Next Experiment Protocol
**Status:** protocol proposal, not a record of completed experiments.

## Objective
Test whether sensor-level spatial encoding and QPSO hyperparameter selection improve short-horizon CO2 and temperature forecasts over persistence and aggregate LSTM, then assess potential decision-support value.

## Forecast definition
Current configuration: 10-minute sampling, 12 input observations (120 minutes), three forecast steps (30 minutes). Do not call this a five-minute forecast. A five-minute study requires a separately justified resampling design and rerun.

## Data protocol
1. Confirm dataset citation, licence/permission, collection period, time zone, sensor units and calibration metadata.
2. Record SHA-256 checksums of both original CSVs.
3. Preserve raw inputs unchanged.
4. Document timestamp exclusions, out-of-range values, duplicates, interpolation and long gaps.
5. Use chronological 70/15/15 partitions; windows cannot cross segments.
6. Fit scalers/transforms on training only.

## Models
- Persistence.
- Aggregate-feature LSTM.
- Spatial-feature LSTM.
- QPSO-selected spatial LSTM.
Use identical splits, target definitions and evaluation code. Compare QPSO with a non-QPSO tuning baseline under a comparable search budget where feasible. Report parameter counts, training time and inference latency.

## Repeated experiments and metrics
Predeclare at least five training seeds. Report mean, standard deviation and suitable 95% confidence intervals. Report MAE/RMSE in native units separately for CO2 and temperature and for each forecast lead; normalized error definition, bias, residual diagnostics and persistence-relative skill. Do not report only the best run.

## Test-set policy
The historical test was already inspected before a validation objective correction. Treat it as exploratory pre-correction evidence. Do not call it a fresh blind test. Freeze the protocol and use a new independent building or future period as the preferred final evaluation.

## Synthetic Uganda scenarios
Publish every assumed parameter and plausible range with sources. Label all outputs synthetic. Do not count scenario rows as measured observations or local validation. Run sensitivity and factorial analyses.

## Control and energy
Current control simulation is rule-based and CO2-only; its cubic fan proxy is dimensionless. For energy claims, use measured electrical energy or a calibrated equipment power model. Evaluate fixed-low/reference/high and rule-based baselines against a forecast-informed policy only after its design is specified. Report CO2 and temperature/comfort constraints, energy in kWh or a calibrated quantity, switching and response delay. Explicitly state simulation versus field evidence.

## Reproducibility package
Archive commit SHA, config, input checksums, package versions, hardware/OS, quality report, partitions, seeds, training history, metrics, residual plots, scenario assumptions/results, control assumptions/results, audit, logs and dashboard version.

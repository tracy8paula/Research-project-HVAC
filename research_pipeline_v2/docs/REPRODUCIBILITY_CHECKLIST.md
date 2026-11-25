# Reproducibility Checklist

## Provenance
- [ ] Dataset citation and licence/permission confirmed.
- [ ] Collection dates, timezone, sensor IDs, units and calibration metadata recorded.
- [ ] SHA-256 checksums recorded for both raw CSVs.
- [ ] Raw files remain unchanged and are not publicly committed without permission.

## Processing
- [ ] Invalid timestamps, outliers, duplicates and interpolation counts reviewed.
- [ ] Resampling/aggregation and continuity rules documented.
- [ ] 70/15/15 split chronological and non-overlapping.
- [ ] Scalers and learned transformations fitted on training only.
- [ ] Integrity audit JSON/CSV archived.

## Forecasting
- [ ] Sampling interval, input history and forecast horizon stated in minutes.
- [ ] Persistence, aggregate LSTM, spatial LSTM and QPSO-spatial LSTM compared fairly.
- [ ] Comparable tuning budgets and ablation documented.
- [ ] At least five predeclared seeds run.
- [ ] Native-unit MAE/RMSE per target and forecast lead reported.
- [ ] Bias, residual diagnostics, uncertainty, parameter count, training time and latency reported.
- [ ] Final independent evaluation is not reused for tuning.

## Synthetic context and control
- [ ] All Uganda-context scenario parameters and sources published.
- [ ] Synthetic outputs never described as measured Uganda data.
- [ ] Fan-energy proxy never described as kWh or measured savings.
- [ ] IAQ and thermal constraints assessed together for control claims.
- [ ] Rule-based simulation distinguished from predictive control and physical deployment.

## Publication
- [ ] Code, poster, dashboard and manuscript agree on research title and horizon.
- [ ] Every reference checked against publisher/DOI records.
- [ ] PRISMA counts tied to an itemised search/screening record for the stated scope.
- [ ] Funding, conflicts and author contributions factually confirmed.
- [ ] Supervisor/co-author approval obtained before submission.

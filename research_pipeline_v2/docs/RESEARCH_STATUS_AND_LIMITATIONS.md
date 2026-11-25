# Research Status and Limitations Register
Snapshot: 25 November 2026. Branch reviewed: research-pipeline-v2.

## Implemented
| Component | Repository evidence | Interpretation |
|---|---|---|
| Data preparation/QC | src/prepare_data.py | Two original sensor streams, timestamp alignment |
| Chronological split | src/split_and_generate_uganda.py | 70/15/15, chronological |
| Uganda scenarios | same | Synthetic assumptions, not local measurements |
| Spatial encoding | src/encode_spatial_features.py | Deterministic virtual coordinates |
| Forecasting | src/train_baselines.py | Persistence, aggregate LSTM, spatial LSTM |
| Optimisation | src/qpso_optimise_spatial_lstm.py | QPSO validation fitness |
| Sensitivity | src/analyse_uganda_sensitivity.py; src/analyse_uganda_factor_interactions.py | Scenario-level analysis |
| Control simulation | src/simulate_rule_based_ventilation_control.py | CO2-only rule simulation, proxy energy |
| Integrity audit | src/audit_data_integrity.py | Read-only checks |
| Packaging | Dockerfile and docker-compose.yml | CPU-compatible container |
| Dashboard | separate dashboard package | Saved-output visualisation, not live inference |

## Recorded experiment facts (must be regenerated to independently verify)
- 11 CO2 sensors; 16 temperature sensors.
- 54,893 aligned 10-minute observations; four continuous segments.
- Train/validation/test rows: 38,425 / 8,234 / 8,234.
- Valid windows: 38,369 / 8,220 / 8,220.
- 12-step history = 120 minutes; three-step horizon = 30 minutes.
- Five synthetic scenarios; 162 encoded sensor features.
- Historical integrity audit: 32/32 pass.
- Historical persistence validation MAE: CO2 2.3057 ppm; temperature 0.07177 °C; normalized macro MAE 0.010139.

These values come from recorded run notes; outputs are not in this branch. Preserve fresh run logs, metrics, config, commit and source checksums before citing them as final evidence.

## Critical test-set caveat
src/evaluate_final_test.py explicitly states that the chronological test partition was opened before a later validation-objective correction. Historical test results are pre-correction. Do not call them blind, untouched or use them for model selection. A fresh independent building or future period, collected/evaluated under a frozen protocol, is the preferred external test.

## Claims not established
- Validated performance in Ugandan buildings.
- Physical spatial relationships from the virtual grid.
- QPSO superiority without fair matched-budget comparisons and repeated seeds.
- Real energy savings: current energy quantity is dimensionless proxy, not kWh.
- Improved thermal comfort: control simulation omits thermal dynamics.
- Learned predictive or reinforcement-learning control.
- Physical HVAC control, embedded edge deployment or hardware latency.
- Health outcomes or regulatory compliance.

## Publication-readiness gaps
1. Focused reproducible literature search with exact query logs, exports and deduplication.
2. Itemised study screening and extraction; do not transfer broader-review 565/36 counts to narrower scope.
3. Full reference metadata verification.
4. Dataset citation, permission/licence and sensor/time metadata.
5. Repeated-seed experiments, per-lead native-unit metrics and uncertainty.
6. Fair QPSO versus non-QPSO tuning-budget comparison and feature ablation.
7. Independent temporal/building validation.
8. Calibrated/measured HVAC energy and joint IAQ/thermal constraints.
9. Consistent title, horizon, results and limitations across code, poster, dashboard and manuscript.

## Safe current claim
"We implemented and audited an experimental pipeline for two-target indoor forecasting using Building 59 observations, and explored synthetic Uganda-context sensitivity scenarios." No stronger local deployment, energy, health or quantum-computing claim is supported yet.

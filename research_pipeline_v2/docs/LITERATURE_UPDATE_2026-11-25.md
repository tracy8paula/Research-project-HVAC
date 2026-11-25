# Focused Literature Leads Identified on 25 November 2026

This is a verification queue for the narrowed review topic, not a claim that the records have already passed a full-text eligibility assessment or been added to the final included-study set.

## Newer, directly relevant work to assess

1. **CO2 prediction in office spaces using a physics-informed neural network (2026).** The publisher record describes a model combining occupant counts and IoT sensor data with a mass-balance equation, and reports an RMSE of 4.04 ppm for a ten-minute prediction. DOI: https://doi.org/10.1016/j.buildenv.2025.114035. This is directly relevant to short-horizon CO2 forecasting and physics-informed modelling. Extract exact dataset, forecast definition, test design, baseline, and metrics from the full text before comparing its RMSE with other studies.

2. **Dynamic and predictive modelling of indoor CO2 and energy consumption in educational buildings (2026).** The publisher record describes an EnergyPlus whole-building model, physics-based CO2 mass balance, LSTM forecasting and multi-scenario energy analysis. DOI: https://doi.org/10.1155/ina/4441388. The abstract states that the EnergyPlus model is benchmarked against published data rather than calibrated with measured energy data. This must therefore be classified as model-based/simulation evidence, not field-measured energy savings.

3. **Operator learning for energy-efficient ventilation control using CFD simulation of a classroom (2026).** The publisher record describes neural-operator/transformer prediction of spatiotemporal CO2 fields, comparison against reduced-order approaches, and CFD-based control evaluation. DOI: https://doi.org/10.1016/j.apenergy.2025.127035. It is highly relevant to forecast-to-control research, but the results arise from computational fluid dynamics simulation and must not be presented as field validation.

4. **Forecasting indoor air temperatures in tropical hospitals in Ghana (2026).** This empirical study uses 24 months of measurements in four hospitals in Accra and Tamale and reports approximately 1 °C accuracy at a 24-hour lead time for daily maximum indoor temperature in the evaluated wards. DOI: https://doi.org/10.1016/j.buildenv.2026.114418. It provides strong tropical African context, but its daily heat-warning horizon is distinct from a 30-minute HVAC forecast.

5. **Indoor temperature prediction with context-aware models in residential buildings (2023).** The publisher record describes data from 18 residential buildings and context-aware LSTM/Transformer models evaluated on multiple test datasets, including a six-hour horizon. DOI: https://doi.org/10.1016/j.buildenv.2023.110772. It is useful for cross-building generalisation and context variables.

6. **Kampala low-income housing thermal comfort and air quality study (2022).** This is empirical local context for naturally ventilated residential buildings; it is not a forecasting or predictive-control experiment. DOI: https://doi.org/10.1016/j.enbenv.2021.05.007.

## Review actions required

For every candidate, obtain the full text where legitimately accessible and extract:
- exact building type, geography and number of buildings/zones;
- sensor types, sample interval, study period and data availability;
- forecast target, input history and lead time;
- model architecture, baseline and tuning procedure;
- chronological/cross-building split design and leakage safeguards;
- metrics by forecast lead, uncertainty and residual analysis;
- whether control was prediction-only, simulation, hardware-in-loop or field-tested;
- whether energy was measured, calibrated or simulated;
- constraints for CO2, temperature, comfort and ventilation;
- limitations and transferability.

Do not combine reported errors across different horizons, datasets, targets or test designs. Do not infer a local Ugandan research gap solely from a handful of papers; describe it as the gap in the evidence located by the documented search strategy.

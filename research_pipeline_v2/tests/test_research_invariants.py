"""Fast tests for core data and scenario invariants."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from prepare_data import clean_sensor_stream, require_columns
from split_and_generate_uganda import chronological_split, generate_uganda_scenarios, simulate_co2_segment
from encode_spatial_features import encode_panel, fit_reading_scaler, virtual_grid

def test_sensor_column_discovery():
    frame = pd.DataFrame(columns=["zone_1_co2", "zone_2_co2", "other"])
    assert require_columns(frame, "zone_", "_co2") == ["zone_1_co2", "zone_2_co2"]

def test_sensor_column_discovery_rejects_empty_match():
    with pytest.raises(ValueError):
        require_columns(pd.DataFrame(columns=["date"]), "zone_", "_co2")

def test_cleaning_uses_median_for_duplicate_timestamps_and_rejects_outlier():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01 00:00","2026-01-01 00:00","2026-01-01 00:10","2026-01-01 00:30"]), "s1":[500.,600.,700.,5000.]})
    sampled, qc = clean_sensor_stream(frame, ["s1"], "date", 250., 2000., "10min", 1)
    assert sampled.loc[pd.Timestamp("2026-01-01 00:00"), "s1"] == 550.
    assert qc["duplicate_timestamp_groups"] == 1
    assert qc["values_above_qc_maximum"] == 1
    assert pd.isna(sampled.loc[pd.Timestamp("2026-01-01 00:30"), "s1"])

def test_chronological_split_is_ordered_and_disjoint():
    frame = pd.DataFrame({"date":pd.date_range("2026-01-01",periods=20,freq="10min"),"x":np.arange(20)})
    a,b,c = chronological_split(frame, {"split":{"train_ratio":.70,"validation_ratio":.15,"test_ratio":.15}})
    assert (len(a),len(b),len(c)) == (14,3,3)
    assert a.date.max() < b.date.min() < c.date.min()

def test_mass_balance_with_zero_occupancy_approaches_outdoor_concentration():
    values = simulate_co2_segment(1000.,25,0,.7)
    assert np.isfinite(values).all()
    assert values[0] == 1000.
    assert 420. < values[-1] < values[0]

def test_mass_balance_rejects_nonpositive_ach():
    with pytest.raises(ValueError):
        simulate_co2_segment(800.,3,2,0.)

def test_synthetic_scenarios_are_explicitly_labelled():
    training = pd.DataFrame({"date":pd.date_range("2026-01-01",periods=4,freq="10min"),"co2_ppm":[500.,510.,520.,530.],"temperature_c":[24.,24.1,24.2,24.3],"segment_id":[1,1,1,1]})
    scenarios = generate_uganda_scenarios(training)
    assert scenarios.scenario_id.nunique() == 5
    assert set(scenarios.data_origin) == {"synthetic_uganda_context_scenario"}
    assert set(scenarios.date) == set(training.date)

def test_virtual_grid_is_deterministic_and_bounded():
    a = virtual_grid(["sensor_c","sensor_a","sensor_b"])
    b = virtual_grid(["sensor_b","sensor_c","sensor_a"])
    assert a == b
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x,y in a.values())

def test_spatial_encoding_is_finite_and_uses_fitted_training_scaler():
    train = pd.DataFrame({"s1":[10.,20.],"s2":[5.,5.]})
    sensors=["s1","s2"]
    scaler=fit_reading_scaler(train,sensors)
    encoded=encode_panel(pd.DataFrame({"s1":[30.,np.nan],"s2":[5.,5.]}),sensors,virtual_grid(sensors),scaler,"co2")
    assert encoded.shape == (2,12)
    assert np.isfinite(encoded.to_numpy()).all()

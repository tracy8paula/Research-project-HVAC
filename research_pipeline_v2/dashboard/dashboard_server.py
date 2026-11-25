"""Live-replay prototype for the HVAC forecasting study.

This is a transparent replay of recorded Building 59 observations, not a live
sensor connection. Rows are emitted at an accelerated wall-clock rate. The
saved validation-selected aggregate LSTM forecasts the next 10/20/30 minutes.
As replay advances, the 30-minute forecast is matched to the corresponding
recorded observation so users can watch forecast error evolve.
"""
from __future__ import annotations

import csv
import json
import bisect
import sys
import threading
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from train_baselines import ForecastLSTM  # noqa: E402

OUTPUT = ROOT / "outputs"
PROCESSED = OUTPUT / "processed"
MODELS = OUTPUT / "models"
METRICS = OUTPUT / "metrics"
DASHBOARD = ROOT / "dashboard"
PORT = 8050

class ReplayEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.frame = pd.read_csv(PROCESSED / "building59_two_stream_10min.csv", parse_dates=["date"])
        self.frame = self.frame.sort_values("date").reset_index(drop=True)
        required = {"date", "co2_ppm", "temperature_c", "segment_id"}
        if not required.issubset(self.frame.columns):
            raise RuntimeError(f"Prepared data missing columns: {sorted(required - set(self.frame.columns))}")
        self.scalers = json.loads((MODELS / "baseline_scalers.json").read_text())
        ckpt = torch.load(MODELS / "lstm_without_spatial.pt", map_location="cpu", weights_only=False)
        self.sequence = int(ckpt["sequence_length"])
        self.horizon = int(ckpt["forecast_horizon_steps"])
        self.interval = 10
        self.model = ForecastLSTM(
            input_size=int(ckpt["input_size"]),
            hidden_dimension=int(ckpt["hidden_dimension"]),
            layers=int(ckpt["lstm_layers"]),
            dropout=float(ckpt["dropout"]),
            horizon=self.horizon,
            output_size=2,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.starts = self._valid_segment_starts()
        if not self.starts:
            raise RuntimeError("No continuous segment is long enough for live replay.")
        self.position = self.starts[0] + self.sequence - 1
        self.pending = {}
        self.errors = {"co2": [], "temperature": []}
        self.forecast_log = deque(maxlen=120)
        self.last_prediction = None
        self.source_label = "Recorded Building 59 observations — accelerated replay"
        self.scenario = "baseline_naturally_ventilated"

    def _valid_segment_starts(self):
        # Vectorised continuity check: avoid repeatedly slicing a DataFrame
        # for every candidate window on startup.
        starts = []
        total = self.sequence + self.horizon
        expected = np.timedelta64(self.interval, "m")
        for _, group in self.frame.groupby("segment_id", sort=False):
            positions = group.index.to_numpy()
            dates = pd.to_datetime(group["date"]).to_numpy(dtype="datetime64[ns]")
            if len(positions) < total:
                continue
            valid_delta = np.diff(dates) == expected
            window_ok = np.convolve(valid_delta.astype(np.int8),
                                    np.ones(total - 1, dtype=np.int8),
                                    mode="valid") == total - 1
            local_starts = np.flatnonzero(window_ok)
            starts.extend((positions[local_starts] + self.sequence - 1).tolist())
        return sorted(starts)

    def _predict(self, index):
        start = index - self.sequence + 1
        rows = self.frame.iloc[start:index + 1]
        if len(rows) != self.sequence or rows["segment_id"].nunique() != 1:
            return None
        xraw = rows[["co2_ppm", "temperature_c"]].to_numpy(dtype=np.float32)
        mins = np.asarray(self.scalers["nonspatial_input_minimum"], dtype=np.float32)
        spans = np.asarray(self.scalers["nonspatial_input_span"], dtype=np.float32)
        x = np.clip((xraw - mins) / spans, 0.0, 1.0).astype(np.float32)
        with torch.no_grad():
            pred_scaled = self.model(torch.from_numpy(x[None, :, :])).numpy()[0]
        target_min = np.asarray(self.scalers["minimum"], dtype=np.float32)
        target_span = np.asarray(self.scalers["span"], dtype=np.float32)
        pred = pred_scaled * target_span + target_min
        return [
            {"lead_minutes": (k + 1) * self.interval,
             "co2_ppm": round(float(pred[k, 0]), 2),
             "temperature_c": round(float(pred[k, 1]), 3)}
            for k in range(self.horizon)
        ]

    def _record_arrived_truth(self, index):
        item = self.pending.pop(index, None)
        if item is None:
            return
        actual = self.frame.iloc[index]
        if int(actual["segment_id"]) != item["segment_id"]:
            return
        self.errors["co2"].append(abs(float(actual["co2_ppm"]) - item["co2_ppm"]))
        self.errors["temperature"].append(abs(float(actual["temperature_c"]) - item["temperature_c"]))

    def _snapshot(self):
        row = self.frame.iloc[self.position]
        prediction = self._predict(self.position)
        self.last_prediction = prediction
        if prediction and self.position + self.horizon < len(self.frame):
            target_idx = self.position + self.horizon
            target = self.frame.iloc[target_idx]
            if int(target["segment_id"]) == int(row["segment_id"]):
                self.pending[target_idx] = {
                    "segment_id": int(row["segment_id"]),
                    "co2_ppm": prediction[-1]["co2_ppm"],
                    "temperature_c": prediction[-1]["temperature_c"],
                }
                self.forecast_log.append({
                    "time": pd.Timestamp(row["date"]).isoformat(),
                    "co2_ppm": float(row["co2_ppm"]),
                    "temperature_c": float(row["temperature_c"]),
                    "predicted_co2_30m": prediction[-1]["co2_ppm"],
                    "predicted_temperature_30m": prediction[-1]["temperature_c"],
                })
        self._record_arrived_truth(self.position)
        return self._payload()

    def _payload(self):
        row = self.frame.iloc[self.position]
        history_start = max(0, self.position - 35)
        hist = self.frame.iloc[history_start:self.position + 1]
        history = [
            {"time": pd.Timestamp(r.date).isoformat(), "co2_ppm": round(float(r.co2_ppm), 2),
             "temperature_c": round(float(r.temperature_c), 3)}
            for r in hist.itertuples()
        ]
        mae_co2 = float(np.mean(self.errors["co2"])) if self.errors["co2"] else None
        mae_temp = float(np.mean(self.errors["temperature"])) if self.errors["temperature"] else None
        return {
            "status": "running",
            "source": self.source_label,
            "model": "Validation-selected aggregate LSTM (lstm_without_spatial.pt)",
            "forecast_horizon_minutes": self.horizon * self.interval,
            "history_minutes": self.sequence * self.interval,
            "sampling_interval_minutes": self.interval,
            "timestamp": pd.Timestamp(row["date"]).isoformat(),
            "current": {"co2_ppm": round(float(row["co2_ppm"]), 2),
                        "temperature_c": round(float(row["temperature_c"]), 3)},
            "prediction": self.last_prediction,
            "history": history,
            "forecast_log": list(self.forecast_log),
            "accuracy": {"matched_30min_forecasts": len(self.errors["co2"]),
                         "co2_mae_ppm": round(mae_co2, 3) if mae_co2 is not None else None,
                         "temperature_mae_c": round(mae_temp, 4) if mae_temp is not None else None},
            "progress": {"current_row": int(self.position + 1), "total_rows": int(len(self.frame)),
                         "segment_id": int(row["segment_id"])},
        }

    def step(self, amount=1):
        with self.lock:
            for _ in range(max(1, min(int(amount), 20))):
                next_index = bisect.bisect_right(self.starts, self.position)
                if next_index >= len(self.starts):
                    self.position = self.starts[0]
                    self.pending.clear()
                    self.errors = {"co2": [], "temperature": []}
                    self.forecast_log.clear()
                else:
                    self.position = self.starts[next_index]
                self._record_arrived_truth(self.position)
                self._snapshot()
            return self._payload()

    def reset(self):
        with self.lock:
            self.position = self.starts[0]
            self.pending.clear()
            self.errors = {"co2": [], "temperature": []}
            self.forecast_log.clear()
            return self._snapshot()

    def static_results(self):
        def read_csv(name):
            path = METRICS / name
            if not path.exists():
                return []
            return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")
        def read_json(name):
            path = METRICS / name
            return json.loads(path.read_text()) if path.exists() else {}
        return {
            "benchmark": read_csv("final_held_out_test_comparison.csv"),
            "scenarios": read_csv("uganda_scenario_summary.csv"),
            "control": read_csv("uganda_rule_based_control_summary.csv"),
            "qpso_search": read_csv("qpso_search_history.csv"),
            "factorial": read_csv("uganda_occupancy_ventilation_factorial.csv"),
            "temperature_sensitivity": read_csv("uganda_temperature_offset_sensitivity.csv"),
            "sensitivity": read_json("uganda_sensitivity_analysis.json"),
            "interaction": read_json("uganda_occupancy_ventilation_interaction.json"),
            "integrity": read_json("data_integrity_audit.json"),
            "control_note": "Offline rule-based simulation. Fan-energy is a dimensionless proxy, not measured kWh.",
            "benchmark_note": "Historical test outputs are exploratory: the test partition was inspected before a later validation-objective correction."
        }

ENGINE = ReplayEngine()

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        payload = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            file = DASHBOARD / "index.html"
            return self._send(200, file.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self._send(200, ENGINE._payload())
        if path == "/api/results":
            return self._send(200, ENGINE.static_results())
        if path == "/health":
            return self._send(200, {"status": "healthy", "service": "hvac-live-replay-dashboard"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/step":
            return self._send(200, ENGINE.step(1))
        if path == "/api/reset":
            return self._send(200, ENGINE.reset())
        return self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        print("[dashboard] " + fmt % args, flush=True)

if __name__ == "__main__":
    print(f"Dashboard available on http://0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

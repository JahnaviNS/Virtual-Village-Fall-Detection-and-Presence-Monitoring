"""
Fall Detection Dashboard — FastAPI Backend
==========================================
Run with:  uvicorn api:app --reload --port 8000

Expects:
  - xgboost_fall_detection_model.pkl  (saved from your notebook)
  - fall_prediction_results.csv       (output of predict_fall_from_csv.py)

The /stream endpoint sends Server-Sent Events every 5s so the
HTML dashboard updates in real time without polling.
"""

import json
import pickle
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("xgboost_fall_detection_model.pkl")
RESULTS_CSV  = Path("fall_prediction_results.csv")
THRESHOLD    = 0.35
HISTORY_LEN  = 60          # frames to keep per room for the live chart
POLL_SECS    = 5           # how often the backend re-reads the CSV

ROOMS = ["Bedroom", "Kitchen", "Living Room", "Office", "Sunroom", "Hallway"]

FEATURE_COLS = [
    "x1", "y1", "z1",
    "z1_lag1", "z1_lag2", "z1_lag3",
    "z1_delta1", "z1_delta3",
    "z1_roll3_mean", "z1_roll3_std",
    "z1_drop_from_lag3",
    "z1_lead_mean5", "z1_lead_std3", "z1_lead8",
    "z1_is_low",
]

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Fall Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── model loading (lazy — won't crash if pkl not present yet) ─────────────────
_model = None

def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            pkg = pickle.load(f)
        _model = pkg["model"] if isinstance(pkg, dict) else pkg
    return _model


# ── in-memory ring buffers for live z1 history per room ───────────────────────
_history: dict[str, deque] = {r: deque(maxlen=HISTORY_LEN) for r in ROOMS}
_alerts: list[dict] = []


def _load_predictions() -> pd.DataFrame:
    """Read the latest fall_prediction_results.csv."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_CSV)
    df["createTime"] = pd.to_datetime(df["createTime"], errors="coerce")
    return df.dropna(subset=["createTime"])


def _build_state() -> dict:
    """
    Compute the full dashboard state from the CSV.
    Called every POLL_SECS seconds.
    """
    df = _load_predictions()

    room_status = {}
    falls_today = 0
    false_alerts_today = 0
    alerts = []

    # Use the most recent date in the data itself (not today's date)
    # This handles historical datasets like kitchen_fall_60sec.json (March 2026)
    data_date = df["createTime"].dt.date.max() if not df.empty else datetime.now().date()

    for room in ROOMS:
        sub = df[df["room"] == room].sort_values("createTime")
        if sub.empty:
            room_status[room] = {
                "z1": None, "fall_probability": 0.0,
                "status": "offline", "history": []
            }
            continue

        latest = sub.iloc[-1]
        z1     = float(latest.get("z1", 0))
        prob   = float(latest.get("fall_probability", 0)) if "fall_probability" in latest else 0.0
        pred   = int(latest.get("predicted_fall", 0))   if "predicted_fall" in latest else 0

        # update ring buffer
        _history[room].append({"t": latest["createTime"].isoformat(), "z1": round(z1, 3)})

        status = "fall" if pred == 1 else ("low" if z1 < 0.4 else "normal")
        room_status[room] = {
            "z1":      round(z1, 3),
            "fall_probability": round(prob, 3),
            "status":  status,
            "history": list(_history[room]),
        }

        # use data's own date range instead of today
        today_rows = sub[sub["createTime"].dt.date == data_date]
        fall_rows  = today_rows[today_rows["predicted_fall"] == 1] \
                     if "predicted_fall" in today_rows.columns else pd.DataFrame()

        for _, row in fall_rows.iterrows():
            p = float(row.get("fall_probability", 0))
            confirmed = p >= THRESHOLD
            if confirmed:
                falls_today += 1
            else:
                false_alerts_today += 1
            alerts.append({
                "room":      room,
                "time":      row["createTime"].strftime("%H:%M:%S"),
                "z1":        round(float(row.get("z1", 0)), 3),
                "prob":      round(p, 3),
                "confirmed": confirmed,
            })

    alerts.sort(key=lambda a: a["time"], reverse=True)

    return {
        "timestamp":       datetime.now().isoformat(),
        "rooms":           room_status,
        "falls_today":     falls_today,
        "false_alerts":    false_alerts_today,
        "alerts":          alerts[:20],
        "model_loaded":    get_model() is not None,
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/state")
def get_state():
    """Snapshot of current dashboard state (JSON)."""
    return _build_state()


@app.get("/stream")
async def stream():
    """
    Server-Sent Events — sends updated state every POLL_SECS seconds.
    The HTML dashboard subscribes here with EventSource.
    """
    async def event_generator():
        while True:
            state = _build_state()
            payload = json.dumps(state)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(POLL_SECS)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/predict")
def predict(data: dict):
    """
    Accept a single frame dict with the 15 feature keys,
    return fall probability and prediction.

    Example body:
    {
      "x1": -0.23, "y1": 2.11, "z1": 0.22,
      "z1_lag1": 0.87, "z1_lag2": 0.91, "z1_lag3": 0.95,
      "z1_delta1": -0.65, "z1_delta3": -0.73,
      "z1_roll3_mean": 0.91, "z1_roll3_std": 0.04,
      "z1_drop_from_lag3": 1.05,
      "z1_lead_mean5": 0.23, "z1_lead_std3": 0.01, "z1_lead8": 0.24,
      "z1_is_low": 1
    }
    """
    model = get_model()
    if model is None:
        return {"error": "model not loaded — place xgboost_fall_detection_model.pkl next to api.py"}

    row = pd.DataFrame([{c: data.get(c, np.nan) for c in FEATURE_COLS}])
    prob  = float(model.predict_proba(row)[0, 1])
    pred  = int(prob >= THRESHOLD)
    return {"fall_prob": round(prob, 4), "fall_pred": pred, "threshold": THRESHOLD}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}

"""
WeatherMind FastAPI Backend
============================
Serves ML predictions and simulates historical data retrieval.
Run with:  uvicorn api:app --reload --port 8000

Endpoints:
  POST /predict         → single-day prediction
  POST /forecast        → N-day forecast
  GET  /historical      → 12-month historical summary for charts
  GET  /heatmap         → 365-day temperature heatmap data
  GET  /metrics         → model accuracy metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import json
import os
import math
from datetime import datetime, timedelta

# ── LOAD ARTIFACTS ─────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "model_artifacts")

def load_model(name):
    rf  = joblib.load(os.path.join(BASE, f"rf_{name}.pkl"))
    gbr = joblib.load(os.path.join(BASE, f"gbr_{name}.pkl"))
    return rf, gbr

with open(os.path.join(BASE, "model_meta.json")) as f:
    META = json.load(f)

MODELS = {t: load_model(t) for t in META["targets"]}
LOCATIONS = META["locations"]
SEASON_CODE = META["season_codes"]
SEASON_TEMP = META["season_temp_delta"]
FEATURES = META["features"]

DIRECTIONS = ['N','NE','NE','E','SE','SE','S','SW','SW','W','NW','NW']

# ── APP ─────────────────────────────────────────────────────────────
app = FastAPI(title="WeatherMind API", version="2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SCHEMAS ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    location:    str   = "Lagos"
    season:      str   = "dry"
    humidity:    float = Field(72, ge=0, le=100)
    wind_speed:  float = Field(18, ge=0, le=200)
    pressure:    float = Field(1013, ge=900, le=1100)
    cloud_cover: float = Field(45, ge=0, le=100)
    dew_point:   float = Field(18, ge=-30, le=40)
    days:        int   = Field(7, ge=1, le=14)

# ── HELPERS ─────────────────────────────────────────────────────────
def get_loc(name: str):
    if name not in LOCATIONS:
        raise HTTPException(404, f"Location '{name}' not found")
    return LOCATIONS[name]

def build_features(req: PredictRequest, day_offset: int = 0):
    loc = get_loc(req.location)
    today = datetime.now() + timedelta(days=day_offset)
    doy   = today.timetuple().tm_yday
    month = today.month
    sc    = SEASON_CODE.get(req.season, 0)
    return np.array([[
        month, doy,
        req.humidity, req.wind_speed, req.pressure,
        req.cloud_cover, req.dew_point,
        sc, loc["lat"], loc["lon"]
    ]])

def ensemble_predict(target: str, X):
    rf, gbr = MODELS[target]
    return float((rf.predict(X)[0] + gbr.predict(X)[0]) / 2)

def weather_icon(rain_p, cloud, temp):
    if rain_p > 0.75: return "⛈"
    if rain_p > 0.5:  return "🌧"
    if rain_p > 0.3:  return "🌦"
    if cloud > 70:    return "☁️"
    if cloud > 40:    return "⛅"
    if temp > 35:     return "🌞"
    return "☀️"

def weather_desc(rain_p, cloud, temp, loc_name):
    if rain_p > 0.75: return "Thunderstorm Likely"
    if rain_p > 0.55: return "Heavy Rain Expected"
    if rain_p > 0.35: return "Scattered Showers"
    if cloud > 70:    return "Mostly Cloudy"
    if cloud > 40:    return "Partly Cloudy"
    if temp > 38:     return "Scorching Hot"
    if temp > 30:     return "Clear & Warm"
    if temp > 20:     return "Mild & Pleasant"
    return "Cool & Clear"

WEEK_DAYS = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]

# ── ROUTES ──────────────────────────────────────────────────────────
@app.post("/predict")
def predict_today(req: PredictRequest):
    X    = build_features(req)
    temp = ensemble_predict("temperature", X)
    rain = np.clip(ensemble_predict("rain_probability", X), 0, 1)
    rain_mm = np.clip(ensemble_predict("rainfall_mm", X), 0, 150)
    uv   = np.clip(ensemble_predict("uv_index", X), 1, 13)

    loc  = get_loc(req.location)
    wind_dir = DIRECTIONS[int(req.wind_speed / 8) % len(DIRECTIONS)]
    feels = temp + (req.humidity - 60) * 0.05 - req.wind_speed * 0.01

    return {
        "location":    req.location,
        "temperature": round(temp, 1),
        "feels_like":  round(feels, 1),
        "rain_probability": round(rain * 100, 1),
        "rainfall_mm": round(rain_mm, 1),
        "uv_index":    round(uv, 1),
        "wind_direction": wind_dir,
        "wind_speed":  req.wind_speed,
        "humidity":    req.humidity,
        "pressure":    req.pressure,
        "icon":        weather_icon(rain, req.cloud_cover, temp),
        "description": weather_desc(rain, req.cloud_cover, temp, req.location),
        "lat": loc["lat"],
        "lon": loc["lon"],
    }

@app.post("/forecast")
def forecast(req: PredictRequest):
    days_data = []
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    for d in range(req.days):
        X      = build_features(req, day_offset=d)
        noise_h = np.random.normal(0, 5)
        noise_c = np.random.normal(0, 10)
        X[0][2] = np.clip(req.humidity + noise_h, 20, 99)
        X[0][5] = np.clip(req.cloud_cover + noise_c, 0, 100)

        temp = ensemble_predict("temperature", X) + np.random.normal(0, 1.5)
        rain = np.clip(ensemble_predict("rain_probability", X) + np.random.normal(0, 0.05), 0, 1)
        rain_mm = np.clip(ensemble_predict("rainfall_mm", X), 0, 150)
        uv   = np.clip(ensemble_predict("uv_index", X), 1, 13)

        date = datetime.now() + timedelta(days=d)
        days_data.append({
            "day":         "Today" if d == 0 else WEEK_DAYS[date.weekday()],
            "date":        date.strftime("%b %d"),
            "temperature_hi": round(temp, 1),
            "temperature_lo": round(temp - 4 - np.random.uniform(0, 3), 1),
            "rain_probability": round(rain * 100, 1),
            "rainfall_mm": round(rain_mm, 1),
            "uv_index":    round(uv, 1),
            "humidity":    round(float(X[0][2]), 1),
            "cloud_cover": round(float(X[0][5]), 1),
            "icon":        weather_icon(rain, float(X[0][5]), temp),
            "description": weather_desc(rain, float(X[0][5]), temp, req.location),
        })
    return {"location": req.location, "forecast": days_data}

@app.get("/historical")
def historical(location: str = "Lagos", season: str = "dry"):
    """12-month monthly average summary for charts."""
    loc = get_loc(location)
    sc  = SEASON_CODE.get(season, 0)
    months = []
    for m in range(1, 13):
        doy   = int((m - 0.5) * 30.44)
        X = np.array([[m, doy, 68, 15, 1013, 50, 16, sc, loc["lat"], loc["lon"]]])
        temp    = ensemble_predict("temperature", X) + np.sin(doy / 60) * 2
        rain_p  = np.clip(ensemble_predict("rain_probability", X), 0, 1)
        rain_mm = np.clip(ensemble_predict("rainfall_mm", X), 0, 150)
        uv      = np.clip(ensemble_predict("uv_index", X), 1, 13)
        months.append({
            "month":          ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
            "avg_temp":       round(temp, 1),
            "avg_temp_hi":    round(temp + 3.5, 1),
            "avg_temp_lo":    round(temp - 4.5, 1),
            "rain_probability": round(rain_p * 100, 1),
            "avg_rainfall_mm":  round(rain_mm, 1),
            "avg_uv":         round(uv, 1),
        })
    return {"location": location, "historical": months}

@app.get("/heatmap")
def heatmap(location: str = "Lagos", season: str = "dry"):
    """365-day temperature heatmap for the calendar view."""
    loc = get_loc(location)
    sc  = SEASON_CODE.get(season, 0)
    data = []
    base_date = datetime(datetime.now().year, 1, 1)
    for doy in range(1, 366):
        m = min(12, int(doy / 30.44) + 1)
        X = np.array([[m, doy, 65 + np.sin(doy/60)*15, 15, 1013, 45, 16, sc, loc["lat"], loc["lon"]]])
        temp = ensemble_predict("temperature", X)
        rain = np.clip(ensemble_predict("rain_probability", X), 0, 1)
        date = (base_date + timedelta(days=doy-1)).strftime("%Y-%m-%d")
        data.append({"date": date, "temp": round(temp, 1), "rain": round(rain*100,1)})
    return {"location": location, "heatmap": data}

@app.get("/metrics")
def metrics():
    return META["metrics"]

@app.get("/locations")
def locations():
    return {"locations": list(LOCATIONS.keys())}

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "2.4",
            "training_samples": META["training_samples"]}

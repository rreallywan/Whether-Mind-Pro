"""
WeatherMind ML Training Script
================================
Trains a Random Forest + Gradient Boosting ensemble on synthetic
10-year historical weather data. Saves model artifacts for FastAPI.

Features used:
  month, day_of_year, humidity, wind_speed, pressure,
  cloud_cover, dew_point, season_code, lat, lon

Targets:
  temperature, rain_probability, uv_index, rainfall_mm
"""

import numpy as np
import pandas as pd
import joblib, json, os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "model_artifacts")
os.makedirs(OUT, exist_ok=True)

# ── LOCATION CLIMATE PROFILES ─────────────────────────────────────
LOCATIONS = {
    "Lagos":   dict(lat=6.5, lon=3.4,  base_temp=30, rain_bias=0.35, temp_amp=3),
    "Abuja":   dict(lat=9.1, lon=7.4,  base_temp=28, rain_bias=0.28, temp_amp=5),
    "Kano":    dict(lat=12.0,lon=8.5,  base_temp=33, rain_bias=0.10, temp_amp=8),
    "Yola":    dict(lat=9.2, lon=12.5, base_temp=32, rain_bias=0.22, temp_amp=6),
    "London":  dict(lat=51.5,lon=-0.1, base_temp=13, rain_bias=0.48, temp_amp=9),
    "NewYork": dict(lat=40.7,lon=-74.0,base_temp=19, rain_bias=0.30, temp_amp=14),
    "Tokyo":   dict(lat=35.7,lon=139.7,base_temp=21, rain_bias=0.38, temp_amp=12),
    "Dubai":   dict(lat=25.2,lon=55.3, base_temp=38, rain_bias=0.04, temp_amp=6),
}

SEASON_CODE = {"dry":0,"wet":1,"harmattan":2,"spring":3,"summer":4,"autumn":5,"winter":6}
SEASON_TEMP_DELTA = {"dry":3,"wet":-2,"harmattan":2,"spring":0,"summer":5,"autumn":-1,"winter":-7}
SEASON_RAIN_DELTA = {"dry":-0.1,"wet":0.25,"harmattan":-0.05,"spring":0.05,
                     "summer":0.1,"autumn":0.05,"winter":0.0}


def generate_dataset(n_years=10):
    rows = []
    for _ in range(n_years * 365):
        loc_name = np.random.choice(list(LOCATIONS.keys()))
        loc = LOCATIONS[loc_name]

        day_of_year = np.random.randint(1, 366)
        month = min(12, max(1, int(day_of_year / 30.44) + 1))

        # Seasonal adjustment via sinusoidal
        season_wave = np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)
        temp_seasonal = loc["base_temp"] + loc["temp_amp"] * season_wave

        humidity    = float(np.clip(np.random.normal(68, 15), 20, 99))
        wind_speed  = float(np.clip(np.random.exponential(15), 0, 90))
        pressure    = float(np.clip(np.random.normal(1013, 8), 960, 1050))
        cloud_cover = float(np.clip(np.random.beta(2, 3) * 100, 0, 100))
        dew_point   = float(np.clip(humidity * 0.3 - 5 + np.random.normal(0, 2), -15, 35))
        season_idx  = np.random.randint(0, 7)
        s_name      = list(SEASON_CODE.keys())[season_idx]

        # Target: temperature
        temp = (temp_seasonal
                + SEASON_TEMP_DELTA[s_name]
                + (pressure - 1013) * 0.04
                + (humidity - 60) * -0.06
                + cloud_cover * -0.07
                + wind_speed * -0.02
                + np.random.normal(0, 1.5))
        temp = float(np.clip(temp, -20, 50))

        # Target: rain probability
        rain_p = (loc["rain_bias"]
                  + SEASON_RAIN_DELTA[s_name]
                  + (humidity - 60) / 200
                  + (cloud_cover - 50) / 300
                  + (1005 - pressure) / 500)
        rain_p = float(np.clip(rain_p + np.random.normal(0, 0.05), 0.01, 0.97))

        # Target: rainfall mm
        rainfall_mm = float(np.clip(
            rain_p * np.random.exponential(12) if rain_p > 0.4 else np.random.exponential(1.5),
            0, 150))

        # Target: UV index
        uv = float(np.clip(
            10 - cloud_cover * 0.08 - (4 if rain_p > 0.5 else 0)
            + (loc["lat"] < 20) * 2 + np.random.normal(0, 0.5),
            1, 13))

        rows.append({
            "month": month,
            "day_of_year": day_of_year,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "cloud_cover": cloud_cover,
            "dew_point": dew_point,
            "season_code": season_idx,
            "lat": loc["lat"],
            "lon": loc["lon"],
            "temperature": temp,
            "rain_probability": rain_p,
            "rainfall_mm": rainfall_mm,
            "uv_index": uv,
            "location": loc_name,
        })

    return pd.DataFrame(rows)


FEATURES = ["month","day_of_year","humidity","wind_speed","pressure",
            "cloud_cover","dew_point","season_code","lat","lon"]
TARGETS  = ["temperature","rain_probability","rainfall_mm","uv_index"]


def train_and_save():
    print("🌤  Generating 10-year historical dataset...")
    df = generate_dataset(10)
    print(f"   Dataset shape: {df.shape}")

    X = df[FEATURES].values
    metrics = {}
    models  = {}

    for target in TARGETS:
        y = df[target].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"\n🔧  Training model for: {target}")

        # Ensemble: RF + GBR averaged
        rf  = RandomForestRegressor(n_estimators=200, max_depth=12,
                                    min_samples_leaf=4, random_state=42, n_jobs=-1)
        gbr = GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                        learning_rate=0.08, random_state=42)

        rf.fit(X_tr, y_tr)
        gbr.fit(X_tr, y_tr)

        y_pred_rf  = rf.predict(X_te)
        y_pred_gbr = gbr.predict(X_te)
        y_pred     = (y_pred_rf + y_pred_gbr) / 2

        mae = mean_absolute_error(y_te, y_pred)
        r2  = r2_score(y_te, y_pred)

        cv_rf  = cross_val_score(rf,  X, y, cv=5, scoring='r2').mean()
        cv_gbr = cross_val_score(gbr, X, y, cv=5, scoring='r2').mean()

        print(f"   MAE={mae:.3f}  R²={r2:.3f}  CV-R²(rf)={cv_rf:.3f}  CV-R²(gbr)={cv_gbr:.3f}")

        joblib.dump(rf,  os.path.join(OUT, f"rf_{target}.pkl"))
        joblib.dump(gbr, os.path.join(OUT, f"gbr_{target}.pkl"))

        # Feature importances
        importances = dict(zip(FEATURES, rf.feature_importances_.tolist()))

        metrics[target] = {
            "mae": round(mae, 3),
            "r2":  round(r2, 3),
            "cv_r2_rf":  round(cv_rf, 3),
            "cv_r2_gbr": round(cv_gbr, 3),
            "feature_importances": importances,
        }
        models[target] = (rf, gbr)

    # Save metadata
    meta = {
        "features": FEATURES,
        "targets":  TARGETS,
        "locations": LOCATIONS,
        "season_codes": SEASON_CODE,
        "season_temp_delta": SEASON_TEMP_DELTA,
        "metrics": metrics,
        "training_samples": len(df),
    }
    with open(os.path.join(OUT, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅  All models saved to", OUT)
    print("\n📊  Summary:")
    for t, m in metrics.items():
        print(f"   {t:20s}  MAE={m['mae']}  R²={m['r2']}")

    return models, metrics


if __name__ == "__main__":
    train_and_save()

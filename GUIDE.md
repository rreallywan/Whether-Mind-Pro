# WeatherMind Pro — Beginner Setup Guide
## Complete Step-by-Step Instructions

---

## What Is This Project?

WeatherMind Pro is a weather prediction system made up of three parts:

| File | What it does |
|---|---|
| `train_model.py` | Trains the AI/ML model using historical weather data |
| `api.py` | Runs a local server so the dashboard can fetch predictions |
| `weathermind-pro.html` | The visual dashboard you open in your browser |

You can use the dashboard **without running the server** (it has a built-in Demo Mode).
But if you want **live ML predictions**, follow all the steps below.

---

## STEP 1 — Install Python

1. Go to **https://www.python.org/downloads/**
2. Download **Python 3.10 or newer**
3. During installation, tick **"Add Python to PATH"** ✅ (very important!)
4. Click Install

To verify it worked, open a terminal (Command Prompt on Windows) and type:
```
python --version
```
You should see something like: `Python 3.11.4`

---

## STEP 2 — Open a Terminal in the Project Folder

### Windows:
1. Open the `weathermind-pro` folder
2. Click the address bar at the top (where it shows the folder path)
3. Type `cmd` and press Enter
4. A black Command Prompt window opens inside your folder ✅

### Mac / Linux:
1. Open Terminal
2. Type `cd ` (with a space), then drag-and-drop your folder into the terminal
3. Press Enter

---

## STEP 3 — Install Required Libraries

Copy and paste this command into your terminal, then press Enter:

```
pip install fastapi uvicorn scikit-learn pandas numpy joblib requests
```

Wait for it to finish (may take 1–3 minutes depending on your internet speed).

---

## STEP 4 — Train the AI Model

This step teaches the model using 10 years of weather data.
Run this command:

```
python train_model.py
```

You will see output like:
```
🌤  Generating 10-year historical dataset...
🔧  Training model for: temperature
    MAE=1.724  R²=0.958
🔧  Training model for: rain_probability
    MAE=0.045  R²=0.910
✅  All models saved to model_artifacts/
```

This creates a folder called `model_artifacts/` — **do not delete it**.
You only need to run this once.

---

## STEP 5 — Start the Prediction Server

Run this command:

```
uvicorn api:app --reload --port 8000
```

You will see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

✅ **Leave this terminal window open** while using the dashboard.

You can verify the server is working by opening:
**http://localhost:8000/health** in your browser.
It should show: `{"status":"ok"}`

---

## STEP 6 — Open the Dashboard

1. Open the `weathermind-pro` folder
2. Double-click **`weathermind-pro.html`**
3. It opens in your web browser automatically

At the top-right you will see:
- 🟢 **API CONNECTED** — server is running, live ML predictions active
- 🟡 **DEMO MODE** — server is not running, using built-in predictions

---

## Using the Dashboard

### Tabs:
| Tab | What you'll find |
|---|---|
| ⚡ Dashboard | Today's prediction, 7-day forecast, confidence scores |
| 🌡 Temperature | High/Low charts, monthly averages, feels-like comparison |
| 🌧 Rainfall | Daily rainfall, rain probability gauge, scatter charts |
| 📅 Heatmap | 365-day calendar heatmap (hover cells for values) |
| 🧠 Model | Feature importance, R² scores, prediction vs historical |

### Input Parameters:
| Field | Meaning |
|---|---|
| Location | City to predict for |
| Season | Current season (affects rainfall & temperature) |
| Humidity | How moist the air is (0–100%) |
| Wind Speed | Wind in km/h |
| Pressure | Atmospheric pressure in hPa (normal = ~1013) |
| Cloud Cover | How cloudy it is (0–100%) |
| Dew Point | Temperature at which dew forms |
| Forecast Days | How many days to predict ahead |

Click **"⚡ Run Full Prediction Suite"** after adjusting any values.

---

## Locations Available

| City | State/Country |
|---|---|
| Lagos | Nigeria |
| Abuja | Nigeria |
| Kano | Nigeria |
| **Yola** | **Adamawa, Nigeria** |
| London | United Kingdom |
| New York | USA |
| Tokyo | Japan |
| Dubai | UAE |

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `python not found` | Re-install Python and tick "Add to PATH" |
| `pip not found` | Use `python -m pip install ...` instead |
| `ModuleNotFoundError` | Run Step 3 again |
| `Address already in use` | Another app is using port 8000. Use `--port 8001` instead |
| Dashboard shows DEMO MODE | Make sure Step 5 is running in a terminal |
| `model_artifacts not found` | Run Step 4 first (`python train_model.py`) |

---

## Folder Structure After Setup

```
weathermind-pro/
│
├── weathermind-pro.html     ← Open this in browser
├── train_model.py           ← Run once to train AI
├── api.py                   ← Run to start server
├── run_server.sh            ← Mac/Linux shortcut to start server
├── GUIDE.md                 ← This file
│
└── model_artifacts/         ← Created after training
    ├── rf_temperature.pkl
    ├── rf_rain_probability.pkl
    ├── rf_rainfall_mm.pkl
    ├── rf_uv_index.pkl
    ├── gbr_temperature.pkl
    ├── gbr_rain_probability.pkl
    ├── gbr_rainfall_mm.pkl
    ├── gbr_uv_index.pkl
    └── model_meta.json
```

---

## Quick Start Summary (3 commands)

```bash
# 1. Install libraries (once)
pip install fastapi uvicorn scikit-learn pandas numpy joblib

# 2. Train the model (once)
python train_model.py

# 3. Start the server (every time you use the app)
uvicorn api:app --reload --port 8000
```

Then open **weathermind-pro.html** in your browser. That's it! 🎉

---

*WeatherMind Pro · ML Model v2.4 · Built for learning & demonstration*

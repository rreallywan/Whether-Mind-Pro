@echo off
echo Starting WeatherMind Pro Server...
uvicorn api:app --reload --port 8000
pause

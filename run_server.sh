#!/bin/bash
cd "$(dirname "$0")"
echo "Starting WeatherMind API on http://localhost:8000"
echo "Docs at http://localhost:8000/docs"
uvicorn api:app --reload --port 8000

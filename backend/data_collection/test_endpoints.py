import os
import sys
import json
from fastapi.testclient import TestClient

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server import app

print("--- EVIDENTIARY VERIFICATION OF API ENDPOINTS ---")

# Use context manager to trigger startup lifecycle events (loads model and scaler)
with TestClient(app) as client:
    print("\n1. GET /api/health")
    r_health = client.get("/api/health")
    print(f"Status Code: {r_health.status_code}")
    print(json.dumps(r_health.json(), indent=2))

    print("\n2. GET /api/system/gee-status")
    r_gee = client.get("/api/system/gee-status")
    print(f"Status Code: {r_gee.status_code}")
    print(json.dumps(r_gee.json(), indent=2))

    print("\n3. POST /api/predict/location (Delhi Pinpoint)")
    req_body = {
        "location": "Delhi Pinpoint",
        "latitude": 28.6139,
        "longitude": 77.2090
    }
    r_pred = client.post("/api/predict/location", json=req_body)
    print(f"Status Code: {r_pred.status_code}")
    if r_pred.status_code == 200:
        res = r_pred.json()
        subset = {
            "location": res.get("location"),
            "prediction": res.get("prediction"),
            "environment": res.get("environment"),
            "recommendation": res.get("recommendation")[:200] + "..."
        }
        print(json.dumps(subset, indent=2))
    else:
        print(r_pred.text)

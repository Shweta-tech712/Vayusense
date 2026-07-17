import requests
import json

base_url = "http://127.0.0.1:8000/api/predict/location"

cities = ["Delhi", "Mumbai", "Pune", "Kochi"]
results = {}

for city in cities:
    print(f"Testing {city} via POST...")
    try:
        resp = requests.post(base_url, json={"location": city})
        if resp.status_code == 200:
            data = resp.json()
            results[city] = {
                "AQI": data.get("prediction", {}).get("AQI"),
                "Weather": data.get("environment", {})
            }
        else:
            print(f"Error {resp.status_code} for {city}: {resp.text}")
    except Exception as e:
        print(f"Exception for {city}: {e}")

with open("api_endpoint_validation_report.json", "w") as f:
    json.dump(results, f, indent=4)
print("Done.")

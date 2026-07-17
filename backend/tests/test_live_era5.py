import sys
import os
import json
import urllib.request
import traceback
import logging

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.services.live_weather_service import live_weather_service

def get_open_meteo_weather(lat, lon):
    try:
        # Open-Meteo provides free no-key access for QA purposes
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m"
        req = urllib.request.Request(url, headers={'User-Agent': 'VayuSense/1.0 QA Script'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            current = data.get("current", {})
            return {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "pressure": current.get("surface_pressure"),
                "rainfall": current.get("precipitation"),
                "wind_speed": current.get("wind_speed_10m"),
                "wind_direction": current.get("wind_direction_10m")
            }
    except Exception as e:
        print(f"Open-Meteo failed: {e}")
        return {}

def test_cities():
    cities = {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Pune": (18.5204, 73.8567),
        "Bengaluru": (12.9716, 77.5946)
    }

    era5_results = []
    qa_results = []

    for name, (lat, lon) in cities.items():
        print(f"\nTesting {name} ({lat}, {lon})...")
        try:
            # 1. Fetch ERA5
            era5_data = live_weather_service.get_live_weather(lat, lon)
            
            # Format report
            era_report = {
                "city": name,
                "latitude": lat,
                "longitude": lon,
                "status": "success",
                "is_live": era5_data.get("is_live"),
                "retrieval_time": era5_data.get("retrieval_time"),
                "weather": era5_data.get("weather")
            }
            era5_results.append(era_report)

            # 2. Fetch Open-Meteo for QA
            open_meteo_data = get_open_meteo_weather(lat, lon)
            qa_report = {
                "city": name,
                "latitude": lat,
                "longitude": lon,
                "era5_temperature": era5_data.get("weather", {}).get("temperature"),
                "om_temperature": open_meteo_data.get("temperature"),
                "era5_humidity": era5_data.get("weather", {}).get("humidity"),
                "om_humidity": open_meteo_data.get("humidity"),
                "era5_wind_speed": era5_data.get("weather", {}).get("wind_speed"),
                "om_wind_speed": open_meteo_data.get("wind_speed")
            }
            qa_results.append(qa_report)
            
            print(f"Success for {name}. Temp: {era_report['weather']['temperature']}C, is_live: {era_report['is_live']}")

        except Exception as e:
            print(f"Failed for {name}: {e}")
            traceback.print_exc()
            era5_results.append({
                "city": name,
                "status": "failed",
                "error": str(e)
            })

    # Save reports
    report_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(report_dir, exist_ok=True)
    
    val_path = os.path.join(report_dir, "live_weather_validation_report.json")
    with open(val_path, "w") as f:
        json.dump(era5_results, f, indent=4)
        
    qa_path = os.path.join(report_dir, "weather_validation_report.json")
    with open(qa_path, "w") as f:
        json.dump(qa_results, f, indent=4)

    print(f"\nGenerated {val_path}")
    print(f"Generated {qa_path}")

if __name__ == "__main__":
    test_cities()

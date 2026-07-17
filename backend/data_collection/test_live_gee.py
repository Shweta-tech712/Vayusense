import os
import sys
import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.config.gee_config import initialize_gee
from backend.services.live_satellite_service import live_satellite_service

def test_live_satellite():
    print("Initializing Google Earth Engine...")
    success, msg = initialize_gee()
    if not success:
        print(f"GEE Auth failed: {msg}")
        return
        
    locations = {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Pune": (18.5204, 73.8567),
        "Bengaluru": (12.9716, 77.5946)
    }
    
    validation_results = {}
    
    for loc_name, (lat, lon) in locations.items():
        print(f"\nQuerying live satellite data for {loc_name} ({lat}, {lon})...")
        results = live_satellite_service.get_live_satellite_data(lat, lon, radius_km=10.0)
        print(f"Acquisition Date: {results.get('acquisition_date', 'Unknown')}")
        print(f"Results: {results}")
        validation_results[loc_name] = results
        
    import json
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "live_satellite_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(validation_results, f, indent=4)
    print(f"\nSaved validation report to {report_path}")

if __name__ == "__main__":
    test_live_satellite()

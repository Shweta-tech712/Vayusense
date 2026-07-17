import sys
import os
import json
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath('c:/Users/Vikram Kharat/Desktop/ISRO'))
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService

# Force loading
ModelService.instance().load()
ps = PredictionService()

cities = ["Delhi", "Mumbai", "Pune", "Kochi", "Bengaluru"]

location_prediction_debug_report = {}
prediction_input_comparison = {}
prediction_output_comparison = {}
dataset_lookup_report = {}

for city in cities:
    try:
        print(f"Testing {city}...")
        loc_meta = ps._resolve_location(city, None, None)
        lat, lon = loc_meta["latitude"], loc_meta["longitude"]
        
        # 1. KDTree lookup
        dist, idx = ps._kdtree.query([lat, lon])
        unique_coords = ps._fused_df[["latitude", "longitude"]].drop_duplicates().values
        near_lat, near_lon = unique_coords[idx]
        
        dataset_lookup_report[city] = {
            "search_lat": lat,
            "search_lon": lon,
            "matched_lat": float(near_lat),
            "matched_lon": float(near_lon),
            "distance": float(dist),
            "kdtree_idx": int(idx)
        }
        
        # 2. Sequence
        raw_seq = ps._build_sequence(lat, lon)
        prediction_input_comparison[city] = raw_seq.tolist()
        
        # 3. Model Output
        scaled = ps._model_svc.transform(raw_seq)
        preds = ps._model_svc.predict(scaled)
        
        raw_output = ps._model_svc._model.predict(scaled, verbose=0)
        if isinstance(raw_output, list):
            raw_tensor = [o.tolist() if isinstance(o, np.ndarray) else o for o in raw_output]
        else:
            raw_tensor = raw_output.tolist()
            
        prediction_output_comparison[city] = {
            "post_processed": preds,
            "raw_tensor": raw_tensor
        }
        
        location_prediction_debug_report[city] = {
            "geocoding": loc_meta,
            "matched_coords": [float(near_lat), float(near_lon)],
            "sequence_shape": raw_seq.shape,
            "final_aqi": preds["AQI"],
            "final_pm25": preds["PM25"]
        }
    except Exception as e:
        print(f"Error on {city}: {e}")
        traceback.print_exc()

with open('location_prediction_debug_report.json', 'w') as f:
    json.dump(location_prediction_debug_report, f, indent=4)
    
with open('prediction_input_comparison.json', 'w') as f:
    json.dump(prediction_input_comparison, f, indent=4)
    
with open('prediction_output_comparison.json', 'w') as f:
    json.dump(prediction_output_comparison, f, indent=4)
    
with open('dataset_lookup_report.json', 'w') as f:
    json.dump(dataset_lookup_report, f, indent=4)

print("Done generating debug reports.")

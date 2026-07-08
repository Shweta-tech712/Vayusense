import os
import sys
import json
import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spatial_matcher")

class SpatialMatcher:
    def __init__(self, stations_path=None, satellite_path=None):
        self.stations_path = stations_path or os.path.join(os.path.dirname(__file__), "..", "config", "cpcb_stations.json")
        self.satellite_path = satellite_path or os.path.join(os.path.dirname(__file__), "..", "datasets", "processed", "sentinel5p_merged.csv")
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate geographical distance in kilometers between two points"""
        R = 6371.0 # Radius of Earth
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c

    def match_coordinates(self):
        logger.info("Initializing spatiotemporal alignment between CPCB ground stations and Sentinel-5P grid...")
        
        if not os.path.exists(self.stations_path):
            logger.error(f"Stations configuration not found at {self.stations_path}")
            return None
        if not os.path.exists(self.satellite_path):
            logger.error(f"Merged Sentinel-5P dataset not found at {self.satellite_path}")
            return None
            
        # Load datasets
        with open(self.stations_path, "r") as f:
            stations = json.load(f)
        sat_df = pd.read_csv(self.satellite_path)
        
        if sat_df.empty:
            logger.warning("Sentinel-5P dataset is empty. Cannot match coordinates.")
            return None
            
        # Extract unique grid coordinates from Sentinel-5P dataset
        unique_sat_coords = sat_df[["latitude", "longitude"]].drop_duplicates().values
        
        matches = []
        for s in stations:
            s_lat = s["latitude"]
            s_lon = s["longitude"]
            
            # Find closest pixel coordinate
            min_dist = float('inf')
            closest_pixel = None
            
            for pixel in unique_sat_coords:
                p_lat, p_lon = pixel[0], pixel[1]
                dist = self.haversine_distance(s_lat, s_lon, p_lat, p_lon)
                if dist < min_dist:
                    min_dist = dist
                    closest_pixel = (p_lat, p_lon)
                    
            logger.info(f"Station '{s['station_name']}' matched with nearest pixel ({closest_pixel[0]}, {closest_pixel[1]}) - Distance: {min_dist:.2f} km")
            
            matches.append({
                "station_id": s["station_id"],
                "station_name": s["station_name"],
                "station_latitude": s_lat,
                "station_longitude": s_lon,
                "matched_pixel_latitude": closest_pixel[0],
                "matched_pixel_longitude": closest_pixel[1],
                "distance_km": float(min_dist)
            })
            
        # Save mapping details
        output_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "processed")
        os.makedirs(output_dir, exist_ok=True)
        mapping_file = os.path.join(output_dir, "station_satellite_mapping.json")
        with open(mapping_file, "w") as f:
            json.dump(matches, f, indent=2)
            
        logger.info(f"Saved spatiotemporal coordinates mapping dictionary to {mapping_file}")
        return matches

if __name__ == "__main__":
    matcher = SpatialMatcher()
    matcher.match_coordinates()

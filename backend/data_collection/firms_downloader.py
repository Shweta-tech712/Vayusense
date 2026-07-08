import os
import sys
import json
import logging
import datetime
import argparse
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# Ensure project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "firms_pipeline.log"))
    ]
)
logger = logging.getLogger("firms_pipeline")

class FIRMSDownloader:
    def __init__(self, config_path=None, stations_path=None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config", "firms_config.json")
        self.stations_path = stations_path or os.path.join(os.path.dirname(__file__), "..", "config", "cpcb_stations.json")
        self.load_configs()
        
    def load_configs(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            with open(self.stations_path, "r") as f:
                self.stations = json.load(f)
            logger.info("Loaded FIRMS config and station metadata successfully.")
        except Exception as e:
            logger.error(f"Failed to load config files: {e}")
            raise

    def get_map_key(self):
        # First check environment variables, then the config file
        key = os.environ.get("FIRMS_MAP_KEY")
        if not key:
            key = self.config.get("map_key")
        if not key:
            raise ValueError("missing MAP_KEY: Please set the FIRMS_MAP_KEY environment variable or configure 'map_key' in firms_config.json.")
        return key

    def get_season(self, month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Pre-Monsoon"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Post-Monsoon"

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0 # km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)
        y = np.sin(dlon_rad) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360

    def download_fire_data(self, start_date, end_date):
        map_key = self.get_map_key()
        bounds = self.config.get("spatial_bounds", {"west": 68.1, "south": 8.4, "east": 97.4, "north": 37.6})
        bounds_str = f"{bounds['west']},{bounds['south']},{bounds['east']},{bounds['north']}"
        
        # Sources configuration
        sources = {
            "MODIS": "MODIS_NRT",
            "VIIRS": "VIIRS_SNPP_NRT"
        }
        
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "raw", "firms")
        os.makedirs(raw_dir, exist_ok=True)
        
        extracted_files = {}
        
        # We query NASA FIRMS day-by-day
        delta = end_date - start_date
        total_days = delta.days + 1
        
        for sat, source_code in sources.items():
            logger.info(f"Retrieving active fire data for {sat} ({source_code}) over {total_days} days...")
            all_records = []
            
            for i in range(total_days):
                curr_date = start_date + datetime.timedelta(days=i)
                date_str = curr_date.strftime("%Y-%m-%d")
                
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source_code}/{bounds_str}/1/{date_str}"
                
                try:
                    logger.info(f"Querying: {url.replace(map_key, '***')}")
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 401 or response.status_code == 403:
                        raise ValueError(f"API unavailable: Unauthorized access or invalid MAP_KEY. Status code {response.status_code}")
                    elif response.status_code != 200:
                        raise RuntimeError(f"API unavailable: NASA FIRMS server returned status {response.status_code}")
                        
                    content = response.text.strip()
                    if not content or "latitude" not in content.lower():
                        logger.warning(f"No active fires found or empty response for {sat} on {date_str}")
                        continue
                        
                    lines = content.splitlines()
                    df_day = pd.read_csv(requests.compat.StringIO(content))
                    if not df_day.empty:
                        df_day["acq_date"] = date_str
                        all_records.append(df_day)
                        
                except requests.exceptions.RequestException as re:
                    raise ConnectionError(f"network issue: Failed to connect to NASA FIRMS server. Error: {re}")
            
            if all_records:
                df_sat = pd.concat(all_records, ignore_index=True)
            else:
                df_sat = pd.DataFrame(columns=[
                    "latitude", "longitude", "bright_t31", "scan", "track", "acq_date", "acq_time", 
                    "satellite", "instrument", "confidence", "version", "bright_ti5", "frp"
                ])
                
            output_file = os.path.join(raw_dir, f"{sat}.csv")
            df_sat.to_csv(output_file, index=False)
            logger.info(f"Saved raw {sat} dataset to {output_file} ({len(df_sat)} rows)")
            extracted_files[sat] = output_file
            
        return extracted_files

    def clean_fire_data(self, extracted_files):
        logger.info("Cleaning raw active fire datasets...")
        clean_dfs = []
        
        bounds = self.config.get("spatial_bounds", {"west": 68.1, "south": 8.4, "east": 97.4, "north": 37.6})
        conf_thresh = self.config.get("confidence_threshold", 70)
        
        for sat, file_path in extracted_files.items():
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            if df.empty:
                continue
                
            # Filter inside geographical bounds
            df = df[
                (df["latitude"] >= bounds["south"]) & (df["latitude"] <= bounds["north"]) &
                (df["longitude"] >= bounds["west"]) & (df["longitude"] <= bounds["east"])
            ]
            
            # Map columns to unified schema
            # Required columns: latitude, longitude, date, time, satellite, instrument, confidence, brightness_temperature, FRP
            df_clean = pd.DataFrame()
            df_clean["latitude"] = df["latitude"]
            df_clean["longitude"] = df["longitude"]
            df_clean["date"] = df["acq_date"]
            df_clean["time"] = df["acq_time"]
            df_clean["satellite"] = df["satellite"].fillna(sat)
            
            # Map instrument
            if "instrument" in df.columns:
                df_clean["instrument"] = df["instrument"]
            else:
                df_clean["instrument"] = "MODIS" if sat == "MODIS" else "VIIRS"
                
            # Map confidence: convert categorical (e.g. 'h', 'n', 'l') to numeric if necessary
            if "confidence" in df.columns:
                # VIIRS uses string flags: 'h' -> 90, 'n' -> 50, 'l' -> 20
                if df["confidence"].dtype == object:
                    conf_map = {"h": 90, "n": 50, "l": 20, "high": 90, "nominal": 50, "low": 20}
                    df_clean["confidence"] = df["confidence"].astype(str).str.lower().map(conf_map).fillna(50)
                else:
                    df_clean["confidence"] = df["confidence"]
            else:
                df_clean["confidence"] = 80 # Default
                
            # Filter by confidence threshold
            df_clean = df_clean[df_clean["confidence"] >= conf_thresh]
            
            # Map brightness temperature (Kelvin)
            if "bright_t31" in df.columns:
                df_clean["brightness_temperature"] = df["bright_t31"]
            elif "bright_ti5" in df.columns:
                df_clean["brightness_temperature"] = df["bright_ti5"]
            else:
                df_clean["brightness_temperature"] = np.nan
                
            # Map FRP
            if "frp" in df.columns:
                df_clean["frp"] = df["frp"]
            else:
                df_clean["frp"] = 0.0
                
            clean_dfs.append(df_clean)
            
        if not clean_dfs:
            logger.warning("No clean active fire records retrieved.")
            return pd.DataFrame()
            
        merged_df = pd.concat(clean_dfs, ignore_index=True)
        merged_df = merged_df.sort_values(by=["date", "latitude", "longitude"]).reset_index(drop=True)
        
        # Deduplicate
        merged_df = merged_df.drop_duplicates(subset=["date", "latitude", "longitude"])
        
        logger.info(f"Merged active fires shape after quality filtering: {merged_df.shape}")
        return merged_df

    def calculate_fire_features(self, merged_df):
        logger.info("Performing feature engineering on active fires...")
        if merged_df.empty:
            return pd.DataFrame()
            
        # Load processed ERA5 and Sentinel-5P datasets to extract wind and HCHO data
        processed_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "processed")
        era5_file = os.path.join(processed_dir, "era5_processed.csv")
        s5p_file = os.path.join(processed_dir, "sentinel5p_merged.csv")
        
        era5_df = pd.read_csv(era5_file) if os.path.exists(era5_file) else pd.DataFrame()
        s5p_df = pd.read_csv(s5p_file) if os.path.exists(s5p_file) else pd.DataFrame()
        
        # Build hourly or daily indexes for quick wind and HCHO lookups
        wind_dict = {}
        if not era5_df.empty:
            for _, r in era5_df.iterrows():
                # Key: (date, round_lat_1dec, round_lon_1dec)
                k = (r["date"], round(r["latitude"], 1), round(r["longitude"], 1))
                wind_dict[k] = (r.get("wind_speed", 2.0), r.get("wind_direction", 270.0))
                
        hcho_dict = {}
        if not s5p_df.empty:
            for _, r in s5p_df.iterrows():
                k = (r["date"], round(r["latitude"], 1), round(r["longitude"], 1))
                hcho_dict[k] = r.get("HCHO", 0.0)

        rows = []
        unique_dates = merged_df["date"].unique()
        
        for date_str in unique_dates:
            df_day = merged_df[merged_df["date"] == date_str]
            day_fires_coords = df_day[["latitude", "longitude"]].values
            day_frp = df_day["frp"].values
            day_conf = df_day["confidence"].values
            
            date_parsed = pd.to_datetime(date_str)
            yr = date_parsed.year
            mn = date_parsed.month
            season = self.get_season(mn)
            
            # 1. Classify Burning Season
            # Punjab-Haryana stubble: October-November
            # Forest fire season: March-May
            if mn in [10, 11]:
                burning_season = "Kharif Crop Burning"
            elif mn in [3, 4, 5]:
                burning_season = "Forest Fire Season"
            else:
                burning_season = "Off-Season"
                
            for stn in self.stations:
                stn_lat = stn["latitude"]
                stn_lon = stn["longitude"]
                
                # Default features if no fires exist on this day
                fire_count = 0
                fire_density = 0.0
                frp_val = 0.0
                avg_frp = 0.0
                max_frp = 0.0
                conf_val = 0.0
                min_dist = 999.0
                
                fire_10k = 0
                fire_50k = 0
                fire_100k = 0
                
                source_region = "Other"
                transport_influence = 0.0
                
                if len(day_fires_coords) > 0:
                    # Calculate distances
                    distances = self.haversine_distance(stn_lat, stn_lon, day_fires_coords[:, 0], day_fires_coords[:, 1])
                    min_idx = np.argmin(distances)
                    min_dist = float(distances[min_idx])
                    
                    # Buffer counts
                    fire_10k = int(np.sum(distances <= 10.0))
                    fire_50k = int(np.sum(distances <= 50.0))
                    fire_100k = int(np.sum(distances <= 100.0))
                    
                    # Fire counts and density in the local neighborhood (e.g. within 100km)
                    nearby_mask = distances <= 100.0
                    nearby_fires_count = np.sum(nearby_mask)
                    
                    if nearby_fires_count > 0:
                        fire_count = int(nearby_fires_count)
                        # Density per 1000 km2 within 100km radius (area is pi * 100^2 = 31415 km2)
                        fire_density = float((fire_count / 31415.9) * 1000.0)
                        
                        nearby_frps = day_frp[nearby_mask]
                        nearby_confs = day_conf[nearby_mask]
                        
                        avg_frp = float(np.mean(nearby_frps))
                        max_frp = float(np.max(nearby_frps))
                        frp_val = avg_frp
                        conf_val = float(np.mean(nearby_confs))
                        
                        # Source Region Classification of the nearest active fire
                        near_lat = day_fires_coords[min_idx][0]
                        near_lon = day_fires_coords[min_idx][1]
                        
                        if 29.0 <= near_lat <= 32.5 and 73.5 <= near_lon <= 77.5:
                            source_region = "Punjab-Haryana Crop Burning"
                        elif 20.0 <= near_lat <= 29.0 and near_lon >= 88.0:
                            source_region = "North-East Biomass Burning"
                        elif 15.0 <= near_lat <= 24.0 and 73.0 <= near_lon <= 84.0:
                            source_region = "Forest Fire Region"
                        else:
                            source_region = "Other"
                            
                        # Transport Influence Score Formulation
                        # Retrieve ERA5 wind at station
                        stn_k = (date_str, round(stn_lat, 1), round(stn_lon, 1))
                        wind_speed, wind_direction = wind_dict.get(stn_k, (2.0, 270.0))
                        
                        # Calculate bearing from fire to station (downwind flow)
                        bearing_fire_to_station = self.calculate_bearing(near_lat, near_lon, stn_lat, stn_lon)
                        
                        # Compare bearing with wind direction
                        angle_diff = min(abs(bearing_fire_to_station - wind_direction), 360 - abs(bearing_fire_to_station - wind_direction))
                        
                        # Cosine similarity decay: positive if wind points towards station
                        alignment = max(0.0, np.cos(np.radians(angle_diff)))
                        
                        # Exponential decay with distance
                        dist_decay = np.exp(-min_dist / 150.0)
                        
                        # Influence potential
                        transport_influence = float(avg_frp * alignment * dist_decay * (wind_speed / 2.0))
                        
                # 2. Fire Severity Index Category
                sev_score = (min(1.0, fire_count / 20.0) * 0.4) + (min(1.0, max_frp / 100.0) * 0.4) + (conf_val / 100.0 * 0.2)
                if fire_count > 0 and sev_score >= 0.6:
                    fire_severity = "High"
                elif fire_count > 0 and sev_score >= 0.25:
                    fire_severity = "Medium"
                else:
                    fire_severity = "Low"
                    
                rows.append({
                    "date": date_str,
                    "year": yr,
                    "month": mn,
                    "season": season,
                    "station_id": stn["station_id"],
                    "latitude": stn_lat,
                    "longitude": stn_lon,
                    "fire_count": fire_count,
                    "fire_density": fire_density,
                    "FRP": frp_val,
                    "average_FRP": avg_frp,
                    "maximum_FRP": max_frp,
                    "confidence": conf_val,
                    "nearest_fire_distance": min_dist,
                    "burning_season": burning_season,
                    "source_region": source_region,
                    "fire_severity_index": fire_severity,
                    "transport_influence_score": transport_influence,
                    # Temp buffer distance flags
                    "fire_within_10km": fire_10k,
                    "fire_within_50km": fire_50k,
                    "fire_within_100km": fire_100k
                })
                
        df_feats = pd.DataFrame(rows)
        if df_feats.empty:
            return df_feats
            
        # 3. Calculate HCHO Lag features
        df_feats = df_feats.sort_values(by=["station_id", "date"]).reset_index(drop=True)
        
        df_feats["fire_count_lag_1"] = df_feats.groupby("station_id")["fire_count"].shift(1).bfill().astype(int)
        df_feats["fire_count_lag_3"] = df_feats.groupby("station_id")["fire_count"].shift(3).bfill().astype(int)
        
        df_feats["FRP_lag_1"] = df_feats.groupby("station_id")["average_FRP"].shift(1).bfill()
        df_feats["FRP_lag_3"] = df_feats.groupby("station_id")["average_FRP"].shift(3).bfill()
        
        return df_feats

    def save_processed_files(self, df_feats, merged_raw_df):
        processed_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "processed")
        geojson_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "geojson")
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(geojson_dir, exist_ok=True)
        
        # Save processed CSV
        cols = [
            "date", "year", "month", "season",
            "latitude", "longitude",
            "fire_count", "fire_density",
            "FRP", "average_FRP", "maximum_FRP",
            "confidence",
            "nearest_fire_distance",
            "fire_count_lag_1", "fire_count_lag_3",
            "FRP_lag_1", "FRP_lag_3",
            "burning_season",
            "source_region",
            "fire_severity_index",
            "transport_influence_score"
        ]
        df_processed = df_feats[cols]
        processed_file = os.path.join(processed_dir, "firms_processed.csv")
        df_processed.to_csv(processed_file, index=False)
        logger.info(f"Saved final processed active fire dataset to {processed_file}")
        
        # Save GeoJSON
        geojson_file = os.path.join(geojson_dir, "fire_hotspots.geojson")
        if not merged_raw_df.empty:
            gdf = gpd.GeoDataFrame(
                merged_raw_df,
                geometry=gpd.points_from_xy(merged_raw_df["longitude"], merged_raw_df["latitude"]),
                crs="EPSG:4326"
            )
            # Reorder or select columns for leaflet popup visualization
            gdf_out = gdf[["date", "time", "satellite", "confidence", "frp", "brightness_temperature", "geometry"]]
            gdf_out.to_file(geojson_file, driver="GeoJSON")
            logger.info(f"Saved active fire hotspots GeoJSON layer to {geojson_file}")
        else:
            # Empty geojson structure
            with open(geojson_file, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
            logger.warning("Merged raw active fires dataframe was empty. Exported empty GeoJSON layer.")
            
        return processed_file

    def validate_fire_dataset(self, processed_file, merged_raw_df):
        logger.info("Running active fires dataset validation checklist...")
        df = pd.read_csv(processed_file)
        
        total_fires = len(merged_raw_df)
        high_conf_fires = int(np.sum(merged_raw_df["confidence"] >= 80)) if not merged_raw_df.empty else 0
        
        season_analysis = {}
        if not df.empty:
            season_counts = df.groupby("burning_season")["fire_count"].sum().to_dict()
            for k, v in season_counts.items():
                season_analysis[k] = int(v)
                
        frp_stats = {}
        if not merged_raw_df.empty:
            frp_stats = {
                "min": float(merged_raw_df["frp"].min()),
                "max": float(merged_raw_df["frp"].max()),
                "mean": float(merged_raw_df["frp"].mean())
            }
        else:
            frp_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
            
        missing_values = {
            "nearest_fire_distance": int(df["nearest_fire_distance"].isna().sum()),
            "transport_influence_score": int(df["transport_influence_score"].isna().sum())
        }
        
        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pipeline": "NASA FIRMS Fire Activity Integration Portal",
            "total_fires": total_fires,
            "high_confidence_fires": high_conf_fires,
            "season_analysis": season_analysis,
            "frp_statistics": frp_stats,
            "missing_values": missing_values,
            "status": "Validated"
        }
        
        report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, "firms_report.json")
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Firms validation report saved to {report_file}")

    def run(self, is_test=False):
        start_str = self.config["start_date"]
        end_str = self.config["end_date"]
        if is_test:
            start_str = "2023-01-01"
            end_str = "2023-01-07"
            
        start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
        
        extracted_files = self.download_fire_data(start_dt, end_dt)
        merged_raw_df = self.clean_fire_data(extracted_files)
        df_feats = self.calculate_fire_features(merged_raw_df)
        if not df_feats.empty:
            processed_file = self.save_processed_files(df_feats, merged_raw_df)
            self.validate_fire_dataset(processed_file, merged_raw_df)
        else:
            logger.error("Failed to generate features. Output processed files were not generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA FIRMS Active Fire Downloader & Pipeline")
    parser.add_argument("--test", action="store_true", help="Run downloader in test mode (7 days)")
    args = parser.parse_args()
    
    downloader = FIRMSDownloader()
    downloader.run(is_test=args.test)

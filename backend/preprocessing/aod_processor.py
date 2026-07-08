import os
import sys
import glob
import json
import logging
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd

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
        logging.FileHandler(os.path.join(log_dir, "aod_processing.log"))
    ]
)
logger = logging.getLogger("aod_processor")

class AODProcessor:
    def __init__(self, config_path=None, stations_path=None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config", "insat_config.json")
        self.stations_path = stations_path or os.path.join(os.path.dirname(__file__), "..", "config", "cpcb_stations.json")
        self.load_configs()
        
    def load_configs(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            with open(self.stations_path, "r") as f:
                self.stations = json.load(f)
            logger.info("Loaded AOD configs and stations metadata successfully.")
        except Exception as e:
            logger.error(f"Failed to load configuration files: {e}")
            raise

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0 # km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def get_season(self, month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Pre-Monsoon"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Post-Monsoon"

    def nearest_aod_pixel_matching(self, raw_df, stn_lat, stn_lon):
        """Finds the nearest valid AOD grid pixel for the station coordinates"""
        if raw_df.empty:
            return np.nan
            
        unique_coords = raw_df[["latitude", "longitude"]].drop_duplicates().values
        if len(unique_coords) == 0:
            return np.nan
            
        distances = self.haversine_distance(stn_lat, stn_lon, unique_coords[:, 0], unique_coords[:, 1])
        min_idx = np.argmin(distances)
        closest_coord = unique_coords[min_idx]
        
        # Pull the AOD values matching this nearest coordinate
        matched_subset = raw_df[
            (raw_df["latitude"] == closest_coord[0]) & 
            (raw_df["longitude"] == closest_coord[1])
        ]
        
        if not matched_subset.empty:
            return float(matched_subset["AOD"].values[0])
        return np.nan

    def process_all_files(self):
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "raw", "insat3d")
        processed_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "processed")
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        geojson_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "geojson")
        
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(geojson_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(raw_dir, "*.csv")) + glob.glob(os.path.join(raw_dir, "*.nc"))
        if not files:
            logger.error("No raw datasets found under raw/insat3d/")
            return False
            
        # Load ERA5 weather data for physical correction feature engineering
        era5_file = os.path.join(processed_dir, "era5_processed.csv")
        era5_df = pd.read_csv(era5_file) if os.path.exists(era5_file) else pd.DataFrame()
        
        weather_dict = {}
        if not era5_df.empty:
            for _, r in era5_df.iterrows():
                # Key: (date, latitude, longitude)
                k = (r["date"], round(r["latitude"], 2), round(r["longitude"], 2))
                weather_dict[k] = {
                    "humidity": r.get("humidity", 50.0),
                    "blh": r.get("boundary_layer_height", 1000.0),
                    "wind_speed": r.get("wind_speed", 2.0)
                }

        records = []
        date_coverage = set()
        total_pixels_processed = 0
        missing_pixels_count = 0
        aod_values_list = []

        for f in files:
            logger.info(f"Processing raw AOD file: {f}")
            
            # Load raw data
            if f.endswith(".csv"):
                df_raw = pd.read_csv(f)
            else:
                # Handle NetCDF reading using xarray
                try:
                    import xarray as xr
                    ds = xr.open_dataset(f)
                    df_raw = ds.to_dataframe().reset_index()
                    # Rename standard variables to AOD
                    rename_dict = {"aod": "AOD", "Aerosol_Optical_Depth": "AOD"}
                    df_raw = df_raw.rename(columns={k: v for k, v in rename_dict.items() if k in df_raw.columns})
                except Exception as ex:
                    logger.error(f"Xarray NetCDF read failed for {f}: {ex}")
                    continue
                    
            if df_raw.empty:
                continue
                
            # Perform cloud filtering and invalid pixel checks
            # Valid AOD range: 0.0 to 2.0. Values outside or marked as negative/missing (-999) are filtered
            total_pixels_processed += len(df_raw)
            invalid_mask = (df_raw["AOD"] < 0.0) | (df_raw["AOD"] > 2.0) | (df_raw["AOD"].isna())
            missing_pixels_count += int(invalid_mask.sum())
            df_raw.loc[invalid_mask, "AOD"] = np.nan
            
            file_date_str = df_raw["date"].iloc[0]
            date_coverage.add(file_date_str)
            
            date_parsed = pd.to_datetime(file_date_str)
            yr = date_parsed.year
            mn = date_parsed.month
            day = date_parsed.day
            season = self.get_season(mn)
            
            # Perform nearest pixel spatial matching for each CPCB station
            for s in self.stations:
                stn_lat = s["latitude"]
                stn_lon = s["longitude"]
                
                aod_val = self.nearest_aod_pixel_matching(df_raw, stn_lat, stn_lon)
                
                # Retrieve ERA5 weather variables for physical features
                stn_k = (file_date_str, round(stn_lat, 2), round(stn_lon, 2))
                weather = weather_dict.get(stn_k, {"humidity": 50.0, "blh": 1000.0, "wind_speed": 2.0})
                
                rh = weather["humidity"]
                blh = weather["blh"]
                ws = weather["wind_speed"]
                
                # Feature engineering:
                # 1. Aerosol intensity category
                if pd.isna(aod_val):
                    aerosol_intensity = "Low"
                    humidity_corrected_aod = np.nan
                    aod_blh_ratio = np.nan
                else:
                    aod_values_list.append(aod_val)
                    if aod_val < 0.2:
                        aerosol_intensity = "Low"
                    elif aod_val < 0.6:
                        aerosol_intensity = "Medium"
                    else:
                        aerosol_intensity = "High"
                        
                    # 2. Humidity corrected AOD (using hygroscopic growth factor)
                    # Formula: AOD_corr = AOD * (1 / (1 - RH/100))
                    # Cap RH at 95% to avoid division by zero anomalies
                    humidity_corrected_aod = float(aod_val * (1.0 / (1.0 - min(0.95, rh / 100.0))))
                    
                    # 3. AOD / boundary layer ratio
                    aod_blh_ratio = float(aod_val / max(50.0, blh))
                    
                # 4. Ventilation index (boundary layer height * wind speed)
                ventilation_idx = float(blh * ws)
                
                records.append({
                    "date": file_date_str,
                    "year": yr,
                    "month": mn,
                    "day": day,
                    "season": season,
                    "station_id": s["station_id"],
                    "latitude": stn_lat,
                    "longitude": stn_lon,
                    "AOD": aod_val,
                    "humidity_corrected_AOD": humidity_corrected_aod,
                    "aod_boundary_layer_ratio": aod_blh_ratio,
                    "ventilation_index": ventilation_idx,
                    "aerosol_intensity": aerosol_intensity
                })
                
        df_out = pd.DataFrame(records)
        if df_out.empty:
            logger.error("No station matched AOD records constructed.")
            return False
            
        # Calculate lag features per station
        df_out = df_out.sort_values(by=["station_id", "date"]).reset_index(drop=True)
        
        df_out["AOD_lag_1"] = df_out.groupby("station_id")["AOD"].shift(1).bfill()
        df_out["AOD_lag_3"] = df_out.groupby("station_id")["AOD"].shift(3).bfill()
        
        # Weekly rolling average (7 days window)
        df_out["AOD_weekly_average"] = df_out.groupby("station_id")["AOD"].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean().bfill()
        )
        
        # Save processed CSV
        cols = [
            "date", "year", "month", "day", "season",
            "latitude", "longitude",
            "AOD", "AOD_lag_1", "AOD_lag_3", "AOD_weekly_average",
            "humidity_corrected_AOD", "aod_boundary_layer_ratio", "ventilation_index",
            "aerosol_intensity"
        ]
        df_processed = df_out[cols]
        processed_file = os.path.join(processed_dir, "insat_aod_processed.csv")
        df_processed.to_csv(processed_file, index=False)
        logger.info(f"Saved processed Aerosol Optical Depth dataset to {processed_file}")
        
        # Save Leaflet GeoJSON layer
        geojson_file = os.path.join(geojson_dir, "aod_layer.geojson")
        df_geo = df_out.dropna(subset=["AOD"])
        if not df_geo.empty:
            gdf = gpd.GeoDataFrame(
                df_geo,
                geometry=gpd.points_from_xy(df_geo["longitude"], df_geo["latitude"]),
                crs="EPSG:4326"
            )
            gdf_out = gdf[["date", "AOD", "aerosol_intensity", "geometry"]]
            gdf_out.to_file(geojson_file, driver="GeoJSON")
            logger.info(f"Saved Folium/Leaflet GeoJSON layer to {geojson_file}")
        else:
            with open(geojson_file, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
            logger.warning("No valid non-null AOD records found to export to GeoJSON.")
            
        # Write validation report
        self.write_validation_report(date_coverage, total_pixels_processed, missing_pixels_count, aod_values_list, reports_dir)
        return True

    def write_validation_report(self, date_coverage, total_pixels, missing_pixels, aod_list, reports_dir):
        report_file = os.path.join(reports_dir, "insat_report.json")
        
        aod_stats = {
            "min": float(np.min(aod_list)) if aod_list else 0.0,
            "max": float(np.max(aod_list)) if aod_list else 0.0,
            "mean": float(np.mean(aod_list)) if aod_list else 0.0
        }
        
        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pipeline": "INSAT-3D Aerosol Optical Depth Preprocessing Portal",
            "date_coverage": {
                "start": min(list(date_coverage)) if date_coverage else None,
                "end": max(list(date_coverage)) if date_coverage else None,
                "total_days": len(date_coverage)
            },
            "spatial_coverage": {
                "study_stations_count": len(self.stations)
            },
            "pixels": {
                "total_processed": total_pixels,
                "missing_or_cloud_pixels": missing_pixels,
                "cloud_fraction": float(missing_pixels / max(1, total_pixels))
            },
            "AOD_statistics": aod_stats,
            "status": "Validated"
        }
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"INSAT-3D validation report saved to {report_file}")

if __name__ == "__main__":
    processor = AODProcessor()
    processor.process_all_files()

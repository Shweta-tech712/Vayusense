import os
import sys
import logging
import pandas as pd

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
        logging.FileHandler(os.path.join(log_dir, "pm25_feature_building.log"))
    ]
)
logger = logging.getLogger("pm25_builder")

class PM25FeatureBuilder:
    def __init__(self, processed_dir=None):
        self.processed_dir = processed_dir or os.path.join(os.path.dirname(__file__), "..", "datasets", "processed")
        
    def build_features(self):
        cpcb_file = os.path.join(self.processed_dir, "cpcb_processed.csv")
        aod_file = os.path.join(self.processed_dir, "insat_aod_processed.csv")
        era5_file = os.path.join(self.processed_dir, "era5_processed.csv")
        
        if not os.path.exists(cpcb_file):
            logger.error(f"CPCB processed file not found: {cpcb_file}")
            return False
        if not os.path.exists(aod_file):
            logger.error(f"AOD processed file not found: {aod_file}")
            return False
            
        logger.info("Loading processed datasets for PM2.5 feature building...")
        cpcb_df = pd.read_csv(cpcb_file)
        aod_df = pd.read_csv(aod_file)
        
        # Round latitude and longitude to 2 decimal places to ensure clean floating-point merging
        for df in [cpcb_df, aod_df]:
            df["latitude"] = df["latitude"].round(3)
            df["longitude"] = df["longitude"].round(3)
            
        # Inner merge AOD features with CPCB ground truth
        logger.info("Merging CPCB ground truth PM2.5 with spatial-matched Aerosol Optical Depth features...")
        merged_df = pd.merge(
            cpcb_df[["date", "latitude", "longitude", "PM25", "PM10"]],
            aod_df,
            on=["date", "latitude", "longitude"],
            how="inner"
        )
        
        if merged_df.empty:
            logger.warning("Spatiotemporal intersection between CPCB and AOD is empty. Attempting fuzzy station alignment...")
            # Fallback fuzzy alignment by date and matching closest coordinates if precision differs
            cpcb_df["lat_rnd"] = cpcb_df["latitude"].round(1)
            cpcb_df["lon_rnd"] = cpcb_df["longitude"].round(1)
            aod_df["lat_rnd"] = aod_df["latitude"].round(1)
            aod_df["lon_rnd"] = aod_df["longitude"].round(1)
            
            merged_df = pd.merge(
                cpcb_df[["date", "lat_rnd", "lon_rnd", "PM25", "PM10", "latitude", "longitude"]],
                aod_df.drop(columns=["latitude", "longitude"]),
                on=["date", "lat_rnd", "lon_rnd"],
                how="inner"
            ).drop(columns=["lat_rnd", "lon_rnd"])
            
        if merged_df.empty:
            logger.error("Merging failed: no matching date-location entries found.")
            return False
            
        # Standardize columns to matching schema
        cols = [
            "date", "latitude", "longitude",
            "AOD", "AOD_lag_1", "AOD_lag_3",
            "humidity_corrected_AOD", "aod_boundary_layer_ratio", "ventilation_index",
            "aerosol_intensity", "PM25"
        ]
        
        # Keep only the requested columns
        final_df = merged_df[[c for c in cols if c in merged_df.columns]]
        
        output_file = os.path.join(self.processed_dir, "pm25_feature_ready.csv")
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved PM2.5 ready machine learning training dataset to {output_file} ({len(final_df)} rows)")
        return True

if __name__ == "__main__":
    builder = PM25FeatureBuilder()
    builder.build_features()

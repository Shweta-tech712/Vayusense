import os
import sys
import glob
import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sentinel5p_merger")

class Sentinel5PMerger:
    def __init__(self, raw_dir=None, processed_dir=None):
        self.raw_dir = raw_dir or os.path.join(os.path.dirname(__file__), "..", "datasets", "raw", "sentinel5p")
        self.processed_dir = processed_dir or os.path.join(os.path.dirname(__file__), "..", "datasets", "processed")
        
    def get_season(self, month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Pre-Monsoon"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Post-Monsoon"

    def merge(self):
        logger.info(f"Scanning raw partitions directory: {self.raw_dir}")
        pollutants = ["NO2", "SO2", "CO", "O3", "HCHO"]
        
        pollutant_dfs = {}
        
        for pol in pollutants:
            pol_path = os.path.join(self.raw_dir, pol, "*.csv")
            files = glob.glob(pol_path)
            if not files:
                logger.warning(f"No partition files found for pollutant: {pol}")
                continue
                
            logger.info(f"Loading {len(files)} files for {pol}...")
            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    if not df.empty:
                        # Apply strict quality filtering during preprocessing
                        qa_thresh = 0.75 if pol == "NO2" else 0.50
                        
                        # Clean: filter by QA and drop invalid or missing pixels
                        df_clean = df[(df["qa_value"] >= qa_thresh) & (df["value"].notna())].copy()
                        
                        # Extract relevant columns
                        df_subset = df_clean[["date", "latitude", "longitude", "value"]].copy()
                        df_subset = df_subset.rename(columns={"value": pol})
                        dfs.append(df_subset)
                except Exception as e:
                    logger.error(f"Error reading file {f}: {e}")
                    
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["date", "latitude", "longitude"])
                pollutant_dfs[pol] = combined_df
                logger.info(f"Filtered & Combined {pol} shape: {combined_df.shape}")
                
        if not pollutant_dfs:
            logger.error("No raw data found to merge.")
            return False
            
        # Outer merge all pollutants on spatiotemporal index ['date', 'latitude', 'longitude']
        master_df = None
        for pol, df in pollutant_dfs.items():
            if master_df is None:
                master_df = df
            else:
                master_df = pd.merge(master_df, df, on=["date", "latitude", "longitude"], how="outer")
                
        if master_df is not None:
            # Sort spatiotemporally
            master_df = master_df.sort_values(by=["date", "latitude", "longitude"]).reset_index(drop=True)
            
            # Fill missing entries inside merged dataset via linear interpolation per location
            numeric_cols = [col for col in pollutants if col in master_df.columns]
            logger.info("Performing spatiotemporal linear interpolation for missing values...")
            master_df[numeric_cols] = master_df.groupby(["latitude", "longitude"])[numeric_cols].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both').fillna(0.0)
            )
            
            # Create temporal features for CNN-LSTM sequence learning
            logger.info("Adding temporal feature transformations...")
            dates_parsed = pd.to_datetime(master_df["date"])
            master_df["year"] = dates_parsed.dt.year
            master_df["month"] = dates_parsed.dt.month
            master_df["day"] = dates_parsed.dt.day
            master_df["day_of_year"] = dates_parsed.dt.dayofyear
            master_df["season"] = master_df["month"].apply(self.get_season)
            
            # Reorder columns as requested
            desired_columns = ["date", "year", "month", "day", "season", "latitude", "longitude"] + [col for col in pollutants if col in master_df.columns]
            master_df = master_df[desired_columns]
            
            # Save final dataset
            os.makedirs(self.processed_dir, exist_ok=True)
            output_file = os.path.join(self.processed_dir, "sentinel5p_merged.csv")
            master_df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully generated merged dataset: {output_file} ({len(master_df)} rows)")
            return True
            
        return False

if __name__ == "__main__":
    merger = Sentinel5PMerger()
    merger.merge()

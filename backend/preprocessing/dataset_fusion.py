import os
import sys
import json
import logging
import datetime
import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler

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
        logging.FileHandler(os.path.join(log_dir, "dataset_fusion.log"))
    ]
)
logger = logging.getLogger("dataset_fusion")

class DatasetFusionPipeline:
    def __init__(self, processed_dir=None, output_dir=None, model_dir=None):
        self.processed_dir = processed_dir or os.path.join(os.path.dirname(__file__), "..", "datasets", "processed")
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "..", "datasets", "final", "v1")
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "..", "..", "models")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "scalers"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "metadata"), exist_ok=True)

    def load_datasets(self):
        files = {
            "cpcb": "cpcb_processed.csv",
            "s5p": "sentinel5p_merged.csv",
            "aod": "insat_aod_processed.csv",
            "era5": "era5_processed.csv",
            "firms": "firms_processed.csv"
        }
        dfs = {}
        for name, filename in files.items():
            path = os.path.join(self.processed_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Processed dataset {name} not found at {path}. Run data pipelines first.")
            dfs[name] = pd.read_csv(path)
            logger.info(f"Loaded {name} dataset: {len(dfs[name])} records.")
        return dfs

    def spatial_join_kdtree(self, target_df, source_df, prefix=""):
        """Aligns source_df coordinates with target_df coordinates using a KDTree"""
        target_coords = target_df[["latitude", "longitude"]].drop_duplicates().values
        source_coords = source_df[["latitude", "longitude"]].drop_duplicates().values
        
        # Build KDTree on source coordinates
        tree = KDTree(source_coords)
        
        coord_map = {}
        for lat, lon in target_coords:
            # Query KDTree
            dist, idx = tree.query([lat, lon])
            closest_lat, closest_lon = source_coords[idx]
            coord_map[(lat, lon)] = (closest_lat, closest_lon)
            
        # Map source_df values to target coordinates
        aligned_rows = []
        for _, row in target_df.iterrows():
            t_lat = row["latitude"]
            t_lon = row["longitude"]
            date_val = row["date"]
            
            s_lat, s_lon = coord_map[(t_lat, t_lon)]
            
            # Retrieve date-location record from source
            match = source_df[
                (source_df["date"] == date_val) & 
                (source_df["latitude"] == s_lat) & 
                (source_df["longitude"] == s_lon)
            ]
            
            if not match.empty:
                s_row = match.iloc[0].to_dict()
                # Remove index/coord keys
                for k in ["date", "latitude", "longitude", "year", "month", "day", "season"]:
                    s_row.pop(k, None)
                aligned_rows.append(s_row)
            else:
                # Append empty dict for missing dates
                aligned_rows.append({})
                
        df_aligned = pd.DataFrame(aligned_rows)
        if prefix:
            df_aligned = df_aligned.add_prefix(prefix)
            
        return pd.concat([target_df.reset_index(drop=True), df_aligned.reset_index(drop=True)], axis=1)

    def process_and_fuse(self):
        dfs = self.load_datasets()
        
        # CPCB acts as our base spatiotemporal frame
        base_df = dfs["cpcb"].copy()
        
        logger.info("Performing spatial and temporal joins using KDTree coordinates alignment...")
        # 1. Join Sentinel-5P
        base_df = self.spatial_join_kdtree(base_df, dfs["s5p"], prefix="")
        # 2. Join INSAT AOD
        base_df = self.spatial_join_kdtree(base_df, dfs["aod"], prefix="")
        # 3. Join ERA5 meteorology
        base_df = self.spatial_join_kdtree(base_df, dfs["era5"], prefix="")
        # 4. Join NASA FIRMS
        base_df = self.spatial_join_kdtree(base_df, dfs["firms"], prefix="")
        
        logger.info(f"Fused dataset baseline shape: {base_df.shape}")
        
        # Deduplicate columns if any duplicate headers exist after join
        base_df = base_df.loc[:, ~base_df.columns.duplicated()]
        
        # Handle missing values: Forward/Backward fill per station, then fill remaining with median
        logger.info("Handling missing value gaps across datasets...")
        numeric_cols = base_df.select_dtypes(include=[np.number]).columns
        base_df[numeric_cols] = base_df.groupby("station_id")[numeric_cols].transform(
            lambda x: x.ffill().bfill().fillna(x.median())
        )
        
        # Encode categoricals to numeric integers
        # Season mapping
        season_map = {"Winter": 0, "Pre-Monsoon": 1, "Monsoon": 2, "Post-Monsoon": 3}
        if "season" in base_df.columns:
            base_df["season"] = base_df["season"].map(season_map).fillna(0).astype(int)
            
        # Aerosol Intensity mapping
        aerosol_map = {"Low": 0, "Medium": 1, "High": 2}
        if "aerosol_intensity" in base_df.columns:
            base_df["aerosol_intensity"] = base_df["aerosol_intensity"].map(aerosol_map).fillna(0).astype(int)
            
        # Fire severity mapping
        severity_map = {"Low": 0, "Medium": 1, "High": 2}
        if "fire_severity_index" in base_df.columns:
            base_df["fire_severity_index"] = base_df["fire_severity_index"].map(severity_map).fillna(0).astype(int)

        # Calculate Lag Features
        logger.info("Calculating lag features (AQI_lag_1, AQI_lag_3, PM25_lag_1)...")
        base_df = base_df.sort_values(by=["station_id", "date"]).reset_index(drop=True)
        base_df["AQI_lag_1"] = base_df.groupby("station_id")["AQI"].shift(1).bfill()
        base_df["AQI_lag_3"] = base_df.groupby("station_id")["AQI"].shift(3).bfill()
        base_df["PM25_lag_1"] = base_df.groupby("station_id")["PM25"].shift(1).bfill()
        
        # Calculate targets: AQI, PM2.5, HCHO_hotspot_probability
        # HCHO Hotspot Probability defined as soft probability: clip(HCHO / 0.004, 0.0, 1.0)
        base_df["HCHO_hotspot_probability"] = np.clip(base_df["HCHO"] / 0.004, 0.0, 1.0)
        
        # Sort chronologically by date
        base_df = base_df.sort_values(by="date").reset_index(drop=True)
        
        return base_df

    def create_training_sequences(self, fused_df, sequence_length=7):
        logger.info(f"Generating temporal sequence windows (T={sequence_length}) for CNN-LSTM...")
        
        # Input features list
        features = [
            # Satellite
            "NO2", "SO2", "CO", "O3", "HCHO", "AOD",
            # Weather
            "temperature_mean", "humidity", "rainfall", "wind_speed", "wind_direction", "boundary_layer_height",
            # Fire
            "fire_count", "FRP", "fire_severity_index", "transport_influence_score",
            # Temporal
            "season", "day_of_year",
            # Lags
            "AQI_lag_1", "AQI_lag_3", "PM25_lag_1"
        ]
        
        # Verify features are actually in dataframe, log warning if missing
        missing_features = [f for f in features if f not in fused_df.columns]
        if missing_features:
            logger.warning(f"Expected features missing from fused dataset: {missing_features}")
            # Fallback rename mappings
            rename_map = {
                "temperature_mean": "temperature",
                "FRP": "average_FRP"
            }
            for mf in missing_features:
                fb = rename_map.get(mf)
                if fb in fused_df.columns:
                    fused_df = fused_df.rename(columns={fb: mf})
                    logger.info(f"Mapped fallback column {fb} -> {mf}")
                    
        # Update missing check
        features = [f for f in features if f in fused_df.columns]
        logger.info(f"Final input features selected for training: {features}")
        
        targets = ["AQI", "PM25", "HCHO_hotspot_probability"]
        
        X_seqs = []
        y_seqs = []
        dates_seqs = []
        station_seqs = []
        
        # Group by station to prevent sequence overlaps/leakage across stations
        for stn_id, df_stn in fused_df.groupby("station_id"):
            df_stn = df_stn.sort_values(by="date")
            stn_feats = df_stn[features].values
            stn_targets = df_stn[targets].values
            stn_dates = df_stn["date"].values
            
            if len(df_stn) < sequence_length:
                continue
                
            for idx in range(len(df_stn) - sequence_length):
                X_seq = stn_feats[idx : idx + sequence_length]
                # Label is next day (index + sequence_length) target
                y_seq = stn_targets[idx + sequence_length]
                
                X_seqs.append(X_seq)
                y_seqs.append(y_seq)
                dates_seqs.append(stn_dates[idx + sequence_length])
                station_seqs.append(stn_id)
                
        return np.array(X_seqs), np.array(y_seqs), dates_seqs, station_seqs, features, targets

    def split_and_scale(self, X, y, dates, stations, features, targets):
        logger.info("Executing chronological train-test partition (80% train, 20% test)...")
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("No sequence samples generated. Check raw files and overlapping dates.")
            
        # Chronological index split
        split_idx = int(n_samples * 0.8)
        
        # Sort sequences by date to guarantee strict chronological split
        sorted_indices = np.argsort(dates)
        X = X[sorted_indices]
        y = y[sorted_indices]
        dates = np.array(dates)[sorted_indices]
        stations = np.array(stations)[sorted_indices]
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]
        
        logger.info(f"Train set shape: X_train {X_train.shape}, y_train {y_train.shape}")
        logger.info(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
        logger.info(f"Train date range: {dates_train[0]} to {dates_train[-1]}")
        logger.info(f"Test date range: {dates_test[0]} to {dates_test[-1]}")
        
        # MinMaxScaler scaling - fitting ONLY on X_train to prevent data leakage
        logger.info("Fitting MinMaxScaler on X_train and transforming X_test...")
        scaler = MinMaxScaler()
        
        # Fit scaler on flattened X_train features
        s_tr, t_tr, f_tr = X_train.shape
        X_train_flat = X_train.reshape(-1, f_tr)
        scaler.fit(X_train_flat)
        
        # Transform training and testing
        X_train_scaled = scaler.transform(X_train_flat).reshape(s_tr, t_tr, f_tr)
        
        s_te, t_te, f_te = X_test.shape
        X_test_scaled = scaler.transform(X_test.reshape(-1, f_te)).reshape(s_te, t_te, f_te)
        
        # Save Scaler
        scaler_file = os.path.join(self.model_dir, "scalers", "feature_scaler.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"MinMaxScaler successfully serialized to {scaler_file}")
        
        # Fit target MinMaxScaler on y_train targets (AQI and PM2.5)
        logger.info("Fitting target MinMaxScaler on y_train targets and transforming y_test...")
        target_scaler = MinMaxScaler()
        target_scaler.fit(y_train[:, :2])
        
        target_scaler_file = os.path.join(self.model_dir, "scalers", "target_scaler.pkl")
        with open(target_scaler_file, "wb") as f:
            pickle.dump(target_scaler, f)
        logger.info(f"Target MinMaxScaler successfully serialized to {target_scaler_file}")
        
        y_train_scaled = y_train.copy()
        y_test_scaled = y_test.copy()
        y_train_scaled[:, :2] = target_scaler.transform(y_train[:, :2])
        y_test_scaled[:, :2] = target_scaler.transform(y_test[:, :2])
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, dates_train, dates_test

    def run(self):
        fused_df = self.process_and_fuse()
        
        # Save intermediate fused dataset
        intermediate_file = os.path.join(self.output_dir, "aqi_training_dataset.csv")
        fused_df.to_csv(intermediate_file, index=False)
        logger.info(f"Saved intermediate fused CSV to {intermediate_file}")
        
        X_seqs, y_seqs, dates_seqs, station_seqs, features, targets = self.create_training_sequences(fused_df)
        
        X_train, X_test, y_train, y_test, dates_train, dates_test = self.split_and_scale(
            X_seqs, y_seqs, dates_seqs, station_seqs, features, targets
        )
        
        # Save Numpy Arrays
        np.save(os.path.join(self.output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.output_dir, "y_test.npy"), y_test)
        logger.info("Saved final scaled numpy training arrays (.npy).")
        
        # Save Metadata file
        meta_data = {
            "dataset_version": "v1.0.0",
            "generation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_features": features,
            "output_targets": targets,
            "sequence_length": 7,
            "split_info": {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_date_start": str(dates_train[0]),
                "train_date_end": str(dates_train[-1]),
                "test_date_start": str(dates_test[0]),
                "test_date_end": str(dates_test[-1])
            }
        }
        
        # Metadata files
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(meta_data, f, indent=2)
            
        feature_names_file = os.path.join(self.model_dir, "metadata", "feature_names.json")
        with open(feature_names_file, "w") as f:
            json.dump({
                "features": features,
                "targets": targets,
                "sequence_length": 7,
                "version": "v1"
            }, f, indent=2)
        logger.info(f"Metadata files saved to {metadata_file} and {feature_names_file}")
        
        # Generate Validation Report
        self.generate_validation_report(fused_df, len(X_seqs), len(features), dates_train, dates_test)

    def generate_validation_report(self, fused_df, total_samples, n_features, dates_train, dates_test):
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_file = os.path.join(reports_dir, "final_dataset_report.json")
        
        target_stats = {}
        for target in ["AQI", "PM25", "HCHO_hotspot_probability"]:
            if target in fused_df.columns:
                target_stats[target] = {
                    "min": float(fused_df[target].min()),
                    "max": float(fused_df[target].max()),
                    "mean": float(fused_df[target].mean()),
                    "std": float(fused_df[target].std())
                }
                
        missing_count = int(fused_df.isna().sum().sum())
        total_elements = int(fused_df.size)
        
        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pipeline": "ISRO Multi-source CNN-LSTM Dataset Fusion Engine",
            "total_sequences_samples": total_samples,
            "number_of_features": n_features,
            "sequence_length": 7,
            "train_date_range": {
                "start": str(dates_train[0]),
                "end": str(dates_train[-1])
            },
            "test_date_range": {
                "start": str(dates_test[0]),
                "end": str(dates_test[-1])
            },
            "missing_data_percentage": float((missing_count / total_elements) * 100.0),
            "target_statistics": target_stats,
            "status": "Validated"
        }
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Final dataset validation report successfully written: {report_file}")

if __name__ == "__main__":
    pipeline = DatasetFusionPipeline()
    pipeline.process_and_fuse()
    pipeline.run()

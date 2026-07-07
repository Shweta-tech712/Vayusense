import os
import pickle
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union, Any
import yaml
from utils.logger import setup_logger

logger = setup_logger("preprocessing_pipeline")

class CPCBPreprocessor:
    """
    Cleans, quality-filters CPCB station records, and computes AQI and sub-indices
    according to Indian Central Pollution Control Board (CPCB) standards.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Breakpoints defined by CPCB for AQI calculation
        # Format: (conc_min, conc_max, aqi_min, aqi_max)
        self.breakpoints = {
            'pm25': [
                (0, 30, 0, 50), (30.1, 60, 51, 100), (60.1, 90, 101, 200),
                (90.1, 120, 201, 300), (120.1, 250, 301, 400), (250.1, 500, 401, 500)
            ],
            'pm10': [
                (0, 50, 0, 50), (50.1, 100, 51, 100), (100.1, 250, 101, 200),
                (250.1, 350, 201, 300), (350.1, 430, 301, 400), (430.1, 500, 401, 500)
            ],
            'no2': [
                (0, 40, 0, 50), (40.1, 80, 51, 100), (80.1, 180, 101, 200),
                (180.1, 280, 201, 300), (280.1, 400, 301, 400), (400.1, 1000, 401, 500)
            ],
            'so2': [
                (0, 40, 0, 50), (40.1, 80, 51, 100), (80.1, 380, 101, 200),
                (380.1, 800, 201, 300), (800.1, 1600, 301, 400), (1600.1, 5000, 401, 500)
            ],
            'co': [ # CO in mg/m3
                (0, 1.0, 0, 50), (1.01, 2.0, 51, 100), (2.01, 10.0, 101, 200),
                (10.01, 17.0, 201, 300), (17.01, 34.0, 301, 400), (34.01, 100, 401, 500)
            ],
            'o3': [
                (0, 50, 0, 50), (50.1, 100, 51, 100), (100.1, 168, 101, 200),
                (168.1, 208, 201, 300), (208.1, 748, 301, 400), (748.1, 1000, 401, 500)
            ]
        }

    def clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes physical anomalies and negative concentration values.
        """
        df = df.copy()
        logger.info("Applying quality filters and removing physical anomalies...")
        
        # Clean coordinates
        bbox = self.config['spatial']['india_bbox']
        df = df[
            (df['longitude'] >= bbox[0]) & (df['longitude'] <= bbox[2]) &
            (df['latitude'] >= bbox[1]) & (df['latitude'] <= bbox[3])
        ]
        
        # Physical bounding limits for concentrations
        limits = {
            'pm25': (0.0, 1000.0),
            'pm10': (0.0, 1500.0),
            'no2': (0.0, 1000.0),
            'so2': (0.0, 2000.0),
            'co': (0.0, 150.0),
            'o3': (0.0, 1000.0)
        }
        
        for col, (lower, upper) in limits.items():
            if col in df.columns:
                # Replace values outside bounds with NaN (to be interpolated later)
                df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
                
        return df

    def calculate_sub_index(self, conc: float, pollutant: str) -> float:
        """
        Computes the CPCB sub-index for a single pollutant based on concentration breakpoints.
        """
        if pd.isna(conc) or conc < 0:
            return np.nan
            
        ranges = self.breakpoints.get(pollutant)
        if not ranges:
            return np.nan
            
        for (c_lo, c_hi, a_lo, a_hi) in ranges:
            if c_lo <= conc <= c_hi:
                # Linear interpolation formula:
                # Sub-Index = [ (I_hi - I_lo)/(C_hi - C_lo) * (C - C_lo) ] + I_lo
                sub_idx = ((a_hi - a_lo) / (c_hi - c_lo)) * (conc - c_lo) + a_lo
                return round(sub_idx)
                
        # If concentration exceeds highest breakpoint, cap it at 500
        if conc > ranges[-1][1]:
            return 500.0
        return np.nan

    def compute_cpcb_aqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies sub-index calculation and computes the final Composite AQI.
        CPCB rules: Composite AQI requires at least 3 monitored pollutants,
        one of which MUST be PM2.5 or PM10.
        """
        df = df.copy()
        logger.info("Computing CPCB pollutant sub-indices and overall AQI...")
        
        sub_index_cols = []
        for pollutant in self.breakpoints.keys():
            if pollutant in df.columns:
                sub_col = f"{pollutant}_sub_index"
                df[sub_col] = df[pollutant].apply(lambda x: self.calculate_sub_index(x, pollutant))
                sub_index_cols.append(sub_col)
                
        # Find maximum sub-index across available pollutants
        df['cpcb_aqi'] = df[sub_index_cols].max(axis=1)
        
        # Check CPCB criteria constraints (>= 3 sub-indices, and one of them is PM2.5/PM10)
        valid_count = df[sub_index_cols].notna().sum(axis=1)
        
        has_pm = pd.Series(False, index=df.index)
        if 'pm25_sub_index' in df.columns:
            has_pm = has_pm | df['pm25_sub_index'].notna()
        if 'pm10_sub_index' in df.columns:
            has_pm = has_pm | df['pm10_sub_index'].notna()
            
        # Invalidate AQI where criteria is not met
        df.loc[(valid_count < 3) | (~has_pm), 'cpcb_aqi'] = np.nan
        
        return df


class SpatialAligner:
    """
    Handles geographical reprojections, coordinate checks,
    and extracts spatial matrix patches centered around coordinates.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.patch_size = self.config['model']['spatial_patch_size']

    def extract_numpy_patch(self, raster_arr: np.ndarray, center_idx: Tuple[int, int]) -> np.ndarray:
        """
        Extracts a square (N x N) patch from a 2D grid centered at center_idx.
        Pads with edge values if the patch extends outside grid boundaries.
        """
        r = self.patch_size // 2
        cy, cx = center_idx
        h, w = raster_arr.shape
        
        y_start = cy - r
        y_end = cy + r + 1
        x_start = cx - r
        x_end = cx + r + 1
        
        # Boundaries check & padding setup
        pad_y_before = max(0, -y_start)
        pad_y_after = max(0, y_end - h)
        pad_x_before = max(0, -x_start)
        pad_x_after = max(0, x_end - w)
        
        slice_y_start = max(0, y_start)
        slice_y_end = min(h, y_end)
        slice_x_start = max(0, x_start)
        slice_x_end = min(w, x_end)
        
        patch = raster_arr[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
        
        if pad_y_before > 0 or pad_y_after > 0 or pad_x_before > 0 or pad_x_after > 0:
            patch = np.pad(patch, ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='edge')
            
        return patch


class TemporalAligner:
    """
    Aligns ground-station time series with daily satellite observations,
    constructing 5D sequences with a temporal lag parameter T.
    """
    def __init__(self, sequence_length: int = 7):
        self.sequence_length = sequence_length

    def create_lagged_sequences(self, station_df: pd.DataFrame, satellite_data_dict: Dict[datetime.date, Dict[str, np.ndarray]], 
                                station_coords: Tuple[float, float], patch_aligner: SpatialAligner) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds temporal lag matrices for deep learning training.
        For a station with records at dates t, creates sequences of shape (T, H, W, C)
        referring to days t-T+1 through t.
        """
        # Ensure data is chronological
        station_df = station_df.sort_values(by='date')
        
        sequences = []
        targets = []
        
        # Sort out dates containing active satellite records
        available_dates = sorted(list(satellite_data_dict.keys()))
        
        for idx in range(self.sequence_length - 1, len(station_df)):
            target_row = station_df.iloc[idx]
            target_date = target_row['date'].date()
            target_val = target_row['cpcb_aqi']
            
            if pd.isna(target_val):
                continue
                
            # Grab sequence of past T dates
            seq_dates = [station_df.iloc[idx - j]['date'].date() for j in reversed(range(self.sequence_length))]
            
            # Verify all sequence dates are present in our satellite dictionary
            valid_seq = True
            seq_features = []
            
            for s_date in seq_dates:
                if s_date not in satellite_data_dict:
                    valid_seq = False
                    break
                    
                # Pull raster bands for this date
                bands_dict = satellite_data_dict[s_date]
                bands_list = []
                
                # Assume bands_dict holds {'AOD': (grid, center_idx), 'HCHO': (grid, center_idx)...}
                for band_name, (grid, center_idx) in bands_dict.items():
                    patch = patch_aligner.extract_numpy_patch(grid, center_idx)
                    bands_list.append(patch)
                    
                # Stack bands along channel axis: shape (H, W, C)
                day_tensor = np.stack(bands_list, axis=-1)
                seq_features.append(day_tensor)
                
            if valid_seq:
                # Stack temporal steps: shape (T, H, W, C)
                seq_tensor = np.stack(seq_features, axis=0)
                sequences.append(seq_tensor)
                targets.append(target_val)
                
        return np.array(sequences), np.array(targets)


class DataNormalizer:
    """
    Standardizes inputs (Z-Score Normalization) for deep learning.
    Saves and loads scaler parameters to prevent data leakage during inference.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform_sequence(self, sequences: np.ndarray) -> np.ndarray:
        """
        Fits scaler on a 5D array (Samples, Time, Height, Width, Channels)
        and returns standardized sequences.
        """
        s, t, h, w, c = sequences.shape
        # Reshape to 2D matrix for scaling (all values of a channel get scaled together)
        flat_data = sequences.reshape(-1, c)
        scaled_flat = self.scaler.fit_transform(flat_data)
        
        self.is_fitted = True
        logger.info("Fitted normalizer on input sequences.")
        return scaled_flat.reshape(s, t, h, w, c)

    def transform_sequence(self, sequences: np.ndarray) -> np.ndarray:
        """
        Transforms 5D array based on previously fitted scaler params.
        """
        if not self.is_fitted:
            logger.error("Attempted to scale data before fitting scaler.")
            raise RuntimeError("Scaler is not fitted yet.")
            
        s, t, h, w, c = sequences.shape
        flat_data = sequences.reshape(-1, c)
        scaled_flat = self.scaler.transform(flat_data)
        return scaled_flat.reshape(s, t, h, w, c)

    def save_scaler(self, filepath: str = "models/production/scaler.pkl") -> None:
        """
        Serializes the fitted scaler to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to: {filepath}")

    def load_scaler(self, filepath: str = "models/production/scaler.pkl") -> None:
        """
        Deserializes a saved scaler from disk.
        """
        if not os.path.exists(filepath):
            logger.error(f"Saved scaler not found at path: {filepath}")
            raise FileNotFoundError(f"Scaler path {filepath} does not exist.")
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded scaler from: {filepath}")


class MissingValueInterpolator:
    """
    Performs scientific gap-filling on datasets using spatial Inverse Distance Weighting (IDW)
    and temporal forward/backward interpolation.
    """
    @staticmethod
    def temporal_interpolate(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fills missing values in time-series using linear temporal interpolation.
        Falls back to forward/backward fill for remaining edge NaNs.
        """
        df = df.copy()
        # Ensure sorted by time
        df = df.sort_values(by='date')
        
        for col in columns:
            if col in df.columns:
                df[col] = df[col].interpolate(method='time', limit_direction='both')
                # If still contains NaN, apply forward fill then backfill
                df[col] = df[col].ffill().bfill()
        return df

    @staticmethod
    def spatial_idw_interpolate(points: List[Tuple[float, float]], values: List[float], 
                                 target_point: Tuple[float, float], power: float = 2.0) -> float:
        """
        Fills a single missing point value using Inverse Distance Weighting (IDW) from nearby stations.
        """
        valid_idx = [i for i, val in enumerate(values) if not pd.isna(val)]
        if not valid_idx:
            return np.nan
            
        distances = []
        target_lon, target_lat = target_point
        
        for idx in valid_idx:
            src_lon, src_lat = points[idx]
            # Simple Euclidean distance in degrees (sufficient for small neighborhood grids)
            dist = np.sqrt((target_lon - src_lon)**2 + (target_lat - src_lat)**2)
            distances.append(max(dist, 1e-6)) # Avoid division by zero
            
        weights = 1.0 / (np.array(distances) ** power)
        weight_sum = np.sum(weights)
        
        if weight_sum == 0:
            return np.nan
            
        interpolated_val = np.sum(weights * np.array([values[i] for i in valid_idx])) / weight_sum
        return float(interpolated_val)

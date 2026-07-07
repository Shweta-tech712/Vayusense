import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Tuple, Union
import yaml
from utils.logger import setup_logger

logger = setup_logger("feature_engineer")

class FeatureEngineer:
    """
    Production-ready feature engineering pipeline for satellite, meteorological,
    fire, and temporal parameters.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def engineer_meteorological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives wind speed and wind direction vectors from ERA5 u and v wind components.
        Converts temperature from Kelvin to Celsius.
        """
        df = df.copy()
        logger.info("Engineering meteorological features...")
        try:
            # Wind speed calculation: WS = sqrt(u^2 + v^2)
            if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
                u = df['u_component_of_wind_10m']
                v = df['v_component_of_wind_10m']
                df['wind_speed'] = np.sqrt(u**2 + v**2)
                
                # Wind direction in degrees: WD = atan2(u, v) * (180/pi)
                # Map to [0, 360] degrees range
                df['wind_direction'] = (np.arctan2(u, v) * 180 / np.pi + 360) % 360
                logger.info("Successfully calculated wind speed and wind direction vectors.")
            else:
                logger.warning("ERA5 wind component bands missing. Skipping vector wind features.")
                
            # Temperature conversion: K to C
            if 'temperature_2m' in df.columns:
                df['temp_celsius'] = df['temperature_2m'] - 273.15
            
            return df
        except Exception as e:
            logger.error(f"Error in meteorological feature engineering: {e}")
            raise e

    def engineer_satellite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates photochemical index indicators, specifically the HCHO/NO2 ratio.
        HCHO/NO2 is a critical indicator of Ozone chemical production regimes.
        """
        df = df.copy()
        logger.info("Engineering satellite features...")
        try:
            # If both HCHO column and NO2 columns exist, calculate their ratio
            # Adding epsilon (1e-9) to prevent division by zero
            if 'hcho' in df.columns and 'no2' in df.columns:
                df['hcho_no2_ratio'] = df['hcho'] / (df['no2'] + 1e-9)
                logger.info("Successfully calculated HCHO/NO2 photochemical ratio.")
            else:
                logger.warning("Sentinel-5P HCHO or NO2 bands missing. Skipping photochemical ratio feature.")
                
            return df
        except Exception as e:
            logger.error(f"Error in satellite feature engineering: {e}")
            raise e

    def engineer_fire_features(self, df: pd.DataFrame, fires_df: pd.DataFrame, buffer_km: float = 50.0) -> pd.DataFrame:
        """
        Spatially intersects CPCB station coordinates with NASA FIRMS active fire points
        within a specified buffer radius to count active fires and sum cumulative FRP.
        """
        df = df.copy()
        logger.info(f"Engineering fire features using a {buffer_km}km spatial buffer...")
        
        if fires_df.empty:
            logger.warning("FIRMS active fire dataset is empty. Filling fire features with 0.")
            df['fire_count_buffer'] = 0.0
            df['cumulative_frp_buffer'] = 0.0
            return df
            
        try:
            # Convert CPCB stations to GeoDataFrame (WGS84)
            stations_gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                crs="EPSG:4326"
            )
            
            # Convert FIRMS fires to GeoDataFrame
            fires_gdf = gpd.GeoDataFrame(
                fires_df,
                geometry=gpd.points_from_xy(fires_df['longitude'], fires_df['latitude']),
                crs="EPSG:4326"
            )
            
            # Project to a metric CRS for accurate distance buffers (EPSG:3857 global web mercator or India UTM)
            # We use EPSG:3857 for robust distance calculations
            stations_metric = stations_gdf.to_crs(epsg=3857)
            fires_metric = fires_gdf.to_crs(epsg=3857)
            
            # Create buffer geometry around stations (buffer_km converted to meters)
            stations_metric['buffer_geom'] = stations_metric.geometry.buffer(buffer_km * 1000)
            
            # We construct a helper table to perform spatial join between buffered stations and fire points
            stations_metric = stations_metric.set_geometry('buffer_geom')
            joined = gpd.sjoin(fires_metric, stations_metric, how='inner', predicate='within')
            
            # Group by station and date to aggregate metrics
            # Note: Assume fires_df has a 'date' column matching CPCB datetime dates
            if 'date_left' in joined.columns:
                date_col = 'date_left'
            else:
                date_col = 'date'
                
            # Aggregate fire count and total Fire Radiative Power (FRP)
            fire_stats = joined.groupby(['station', date_col]).agg(
                fire_count=('frp', 'count'),
                total_frp=('frp', 'sum')
            ).reset_index()
            
            # Rename grouping column to match CPCB standard
            fire_stats = fire_stats.rename(columns={date_col: 'date'})
            fire_stats['date'] = pd.to_datetime(fire_stats['date'])
            
            # Merge aggregates back into the main CPCB DataFrame
            df = df.merge(fire_stats, on=['station', 'date'], how='left')
            df['fire_count'] = df['fire_count'].fillna(0.0)
            df['total_frp'] = df['total_frp'].fillna(0.0)
            
            logger.info("Successfully calculated active fire count and cumulative FRP features.")
            return df
            
        except Exception as e:
            logger.error(f"Error in spatial fire feature engineering: {e}")
            # Fallback
            df['fire_count'] = 0.0
            df['total_frp'] = 0.0
            return df

    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts calendar parameters and applies cyclical sine/cosine encoding
        to month and day-of-year variables to model seasonality.
        """
        df = df.copy()
        logger.info("Engineering temporal features...")
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Cyclical month encoding
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Cyclical day of year encoding
            df['sin_day_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['cos_day_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
            
            logger.info("Successfully engineered cyclical temporal features.")
            return df
        except Exception as e:
            logger.error(f"Error in temporal feature engineering: {e}")
            raise e

    def engineer_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates spatial controls like normalized lat/lon grids to capture regional microclimates.
        """
        df = df.copy()
        logger.info("Engineering spatial features...")
        try:
            bbox = self.config['spatial']['india_bbox']
            # Normalise coordinates to [0, 1] range based on national bounding limits
            df['norm_lat'] = (df['latitude'] - bbox[1]) / (bbox[3] - bbox[1])
            df['norm_lon'] = (df['longitude'] - bbox[0]) / (bbox[2] - bbox[0])
            
            logger.info("Successfully generated normalized coordinates.")
            return df
        except Exception as e:
            logger.error(f"Error in spatial feature engineering: {e}")
            raise e

    def engineer_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Calculates multi-day historical lag variables ($t-1, t-2, t-3$) for the target features.
        Must group by station first to prevent spatial data blending.
        """
        df = df.copy()
        logger.info(f"Engineering lag features for columns {columns} with shifts {lags}...")
        try:
            df = df.sort_values(by=['station', 'date'])
            
            for col in columns:
                if col in df.columns:
                    for lag in lags:
                        df[f"{col}_lag_{lag}"] = df.groupby('station')[col].shift(lag)
                        
            logger.info("Successfully calculated spatial-temporal lag features.")
            return df
        except Exception as e:
            logger.error(f"Error in lag feature engineering: {e}")
            raise e

    def engineer_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int] = [3, 7]) -> pd.DataFrame:
        """
        Calculates moving rolling averages for spatial points to capture baseline trends.
        """
        df = df.copy()
        logger.info(f"Engineering rolling features for columns {columns} with window sizes {windows}...")
        try:
            df = df.sort_values(by=['station', 'date'])
            
            for col in columns:
                if col in df.columns:
                    for window in windows:
                        # minimum periods is set to 1 to prevent excessive NaN generation at start points
                        df[f"{col}_roll_mean_{window}"] = (df.groupby('station')[col]
                                                           .transform(lambda x: x.rolling(window, min_periods=1).mean()))
                        
            logger.info("Successfully calculated rolling average features.")
            return df
        except Exception as e:
            logger.error(f"Error in rolling feature engineering: {e}")
            raise e

    @staticmethod
    def map_aqi_categories(aqi: float) -> str:
        """
        Translates a numerical CPCB AQI value into its official air quality classification string.
        """
        if pd.isna(aqi) or aqi < 0:
            return "Unknown"
        elif aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Satisfactory"
        elif aqi <= 200:
            return "Moderate"
        elif aqi <= 300:
            return "Poor"
        elif aqi <= 400:
            return "Very Poor"
        else:
            return "Severe"

    def apply_pipeline(self, df: pd.DataFrame, fires_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Orchestrates and executes all feature engineering routines sequentially.
        """
        logger.info("Starting master feature engineering pipeline...")
        
        df = self.engineer_temporal_features(df)
        df = self.engineer_spatial_features(df)
        df = self.engineer_meteorological_features(df)
        df = self.engineer_satellite_features(df)
        
        if fires_df is not None:
            df = self.engineer_fire_features(df, fires_df)
        else:
            df['fire_count'] = 0.0
            df['total_frp'] = 0.0
            
        # Define target features for lag and rolling calculations
        lag_cols = ['cpcb_aqi', 'pm25', 'pm10']
        if 'hcho' in df.columns:
            lag_cols.append('hcho')
        if 'wind_speed' in df.columns:
            lag_cols.append('wind_speed')
            
        df = self.engineer_lag_features(df, columns=lag_cols)
        df = self.engineer_rolling_features(df, columns=lag_cols)
        
        # Map AQI categories
        df['aqi_category'] = df['cpcb_aqi'].apply(self.map_aqi_categories)
        
        logger.info("Master feature engineering pipeline execution completed.")
        return df

import os
import sqlite3
import datetime
import urllib.request
import json
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger
from preprocessing.preprocessing_pipeline import CPCBPreprocessor

logger = setup_logger("cpcb_pipeline")

class CPCBDownloader:
    """
    Handles downloading CPCB air quality datasets.
    Queries the official api.data.gov.in endpoint or fetches historic portal packages.
    """
    def __init__(self, config_path: str = "config/config.yaml", api_key: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.api_key = api_key or os.environ.get("DATA_GOV_IN_API_KEY")
        self.raw_dir = self.config['paths']['raw_cpcb_dir']
        os.makedirs(self.raw_dir, exist_ok=True)

    def download_realtime_data(self) -> str:
        """
        Downloads real-time ambient air quality data for all active India stations
        using the api.data.gov.in resource API.
        """
        if not self.api_key:
            logger.warning("api.data.gov.in API key missing. Cannot fetch real-time data programmatically. Place CPCB files manually in data/raw/")
            return ""
            
        # Target resource ID for Real-time ambient air quality index data
        resource_id = "3b01bbb4-0773-40f2-b3e9-729b49b4434a"
        url = f"https://api.data.gov.in/model/v1?api-key={self.api_key}&format=json&resource_id={resource_id}&limit=1000"
        
        local_filename = os.path.join(self.raw_dir, f"cpcb_realtime_{datetime.date.today().isoformat()}.json")
        logger.info(f"Querying Open Data India API: {url}...")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            with open(local_filename, 'w') as f:
                json.dump(data, f)
                
            logger.info(f"Successfully downloaded API database to: {local_filename}")
            return local_filename
        except Exception as e:
            logger.error(f"Failed to query api.data.gov.in: {e}")
            return ""


class CPCBFileCleaner:
    """
    Standardizes column formats, parses datetimes, removes bad entries,
    and computes pollutant sub-indices and composite CPCB AQI.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.preprocessor = CPCBPreprocessor(config_path)
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the complete cleansing chain on the ground station DataFrame.
        """
        try:
            # 1. Clean outliers and geographic boundary limits
            df = self.preprocessor.clean_outliers(df)
            
            # 2. Compute CPCB AQI based on breakpoint standards
            df = self.preprocessor.compute_cpcb_aqi(df)
            
            # 3. Drop records that do not contain valid coordinates or dates
            df = df.dropna(subset=['date', 'latitude', 'longitude'])
            
            # 4. Standardize column cases
            df.columns = [col.lower().strip() for col in df.columns]
            
            logger.info("Dataset cleaning complete.")
            return df
        except Exception as e:
            logger.error(f"Error during dataset cleaning: {e}")
            raise e


class CPCBValidator:
    """
    Validates CPCB data schemas, geographic boundaries, and value distributions.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.bbox = self.config['spatial']['india_bbox']

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates the cleaned DataFrame against production constraints.
        Returns True if the dataset meets quality control standards.
        """
        logger.info("Starting CPCB dataset quality validation...")
        
        if df.empty:
            logger.error("Validation failed: DataFrame is empty.")
            return False
            
        # Check required columns
        required_cols = {'date', 'station', 'latitude', 'longitude', 'cpcb_aqi'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error(f"Validation failed: Missing core columns: {missing}")
            return False
            
        # 1. Coordinate check
        lat_check = df['latitude'].between(self.bbox[1], self.bbox[3]).all()
        lon_check = df['longitude'].between(self.bbox[0], self.bbox[2]).all()
        if not (lat_check and lon_check):
            logger.warning("Validation Warning: Some coordinates lie outside the India bounding box.")
            
        # 2. Value range constraints
        aqi_check = df['cpcb_aqi'].dropna().between(0, 500).all()
        if not aqi_check:
            logger.error("Validation failed: Detected CPCB AQI values outside [0, 500] range.")
            return False
            
        # 3. Datetime checks
        if df['date'].isnull().any():
            logger.error("Validation failed: Found null timestamps in the date column.")
            return False
            
        logger.info(f"Validation successful. Total rows: {len(df)}. Max AQI: {df['cpcb_aqi'].max()}")
        return True


class CPCBDatabaseCreator:
    """
    Compiles cleaned station records into CSV and SQLite databases.
    Enables rapid spatial-temporal querying for models and dashboards.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.processed_dir = self.config['paths']['processed_dir']
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.db_path = os.path.join(self.processed_dir, "cpcb_air_quality.db")
        self.csv_path = os.path.join(self.processed_dir, "cpcb_aligned_database.csv")

    def initialize_sqlite_db(self) -> None:
        """
        Initializes the SQLite tables with proper constraints and indexes.
        """
        logger.info(f"Initializing SQLite database at: {self.db_path}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS air_quality_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        station TEXT NOT NULL,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        pm25 REAL,
                        pm10 REAL,
                        no2 REAL,
                        so2 REAL,
                        co REAL,
                        o3 REAL,
                        pm25_sub_index REAL,
                        pm10_sub_index REAL,
                        no2_sub_index REAL,
                        so2_sub_index REAL,
                        co_sub_index REAL,
                        o3_sub_index REAL,
                        cpcb_aqi REAL,
                        UNIQUE(date, station)
                    )
                """)
                # Indexes for query speedups
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON air_quality_data (date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_station ON air_quality_data (station)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_coords ON air_quality_data (latitude, longitude)")
                conn.commit()
            logger.info("SQLite database tables and indexes initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise e

    def save_to_database(self, df: pd.DataFrame) -> None:
        """
        Appends or writes clean data to both the consolidated CSV and SQLite database.
        """
        # Save to CSV
        logger.info(f"Writing cleaned data to CSV database: {self.csv_path}")
        try:
            if os.path.exists(self.csv_path):
                # Append and drop duplicates
                existing_df = pd.read_csv(self.csv_path)
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined['date'] = pd.to_datetime(combined['date'])
                combined = combined.drop_duplicates(subset=['date', 'station'])
                combined.to_csv(self.csv_path, index=False)
            else:
                df.to_csv(self.csv_path, index=False)
        except Exception as e:
            logger.error(f"Failed to write data to CSV: {e}")
            raise e
            
        # Save to SQLite
        logger.info("Writing data to SQLite database...")
        try:
            df_db = df.copy()
            df_db['date'] = df_db['date'].dt.strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                # Convert DataFrame columns to align with database table columns
                # Iterate and insert to handle "ON CONFLICT REPLACE" logic
                cursor = conn.cursor()
                for _, row in df_db.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO air_quality_data (
                            date, station, latitude, longitude, pm25, pm10, no2, so2, co, o3,
                            pm25_sub_index, pm10_sub_index, no2_sub_index, so2_sub_index, co_sub_index, o3_sub_index,
                            cpcb_aqi
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['date'], row['station'], row['latitude'], row['longitude'],
                        row.get('pm25', None), row.get('pm10', None), row.get('no2', None), row.get('so2', None), row.get('co', None), row.get('o3', None),
                        row.get('pm25_sub_index', None), row.get('pm10_sub_index', None), row.get('no2_sub_index', None), row.get('so2_sub_index', None), row.get('co_sub_index', None), row.get('o3_sub_index', None),
                        row.get('cpcb_aqi', None)
                    ))
                conn.commit()
            logger.info("SQLite transaction committed.")
        except Exception as e:
            logger.error(f"Failed to insert data into SQLite database: {e}")
            raise e

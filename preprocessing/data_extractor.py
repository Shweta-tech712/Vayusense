import os
import ftplib
import datetime
import pandas as pd
import numpy as np
import ee
import yaml
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger
from utils.gee_auth import initialize_gee

logger = setup_logger("data_extractor")

class GEEDataExtractor:
    """
    Handles data query and extraction from Google Earth Engine (GEE) API.
    Extracts Sentinel-5P, MODIS AOD, NASA FIRMS, and ERA5 meteorological parameters.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize GEE
        logger.info("Initializing GEE session for data extraction...")
        initialize_gee()
        
        # Extract spatial boundaries
        bbox = self.config['spatial']['india_bbox']
        self.roi = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
        logger.info(f"Defined Region of Interest (ROI) over India: {bbox}")

    def get_sentinel5p_image(self, date_str: str, product: str = "hcho") -> ee.Image:
        """
        Retrieves Sentinel-5P column density images for a specific date over India.
        """
        try:
            p_config = self.config['sensors']['sentinel_5p']
            date = ee.Date(date_str)
            
            if product == "hcho":
                collection = p_config['hcho_collection']
                band = p_config['hcho_band']
                qa_band = p_config['hcho_qa_band']
                qa_thresh = p_config['hcho_qa_threshold']
            elif product == "no2":
                collection = p_config['no2_collection']
                band = p_config['no2_band']
                qa_band = "qa_value"
                qa_thresh = p_config['no2_qa_threshold']
            else:
                raise ValueError(f"Unsupported product: {product}")
                
            img_coll = (ee.ImageCollection(collection)
                        .filterDate(date, date.advance(1, 'day'))
                        .filterBounds(self.roi))
            
            # Reduce to daily composite (mean)
            composite = img_coll.select([band, qa_band]).mean()
            
            # Apply quality assurance band filtering
            # Note: We return the image with its QA band; mask execution is deferred to preprocessing
            return composite.clip(self.roi)
        except Exception as e:
            logger.error(f"Error fetching Sentinel-5P {product} on {date_str}: {e}")
            return None

    def get_modis_aod(self, date_str: str) -> ee.Image:
        """
        Retrieves MODIS MCD19A2 AOD (Aerosol Optical Depth) daily composite.
        """
        try:
            m_config = self.config['sensors']['modis']
            date = ee.Date(date_str)
            
            img_coll = (ee.ImageCollection(m_config['aod_collection'])
                        .filterDate(date, date.advance(1, 'day'))
                        .filterBounds(self.roi)
                        .select(m_config['aod_band']))
            
            # Daily mean composite
            aod_img = img_coll.mean().multiply(0.001) # Apply scale factor
            return aod_img.clip(self.roi)
        except Exception as e:
            logger.error(f"Error fetching MODIS AOD on {date_str}: {e}")
            return None

    def get_era5_meteorology(self, date_str: str) -> ee.Image:
        """
        Retrieves ERA5 Land daily meteorological parameters (Temp, Winds, Precip).
        """
        try:
            e_config = self.config['sensors']['era5']
            date = ee.Date(date_str)
            
            img_coll = (ee.ImageCollection(e_config['collection'])
                        .filterDate(date, date.advance(1, 'day'))
                        .filterBounds(self.roi)
                        .select(e_config['bands']))
            
            # Mean composite for the day
            met_img = img_coll.mean()
            return met_img.clip(self.roi)
        except Exception as e:
            logger.error(f"Error fetching ERA5 Land meteorology on {date_str}: {e}")
            return None

    def get_firms_fires(self, date_str: str) -> ee.FeatureCollection:
        """
        Retrieves NASA FIRMS active fire pixels for a specific date.
        """
        try:
            f_config = self.config['sensors']['firms']
            date = ee.Date(date_str)
            
            fires = (ee.FeatureCollection(f_config['collection'])
                     .filterDate(date, date.advance(1, 'day'))
                     .filterBounds(self.roi))
            
            return fires
        except Exception as e:
            logger.error(f"Error fetching NASA FIRMS fires on {date_str}: {e}")
            return None

    def extract_point_data(self, image: ee.Image, coords: List[Tuple[float, float]], scale: int = 10000) -> List[Dict[str, Any]]:
        """
        Helper to extract pixel values at point coordinates.
        Useful for building ground-station matched training datasets.
        """
        if image is None:
            return [{} for _ in coords]
            
        try:
            points = [ee.Feature(ee.Geometry.Point(lon, lat), {'idx': i}) for i, (lon, lat) in enumerate(coords)]
            features = ee.FeatureCollection(points)
            
            extracted = image.reduceRegions(
                collection=features,
                reducer=ee.Reducer.mean(),
                scale=scale
            ).getInfo()
            
            results = []
            for feat in extracted['features']:
                results.append(feat['properties'])
            return results
        except Exception as e:
            logger.error(f"Failed to extract point values: {e}")
            return [{} for _ in coords]


class MOSDACDownloader:
    """
    Downloads INSAT-3D Aerosol Optical Depth (AOD) HDF5 datasets from ISRO MOSDAC FTP Server.
    Requires MOSDAC credentials set in FTP config parameters.
    """
    def __init__(self, ftp_host: str = "ftp.mosdac.gov.in", username: str = "", password: str = ""):
        self.ftp_host = ftp_host
        self.username = username
        self.password = password
        self.raw_dir = "data/raw/insat3d/"
        os.makedirs(self.raw_dir, exist_ok=True)

    def download_insat_aod(self, target_date: datetime.date) -> str:
        """
        Connects to MOSDAC FTP and downloads the HDF5 AOD file for the target date.
        Returns the local path of the downloaded file.
        """
        # FTP paths are typical of format: /products/INSAT-3D/IMAGER/L2B_AOD/YYYY/MM/
        year = target_date.strftime("%Y")
        month = target_date.strftime("%m")
        day = target_date.strftime("%d")
        
        logger.info(f"Connecting to MOSDAC FTP server: {self.ftp_host}...")
        
        if not self.username or not self.password:
            logger.warning("MOSDAC credentials are empty. Skipping direct FTP download. Place files manually in data/raw/insat3d/")
            return ""
            
        try:
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login(user=self.username, passwd=self.password)
                logger.info("MOSDAC FTP Authentication successful.")
                
                # Navigate to the daily product directory
                remote_dir = f"/products/INSAT-3D/IMAGER/L2B_AOD/{year}/{month}"
                ftp.cwd(remote_dir)
                
                # List files matching target day
                files = ftp.nlst()
                target_prefix = f"3D_AOD_{year}_{month}_{day}"
                matching_files = [f for f in files if f.startswith(target_prefix)]
                
                if not matching_files:
                    logger.warning(f"No INSAT-3D AOD file found on MOSDAC for date: {target_date}")
                    return ""
                    
                filename = matching_files[0] # Fetch first available daily retrieval
                local_filepath = os.path.join(self.raw_dir, filename)
                
                logger.info(f"Downloading INSAT-3D file: {filename} to {local_filepath}...")
                with open(local_filepath, "wb") as local_file:
                    ftp.retrbinary(f"RETR {filename}", local_file.write)
                    
                logger.info("Download completed successfully.")
                return local_filepath
                
        except Exception as e:
            logger.error(f"Failed to download INSAT-3D data from MOSDAC FTP: {e}")
            return ""


class CPCBDataLoader:
    """
    Ingests ground monitoring station data exported from CPCB CCMP Web Portal.
    Standardized CSV files contain PM2.5, PM10, Gaseous concentrations and coordinates.
    """
    def __init__(self, raw_dir: str = "data/raw/"):
        self.raw_dir = raw_dir

    def load_station_data(self, filename: str) -> pd.DataFrame:
        """
        Loads a CPCB CSV file, standardizes timestamps, columns, and station coordinates.
        Expected columns: Timestamp/Date, PM2.5, PM10, NO2, SO2, CO, Ozone, Station_Name, Latitude, Longitude.
        """
        filepath = os.path.join(self.raw_dir, filename)
        logger.info(f"Loading ground station CSV: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"CPCB raw file not found at path: {filepath}")
            raise FileNotFoundError(f"File {filepath} does not exist.")
            
        try:
            df = pd.read_csv(filepath)
            
            # Essential field validations
            required_cols = ['date', 'station', 'latitude', 'longitude', 'pm25']
            missing_cols = [col for col in required_cols if col not in df.columns.str.lower()]
            
            if missing_cols:
                logger.warning(f"File {filename} is missing standard headers {missing_cols}. Attempting mapping...")
                # Apply flexible mappings for standard CPCB portal exports
                col_mapping = {
                    'date/time': 'date', 'datetime': 'date', 'timestamp': 'date',
                    'station name': 'station', 'station_name': 'station',
                    'lat': 'latitude', 'lon': 'longitude', 'long': 'longitude',
                    'pm2.5': 'pm25', 'pm-2.5': 'pm25', 'pm10': 'pm10', 'pm-10': 'pm10'
                }
                df = df.rename(columns=lambda x: col_mapping.get(x.lower(), x.lower()))
            
            # Format datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Standardize numeric types
            numeric_cols = ['latitude', 'longitude', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded {len(df)} records for stations from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing CPCB file {filename}: {e}")
            raise e

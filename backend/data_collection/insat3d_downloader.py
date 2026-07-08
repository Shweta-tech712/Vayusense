import os
import sys
import json
import logging
import datetime
import argparse
import ftplib
import requests

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
        logging.FileHandler(os.path.join(log_dir, "insat3d_downloader.log"))
    ]
)
logger = logging.getLogger("insat3d_downloader")

class INSAT3DDownloader:
    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config", "insat_config.json")
        self.load_config()
        
    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            logger.info("Loaded INSAT configuration successfully.")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def get_mosdac_credentials(self):
        username = os.environ.get("MOSDAC_USERNAME")
        password = os.environ.get("MOSDAC_PASSWORD")
        if not username or not password:
            # Check config as secondary source
            username = self.config.get("ftp_username")
            password = self.config.get("ftp_password")
        return username, password

    def download_from_mosdac(self, target_date, output_dir):
        username, password = self.get_mosdac_credentials()
        if not username or not password:
            logger.warning("missing MOSDAC credentials: FTP username or password not configured. Initiating fallback to MODIS MAIAC...")
            return None
            
        host = self.config.get("ftp_host", "ftp.mosdac.gov.in")
        year = target_date.strftime("%Y")
        month = target_date.strftime("%m")
        day = target_date.strftime("%d")
        
        try:
            logger.info(f"Connecting to MOSDAC FTP server: {host}...")
            ftp = ftplib.FTP(host, timeout=30)
        except Exception as e:
            raise ConnectionError(f"network issue: Failed to establish connection with MOSDAC FTP server. Error: {e}")
            
        try:
            ftp.login(user=username, passwd=password)
            logger.info("FTP Authentication successful.")
        except Exception as e:
            ftp.close()
            raise ValueError(f"FTP server unavailable: Authentication failed or credentials rejected by {host}. Error: {e}")
            
        try:
            # Typical directory: /products/INSAT-3D/IMAGER/L2B_AOD/YYYY/MM
            remote_dir = f"/products/INSAT-3D/IMAGER/L2B_AOD/{year}/{month}"
            ftp.cwd(remote_dir)
            
            files = ftp.nlst()
            target_prefix = f"3D_AOD_{year}_{month}_{day}"
            matching_files = [f for f in files if f.startswith(target_prefix)]
            
            if not matching_files:
                logger.warning(f"No INSAT-3D AOD files found on FTP for {target_date.strftime('%Y-%m-%d')}.")
                ftp.close()
                return None
                
            filename = matching_files[0]
            local_file = os.path.join(output_dir, filename)
            
            logger.info(f"Downloading {filename} from FTP...")
            with open(local_file, "wb") as f_out:
                ftp.retrbinary(f"RETR {filename}", f_out.write)
                
            ftp.close()
            logger.info(f"Successfully downloaded INSAT-3D file: {local_file}")
            return local_file
        except Exception as e:
            ftp.close()
            raise RuntimeError(f"API unavailable: Failed to fetch directory or files from MOSDAC FTP. Error: {e}")

    def download_modis_fallback(self, target_date, output_dir):
        logger.info("Initializing fallback retrieval: Downloading MODIS MAIAC AOD from Google Earth Engine...")
        
        # Verify GEE authorization
        try:
            import ee
            from backend.config.gee_config import initialize_gee
            success, msg = initialize_gee()
            if not success:
                raise ValueError(f"missing GEE credentials for MODIS MAIAC AOD fallback: {msg}")
        except ImportError:
            raise ValueError("missing GEE credentials for MODIS MAIAC AOD fallback: earthengine-api not installed.")
            
        try:
            # Query MODIS MCD19A2 AOD
            date_str = target_date.strftime("%Y-%m-%d")
            next_date_str = (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            
            bounds = self.config.get("spatial_bounds", {"west": 68.1, "south": 8.4, "east": 97.4, "north": 37.6})
            roi = ee.Geometry.BBox(bounds["west"], bounds["south"], bounds["east"], bounds["north"])
            
            img_coll = (ee.ImageCollection("MODIS/061/MCD19A2_GRT_DNB_L2")
                        .filterDate(date_str, next_date_str)
                        .filterBounds(roi)
                        .select("Optical_Depth_055"))
            
            if int(img_coll.size().getInfo()) == 0:
                logger.warning(f"No MODIS MAIAC AOD granules found for {date_str} over study bounds.")
                return None
                
            aod_img = img_coll.mean().multiply(0.001) # Scale factor
            
            # Since we only download raw data in this script, we export points or a bounding grid
            # Let's save a structured CSV with MODIS AOD to represent raw fallback dataset
            # We sample a resolution matching spatial_resolution from config (e.g. 0.05 deg)
            lon_grid = np.arange(bounds["west"], bounds["east"], self.config["spatial_resolution"])
            lat_grid = np.arange(bounds["south"], bounds["north"], self.config["spatial_resolution"])
            
            # Extract point properties
            # Sample stations coordinates from cpcb_stations.json to make it efficient
            stations_path = os.path.join(os.path.dirname(__file__), "..", "config", "cpcb_stations.json")
            with open(stations_path, "r") as f:
                stations = json.load(f)
                
            points = [ee.Feature(ee.Geometry.Point(s["longitude"], s["latitude"]), {"station_id": s["station_id"]}) for s in stations]
            fc = ee.FeatureCollection(points)
            
            extracted = aod_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.first(),
                scale=5000
            ).getInfo()
            
            rows = []
            for feat in extracted.get("features", []):
                props = feat.get("properties", {})
                aod_val = props.get("first")
                coords = feat["geometry"]["coordinates"]
                rows.append({
                    "date": date_str,
                    "latitude": coords[1],
                    "longitude": coords[0],
                    "station_id": props.get("station_id"),
                    "AOD": aod_val if aod_val is not None else -999.0, # -999 represents cloud/invalid
                    "source": "MODIS_MAIAC"
                })
                
            df_modis = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f"MODIS_MAIAC_AOD_{target_date.strftime('%Y%m%d')}.csv")
            df_modis.to_csv(output_file, index=False)
            logger.info(f"Successfully saved raw fallback MODIS MAIAC dataset to: {output_file}")
            return output_file
        except Exception as e:
            raise RuntimeError(f"API unavailable: Failed to retrieve fallback MODIS AOD from GEE. Error: {e}")

    def run(self, is_test=False):
        start_str = self.config["start_date"]
        end_str = self.config["end_date"]
        if is_test:
            start_str = "2023-01-01"
            end_str = "2023-01-07"
            
        start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
        
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "raw", "insat3d")
        os.makedirs(raw_dir, exist_ok=True)
        
        delta = end_dt - start_dt
        total_days = delta.days + 1
        
        downloaded_paths = []
        
        for i in range(total_days):
            curr_date = start_dt + datetime.timedelta(days=i)
            logger.info(f"Processing download pipeline for date: {curr_date.strftime('%Y-%m-%d')}")
            
            # 1. Primary: Download from MOSDAC FTP
            filepath = self.download_from_mosdac(curr_date, raw_dir)
            
            # 2. Fallback: Download from MODIS GEE
            if not filepath:
                filepath = self.download_modis_fallback(curr_date, raw_dir)
                
            if filepath:
                downloaded_paths.append(filepath)
                
        logger.info(f"INSAT-3D/MODIS download pipeline finished. Total raw datasets stored: {len(downloaded_paths)}")
        return downloaded_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INSAT-3D Aerosol Optical Depth FTP Downloader")
    parser.add_argument("--test", action="store_true", help="Run downloader in test mode (7 days)")
    args = parser.parse_args()
    
    downloader = INSAT3DDownloader()
    downloader.run(is_test=args.test)

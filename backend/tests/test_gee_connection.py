import os
import sys
import logging

# Ensure root directory is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.config.gee_config import initialize_gee, check_gee_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_gee_connection")

COLLECTIONS = {
    "NO2": "COPERNICUS/S5P/OFFL/L3_NO2",
    "SO2": "COPERNICUS/S5P/OFFL/L3_SO2",
    "CO": "COPERNICUS/S5P/OFFL/L3_CO",
    "O3": "COPERNICUS/S5P/OFFL/L3_O3",
    "HCHO": "COPERNICUS/S5P/OFFL/L3_HCHO"
}

def run_tests():
    logger.info("Initializing GEE connection test...")
    
    success, msg = initialize_gee()
    if not success:
        print(f"Google Earth Engine Authentication Failed: {msg}")
        # We can still proceed with simulated check to test code structure
        sys.exit(1)
        
    status = check_gee_connection()
    if status.get("status") != "connected":
        print(f"GEE Connection Verification Failed: {status.get('reason')}")
        sys.exit(1)
        
    print("\nGoogle Earth Engine Connected Successfully\n")
    
    import ee
    
    for pol, coll_id in COLLECTIONS.items():
        try:
            logger.info(f"Testing access to {pol} collection: {coll_id}...")
            collection = (ee.ImageCollection(coll_id)
                          .filterDate("2023-01-01", "2023-01-10"))
            
            count = int(collection.size().getInfo())
            print(f"Sentinel-5P Images Found ({pol}): {count}")
        except Exception as e:
            print(f"Failed to access Sentinel-5P {pol} Dataset: {e}")

if __name__ == "__main__":
    run_tests()

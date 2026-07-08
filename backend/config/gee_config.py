import os
import logging
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger("gee_config")

# Global connection flag
_gee_initialized = False
_gee_init_error = None

def initialize_gee():
    global _gee_initialized, _gee_init_error
    
    project_id = os.getenv("GEE_PROJECT_ID")
    if not project_id:
        msg = "GEE_PROJECT_ID not set in .env file or environment variables"
        logger.warning(msg)
        _gee_initialized = False
        _gee_init_error = msg
        return False, msg
        
    try:
        import ee
        # Initialize with project ID as required
        ee.Initialize(project=project_id)
        logger.info(f"Successfully initialized Google Earth Engine with project: {project_id}")
        _gee_initialized = True
        _gee_init_error = None
        return True, "Success"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Google Earth Engine initialization failed: {error_msg}")
        _gee_initialized = False
        _gee_init_error = error_msg
        return False, error_msg

def check_gee_connection():
    global _gee_initialized, _gee_init_error
    if not _gee_initialized:
        return {
            "status": "not_connected",
            "service": "Google Earth Engine",
            "reason": _gee_init_error or "Initialization was not executed"
        }
        
    try:
        import ee
        # Run a simple query (getting info for a small boundary) to verify remote connection works
        ee.Geometry.Point([77.0, 28.0]).getInfo()
        return {
            "status": "connected",
            "service": "Google Earth Engine"
        }
    except Exception as e:
        logger.error(f"GEE connection verification failed: {e}")
        return {
            "status": "not_connected",
            "service": "Google Earth Engine",
            "reason": str(e)
        }

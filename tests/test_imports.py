import os
import sys

# Ensure the root workspace is in the python path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """
    Validates that all modular pipelines and classes can be loaded.
    """
    print("Testing pipeline module imports...")
    try:
        from utils.logger import setup_logger
        from utils.gee_auth import initialize_gee
        print("[OK] Utils imported successfully.")
        
        from preprocessing.preprocessing_pipeline import CPCBPreprocessor, SpatialAligner, TemporalAligner, DataNormalizer
        from preprocessing.cpcb_pipeline import CPCBDownloader, CPCBFileCleaner, CPCBValidator, CPCBDatabaseCreator
        from preprocessing.data_extractor import GEEDataExtractor, MOSDACDownloader, CPCBDataLoader
        print("[OK] Preprocessing pipelines imported successfully.")
        
        from datasets.gee_pipeline import GEEPipeline
        print("[OK] GEE Dataset pipelines imported successfully.")
        
        from feature_engineering.feature_engineer import FeatureEngineer
        print("[OK] Feature Engineering modules imported successfully.")
        
        from aqi_prediction.cnn_lstm import AQICNNLSTMModel
        from aqi_prediction.train import AQIModelTrainer
        from aqi_prediction.predict import AQIPredictor
        print("[OK] Deep Learning modeling modules imported successfully.")
        
        from hotspot_detection.hotspot_detector import HCHOHotspotDetector
        print("[OK] GIS Hotspot Detection modules imported successfully.")
        
        from transport_analysis.wind_transport import WindTransportAnalyzer
        print("[OK] Meteorological Transport modules imported successfully.")
        
        print("\nAll core modules verified successfully.")
    except Exception as e:
        print(f"[ERROR] Import validation failed: {e}")
        raise e

if __name__ == "__main__":
    test_imports()

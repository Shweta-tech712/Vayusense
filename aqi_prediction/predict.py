import os
import pickle
import yaml
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger
from preprocessing.preprocessing_pipeline import DataNormalizer, SpatialAligner

logger = setup_logger("model_inference")

class AQIPredictor:
    """
    Runs inference over a spatial grid of India to reconstruct
    a continuous Surface AQI prediction map.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_path = self.config['model']['model_save_path']
        self.scaler_path = os.path.join(
            os.path.dirname(self.model_path),
            "scaler.pkl"
        )
        
        # Instantiate spatial aligner
        self.patch_aligner = SpatialAligner(config_path)
        self.patch_size = self.config['model']['spatial_patch_size']
        self.sequence_length = self.config['model']['sequence_length']
        
        self.model = None
        self.scaler = None
        self._load_production_assets()

    def _load_production_assets(self) -> None:
        """
        Loads the saved Keras model and normalizer scaler.
        """
        logger.info("Loading production assets for inference...")
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Production model weights not found at path: {self.model_path}")
                raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
            
            # Load Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Production Keras model loaded successfully.")
            
            # Load normalizer scaler
            self.scaler = DataNormalizer()
            self.scaler.load_scaler(self.scaler_path)
            logger.info("Production normalizer scaler loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load production inference assets: {e}")
            raise e

    def generate_india_meshgrid(self, grid_resolution: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates lat/lon coordinates arrays across the India bounding box.
        grid_resolution: determines grid dimensions (e.g. 50 x 50 cells).
        """
        bbox = self.config['spatial']['india_bbox']
        # Lon: bbox[0] to bbox[2], Lat: bbox[1] to bbox[3]
        lon_grid = np.linspace(bbox[0], bbox[2], grid_resolution)
        lat_grid = np.linspace(bbox[1], bbox[3], grid_resolution)
        
        # Create 2D meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        return lon_mesh, lat_mesh

    def predict_surface_aqi(self, satellite_data_cubes: List[Dict[str, np.ndarray]], grid_resolution: int = 40) -> np.ndarray:
        """
        Predicts surface AQI over a geographic grid.
        satellite_data_cubes: List of length T (7) containing band grids on each day.
        """
        logger.info(f"Starting Surface AQI spatial inference (Grid size: {grid_resolution}x{grid_resolution})...")
        
        # 1. Generate coordinate grid
        lon_mesh, lat_mesh = self.generate_india_meshgrid(grid_resolution)
        h, w = lon_mesh.shape
        
        # Flatten grids to build a list of coordinate tasks
        flat_lons = lon_mesh.flatten()
        flat_lats = lat_mesh.flatten()
        
        # 2. Extract sequences for all coordinate locations
        # Sequence shape: (Samples, Time, Height, Width, Channels)
        sequences = []
        valid_indices = []
        
        # Assume satellite_data_cubes contains daily grids of shape (raster_h, raster_w, bands)
        # Find spatial grid indices corresponding to the coordinates
        # For this demonstration, we simulate coordinate matching to the satellite_data_cubes dimensions
        raster_h, raster_w = satellite_data_cubes[0][list(satellite_data_cubes[0].keys())[0]].shape
        
        for idx in range(len(flat_lons)):
            lon, lat = flat_lons[idx], flat_lats[idx]
            
            # Map lat/lon to raster grid pixel indices (simulated linearly based on bounds)
            bbox = self.config['spatial']['india_bbox']
            col_idx = int(((lon - bbox[0]) / (bbox[2] - bbox[0])) * (raster_w - 1))
            row_idx = int(((lat - bbox[1]) / (bbox[3] - bbox[1])) * (raster_h - 1))
            
            # Bound indexes to grid limits
            col_idx = max(0, min(col_idx, raster_w - 1))
            row_idx = max(0, min(row_idx, raster_h - 1))
            
            # Construct T-day temporal lag tensor
            seq_features = []
            valid_seq = True
            
            for t in range(self.sequence_length):
                bands_dict = satellite_data_cubes[t]
                bands_list = []
                
                for band_name, grid in bands_dict.items():
                    # Extract spatial patch centered at pixel indices
                    patch = self.patch_aligner.extract_numpy_patch(grid, (row_idx, col_idx))
                    bands_list.append(patch)
                    
                day_tensor = np.stack(bands_list, axis=-1)
                seq_features.append(day_tensor)
                
            if valid_seq:
                seq_tensor = np.stack(seq_features, axis=0)
                sequences.append(seq_tensor)
                valid_indices.append(idx)
                
        if not sequences:
            logger.error("No valid sequences could be generated over the meshgrid coordinates.")
            return np.zeros_like(lon_mesh)
            
        # Convert to numpy array: shape (Samples, T, H, W, C)
        X_infer = np.array(sequences)
        
        # 3. Standardize inputs using the production scaler
        X_scaled = self.scaler.transform_sequence(X_infer)
        
        # 4. Run batch model predictions
        logger.info(f"Submitting batch predictions for {len(X_scaled)} grid points...")
        raw_predictions = self.model.predict(X_scaled).flatten()
        
        # 5. Reconstruct 2D raster grid surface
        prediction_grid = np.full(flat_lons.shape, np.nan)
        
        # Apply physical minimum constraint (AQI cannot be negative)
        prediction_grid[valid_indices] = np.clip(raw_predictions, 0.0, 500.0)
        
        # Reshape back to 2D
        aqi_raster = prediction_grid.reshape(h, w)
        logger.info("Surface AQI spatial grid predictions complete.")
        return aqi_raster

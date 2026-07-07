import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from typing import Tuple, List, Dict
from utils.logger import setup_logger
from preprocessing.preprocessing_pipeline import DataNormalizer
from aqi_prediction.cnn_lstm import AQICNNLSTMModel

logger = setup_logger("model_trainer")

class AQIModelTrainer:
    """
    Handles training, spatial cross-validation, and performance evaluation
    for the CNN-LSTM air quality model.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_builder = AQICNNLSTMModel(config_path)
        self.scaler = DataNormalizer()
        logger.info("Initialized AQI model training pipeline.")

    def run_spatial_cross_validation(self, X: np.ndarray, y: np.ndarray, station_groups: np.ndarray, n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Runs GroupKFold cross-validation split by Station Groups
        to ensure no spatial data leakage between training and validation folds.
        X shape: (samples, time_steps, height, width, channels)
        y shape: (samples,)
        """
        logger.info(f"Starting {n_splits}-Fold Spatial Cross Validation split by stations...")
        gkf = GroupKFold(n_splits=n_splits)
        
        metrics = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        fold = 1
        for train_idx, val_idx in gkf.split(X, y, groups=station_groups):
            logger.info(f"--- Processing Fold {fold}/{n_splits} ---")
            
            # Slice fold data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features to prevent leakage (fit on train fold, transform val fold)
            X_train_scaled = self.scaler.fit_transform_sequence(X_train)
            X_val_scaled = self.scaler.transform_sequence(X_val)
            
            # Build and compile model instance
            num_channels = X.shape[-1]
            model = self.model_builder.build_model(num_channels=num_channels)
            
            # Train model
            logger.info(f"Fitting model on {len(X_train)} training sequences...")
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                batch_size=self.config['model']['batch_size'],
                epochs=self.config['model']['epochs'],
                callbacks=self.model_builder.get_callbacks(),
                verbose=1
            )
            
            # Evaluate on validation split
            logger.info("Evaluating performance on validation fold...")
            predictions = model.predict(X_val_scaled).flatten()
            
            mae = mean_absolute_error(y_val, predictions)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            r2 = r2_score(y_val, predictions)
            
            logger.info(f"Fold {fold} Results: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
            
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            metrics['r2'].append(r2)
            
            fold += 1
            
        # Log aggregated cross-validation metrics
        logger.info("--- Spatial Cross-Validation Completed ---")
        logger.info(f"Aggregated Performance (Mean ± Std):")
        logger.info(f"MAE:  {np.mean(metrics['mae']):.2f} ± {np.std(metrics['mae']):.2f}")
        logger.info(f"RMSE: {np.mean(metrics['rmse']):.2f} ± {np.std(metrics['rmse']):.2f}")
        logger.info(f"R2:   {np.mean(metrics['r2']):.2f} ± {np.std(metrics['r2']):.2f}")
        
        return metrics

    def train_final_production_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the final production model on the entire dataset and serializes
        the weights and data normalizer to disk.
        """
        logger.info("Training final production model on full dataset...")
        
        # Scale entire dataset
        X_scaled = self.scaler.fit_transform_sequence(X)
        
        # Save scaler state for inference pipelines
        scaler_save_path = os.path.join(
            os.path.dirname(self.config['model']['model_save_path']),
            "scaler.pkl"
        )
        self.scaler.save_scaler(scaler_save_path)
        
        # Build model
        num_channels = X.shape[-1]
        model = self.model_builder.build_model(num_channels=num_channels)
        
        # Train on entire dataset (no early stopping callbacks to ensure full fit)
        # Checkpoint is kept to monitor loss metrics
        model.fit(
            X_scaled, y,
            batch_size=self.config['model']['batch_size'],
            epochs=self.config['model']['epochs'] - 10, # Reduce epochs slightly since no early stopping
            verbose=1
        )
        
        # Serialize Keras Model to disk
        model_save_path = self.config['model']['model_save_path']
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logger.info(f"Final production model weights saved successfully to: {model_save_path}")

"""
backend/services/model_service.py

Singleton that loads the trained CNN-LSTM model and MinMaxScaler exactly once
at FastAPI startup. All prediction requests share the same warm in-memory objects.
"""
import os
import json
import pickle
import logging
import threading
import numpy as np

logger = logging.getLogger("model_service")

_BASE = os.path.dirname(__file__)          # backend/services/
_BACKEND = os.path.abspath(os.path.join(_BASE, ".."))   # backend/
_PROJECT = os.path.abspath(os.path.join(_BACKEND, "..")) # project root

MODEL_PATH   = os.path.join(_BACKEND, "models", "v1", "cnn_lstm_aqi_model.keras")
SCALER_PATH  = os.path.join(_PROJECT, "models", "scalers", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(_PROJECT, "models", "scalers", "target_scaler.pkl")
META_PATH    = os.path.join(_BACKEND, "models", "v1", "model_metadata.json")
FEAT_PATH    = os.path.join(_PROJECT, "models", "metadata", "feature_names.json")


class ModelService:
    """Thread-safe singleton for Keras model + MinMaxScaler."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._model = None
                obj._scaler = None
                obj._target_scaler = None
                obj._metadata = {}
                obj._feature_names = []
                obj._loaded = False
                cls._instance = obj
        return cls._instance

    # ---------- public API ----------

    @classmethod
    def instance(cls) -> "ModelService":
        return cls()

    def load(self):
        """Load model + scaler from disk. Safe to call multiple times."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_model()
            self._load_scaler()
            self._load_metadata()
            self._loaded = True
            logger.info("ModelService: CNN-LSTM model and scaler loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def metadata(self):
        return self._metadata

    def predict(self, sequence: np.ndarray) -> dict:
        """
        Parameters
        ----------
        sequence : np.ndarray  shape (1, seq_len, n_features)  already scaled

        Returns
        -------
        dict  {"AQI": float, "PM25": float, "hcho_column": float, "hcho_hotspot_probability": float}
        """
        self._require_loaded()
        preds = self._model.predict(sequence, verbose=0)
        # Multi-output model returns a list of arrays
        aqi_raw = float(preds[0].flatten()[0])
        pm25_raw = float(preds[1].flatten()[0])
        hcho_raw = float(np.clip(preds[2].flatten()[0], 0.0, 1.0))
        
        # Target inverse transform for AQI and PM2.5
        scaled_targets = np.array([[aqi_raw, pm25_raw]])
        inverse_targets = self._target_scaler.inverse_transform(scaled_targets)
        
        aqi = float(np.clip(inverse_targets[0, 0], 0.0, 500.0))
        pm25 = float(np.clip(inverse_targets[0, 1], 0.0, 999.0))
        
        # Calculate HCHO column density from predicted probability
        hcho_col = float(hcho_raw * 0.004)
        
        logger.info(f"[MODEL_RAW_OUTPUT]\nAQI: {aqi_raw:.4f}\nPM25: {pm25_raw:.4f}\nHCHO: {hcho_raw:.4f}")
        logger.info(f"[MODEL_FINAL_OUTPUT]\nAQI: {aqi:.1f}\nPM25: {pm25:.1f}\nHCHO: {hcho_raw:.4f}")
        
        return {
            "AQI": round(aqi, 1),
            "PM25": round(pm25, 1),
            "hcho_column": round(hcho_col, 6),
            "hcho_hotspot_probability": round(hcho_raw, 4)
        }

    def transform(self, raw_sequence: np.ndarray) -> np.ndarray:
        """Scale a raw (1, seq_len, n_features) array using the fitted scaler."""
        self._require_loaded()
        s, t, f = raw_sequence.shape
        flat = raw_sequence.reshape(-1, f)
        scaled = self._scaler.transform(flat)
        return scaled.reshape(s, t, f)

    # ---------- private helpers ----------

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run backend/training/train_cnn_lstm.py first."
            )
        import tensorflow as tf
        self._model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Loaded Keras model from {MODEL_PATH}")

    def _load_scaler(self):
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Feature scaler not found at {SCALER_PATH}. "
                "Run backend/preprocessing/dataset_fusion.py first."
            )
        with open(SCALER_PATH, "rb") as f:
            self._scaler = pickle.load(f)
        logger.info(f"Loaded MinMaxScaler from {SCALER_PATH}")

        if not os.path.exists(TARGET_SCALER_PATH):
            raise FileNotFoundError(
                f"Target scaler not found at {TARGET_SCALER_PATH}. "
                "Run backend/preprocessing/dataset_fusion.py first."
            )
        with open(TARGET_SCALER_PATH, "rb") as f:
            self._target_scaler = pickle.load(f)
        logger.info(f"Loaded target MinMaxScaler from {TARGET_SCALER_PATH}")

    def _load_metadata(self):
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                self._metadata = json.load(f)
        if os.path.exists(FEAT_PATH):
            with open(FEAT_PATH, "r") as f:
                self._feature_names = json.load(f).get("features", [])
        logger.info(f"Loaded model metadata. Features: {len(self._feature_names)}")

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "ModelService has not been loaded yet. "
                "Call ModelService.instance().load() at application startup."
            )

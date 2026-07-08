"""
backend/services/prediction_service.py

Orchestrates end-to-end prediction:
  location string / lat-lon
      → geocoding
      → nearest data row lookup (from fused CSV)
      → 7-day sequence construction
      → MinMaxScaler transform
      → CNN-LSTM model inference
      → AQI classification, confidence score, feature contribution
      → structured JSON response
"""
import os
import json
import datetime
import logging
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Optional

from backend.services.model_service import ModelService
from backend.services.cache_service import CacheService
from backend.explainability.explain_service import ExplainService

logger = logging.getLogger("prediction_service")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE    = os.path.dirname(__file__)            # backend/services
_BACKEND = os.path.abspath(os.path.join(_BASE, ".."))   # backend/
FUSED_CSV = os.path.join(_BACKEND, "datasets", "final", "v1", "aqi_training_dataset.csv")
EVAL_JSON = os.path.join(_BACKEND, "reports", "model_evaluation.json")

# ─────────────────────────────────────────────────────────────────────────────
# Static geocoding table (city / state → lat, lon, display name)
# ─────────────────────────────────────────────────────────────────────────────
LOCATION_DB = {
    "delhi":        (28.6139, 77.2090, "Delhi",        "Delhi NCR"),
    "new delhi":    (28.6139, 77.2090, "New Delhi",    "Delhi NCR"),
    "mumbai":       (19.0760, 72.8777, "Mumbai",       "Maharashtra"),
    "pune":         (18.5204, 73.8567, "Pune",         "Maharashtra"),
    "bengaluru":    (12.9716, 77.5946, "Bengaluru",    "Karnataka"),
    "bangalore":    (12.9716, 77.5946, "Bengaluru",    "Karnataka"),
    "kolkata":      (22.5726, 88.3639, "Kolkata",      "West Bengal"),
    "chennai":      (13.0827, 80.2707, "Chennai",      "Tamil Nadu"),
    "hyderabad":    (17.3850, 78.4867, "Hyderabad",    "Telangana"),
    "patna":        (25.5941, 85.1376, "Patna",        "Bihar"),
    "indore":       (22.7196, 75.8577, "Indore",       "Madhya Pradesh"),
    "bhopal":       (23.2599, 77.4126, "Bhopal",       "Madhya Pradesh"),
    "lucknow":      (26.8467, 80.9462, "Lucknow",      "Uttar Pradesh"),
    "chandigarh":   (30.7333, 76.7794, "Chandigarh",   "Chandigarh UT"),
    "amritsar":     (31.6340, 74.8723, "Amritsar",     "Punjab"),
    "ludhiana":     (30.9010, 75.8573, "Ludhiana",     "Punjab"),
    "jaipur":       (26.9124, 75.7873, "Jaipur",       "Rajasthan"),
    "ahmedabad":    (23.0225, 72.5714, "Ahmedabad",    "Gujarat"),
    "surat":        (21.1702, 72.8311, "Surat",        "Gujarat"),
    "nagpur":       (21.1458, 79.0882, "Nagpur",       "Maharashtra"),
    "agra":         (27.1767, 78.0081, "Agra",         "Uttar Pradesh"),
    "varanasi":     (25.3176, 82.9739, "Varanasi",     "Uttar Pradesh"),
    "kanpur":       (26.4499, 80.3319, "Kanpur",       "Uttar Pradesh"),
    "guwahati":     (26.1445, 91.7362, "Guwahati",     "Assam"),
    "bhubaneswar":  (20.2961, 85.8245, "Bhubaneswar",  "Odisha"),
    "visakhapatnam":(17.6868, 83.2185, "Visakhapatnam","Andhra Pradesh"),
    "kochi":        (9.9312,  76.2673, "Kochi",        "Kerala"),
    "thiruvananthapuram": (8.5241, 76.9366, "Thiruvananthapuram", "Kerala"),
    # states (centroid)
    "punjab":       (30.9010, 75.8573, "Punjab (state)", "Punjab"),
    "maharashtra":  (19.7515, 75.7139, "Maharashtra (state)", "Maharashtra"),
    "rajasthan":    (27.0238, 74.2179, "Rajasthan (state)", "Rajasthan"),
    "uttar pradesh":(26.8467, 80.9462, "Uttar Pradesh (state)", "Uttar Pradesh"),
    "up":           (26.8467, 80.9462, "Uttar Pradesh", "Uttar Pradesh"),
    "gujarat":      (22.2587, 71.1924, "Gujarat (state)", "Gujarat"),
    "karnataka":    (15.3173, 75.7139, "Karnataka (state)", "Karnataka"),
}

SEQUENCE_LENGTH = 7

# ─────────────────────────────────────────────────────────────────────────────
# AQI classification helpers
# ─────────────────────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    (50,  "Good",          "Air quality is satisfactory with little or no risk."),
    (100, "Satisfactory",  "Acceptable quality; mild concern for sensitive groups."),
    (200, "Moderate",      "Moderate outdoor caution advised for sensitive individuals."),
    (300, "Poor",          "Reduce outdoor exposure; keep windows closed."),
    (400, "Very Poor",     "Avoid prolonged outdoor activity; use N95 masks."),
    (500, "Severe",        "Stay indoors; health emergency conditions possible."),
]

def classify_aqi(aqi: float) -> tuple[str, str]:
    for threshold, category, advice in AQI_CATEGORIES:
        if aqi <= threshold:
            return category, advice
    return "Severe", AQI_CATEGORIES[-1][2]

def classify_hcho(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"

def classify_aerosol(aod: float) -> str:
    if aod >= 0.6:
        return "High"
    if aod >= 0.3:
        return "Medium"
    return "Low"

def classify_fire(fire_count: float) -> str:
    if fire_count >= 10:
        return "High"
    if fire_count >= 3:
        return "Medium"
    return "Low"

def wind_bearing_to_cardinal(deg: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[int((deg % 360) / 22.5) % 16]


# ─────────────────────────────────────────────────────────────────────────────
# Confidence score calculation
# ─────────────────────────────────────────────────────────────────────────────
def compute_confidence(raw_seq: np.ndarray, eval_metrics: dict) -> float:
    """
    Confidence = 0.5 × model_val_score
               + 0.3 × input_completeness
               + 0.2 × data_quality_score
    """
    # 1. Model validation score from saved evaluation report
    try:
        r2 = float(eval_metrics.get("aqi", {}).get("r2", 0.5))
        model_score = float(np.clip(r2, 0.0, 1.0))
    except Exception:
        model_score = 0.5

    # 2. Input completeness: fraction of non-NaN values in sequence
    total_vals = raw_seq.size
    valid_vals = int(np.isfinite(raw_seq).sum())
    completeness = valid_vals / max(1, total_vals)

    # 3. Data quality: score on how many timesteps are available (full seq = 1.0)
    quality = min(1.0, raw_seq.shape[1] / SEQUENCE_LENGTH)

    confidence = 0.50 * model_score + 0.30 * completeness + 0.20 * quality
    return round(float(np.clip(confidence, 0.0, 1.0)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main service
# ─────────────────────────────────────────────────────────────────────────────
class PredictionService:
    def __init__(self):
        self._model_svc  = ModelService.instance()
        self._cache      = CacheService.instance()
        self._fused_df: Optional[pd.DataFrame] = None
        self._kdtree: Optional[KDTree] = None
        self._era5_df: Optional[pd.DataFrame] = None
        self._era5_kdtree: Optional[KDTree] = None
        self._explain_svc: Optional[ExplainService] = None
        self._eval_metrics: dict = {}
        self._dataset_version = "v1"
        self._load_fused_data()
        self._load_era5_data()
        self._load_eval_metrics()

    # ---------- public API ----------

    def predict(self, location: str,
                lat: Optional[float] = None,
                lon: Optional[float] = None) -> dict:
        cache_key = f"{location}_{lat}_{lon}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        loc_meta = self._resolve_location(location, lat, lon)
        raw_seq  = self._build_sequence(loc_meta["latitude"], loc_meta["longitude"])
        scaled   = self._model_svc.transform(raw_seq)
        preds    = self._model_svc.predict(scaled)

        confidence = compute_confidence(raw_seq, self._eval_metrics)
        contribution = self._explain_svc.explain(raw_seq) if self._explain_svc else {}

        # Query nearest ERA5 weather values
        weather = self._get_nearest_weather(loc_meta["latitude"], loc_meta["longitude"])

        result = self._build_response(loc_meta, preds, raw_seq, confidence, contribution, weather)
        self._cache.set(cache_key, result)
        return result

    # ---------- private helpers ----------

    def _load_fused_data(self):
        if not os.path.exists(FUSED_CSV):
            logger.warning(f"Fused dataset not found at {FUSED_CSV}. "
                           "Sequences will use zero-filled fallback values.")
            return
        df = pd.read_csv(FUSED_CSV)
        df = df.select_dtypes(include=[np.number]).dropna(how="all")
        coords = df[["latitude", "longitude"]].drop_duplicates().values
        self._kdtree    = KDTree(coords)
        self._fused_df  = df
        logger.info(f"PredictionService: loaded fused dataset {df.shape} rows.")

        # Initialise ExplainService now that we know feature names
        self._explain_svc = ExplainService(self._model_svc.feature_names)

    def _load_era5_data(self):
        era5_path = os.path.abspath(os.path.join(_BACKEND, "datasets", "processed", "era5_processed.csv"))
        if not os.path.exists(era5_path):
            logger.warning(f"ERA5 processed dataset not found at {era5_path}.")
            return
        try:
            df = pd.read_csv(era5_path)
            coords = df[["latitude", "longitude"]].drop_duplicates().values
            self._era5_kdtree = KDTree(coords)
            self._era5_df = df
            logger.info(f"PredictionService: loaded ERA5 dataset {df.shape} rows.")
        except Exception as e:
            logger.error(f"Failed to load ERA5 processed dataset: {e}")

    def _get_nearest_weather(self, lat: float, lon: float) -> dict:
        # 1. Search ERA5 processed dataset
        if self._era5_df is not None and self._era5_kdtree is not None:
            try:
                dist, idx = self._era5_kdtree.query([lat, lon])
                if dist < 0.15:
                    unique_coords = self._era5_df[["latitude", "longitude"]].drop_duplicates().values
                    near_lat, near_lon = unique_coords[idx]
                    
                    sub_df = self._era5_df[
                        (np.abs(self._era5_df["latitude"] - near_lat) < 0.01) &
                        (np.abs(self._era5_df["longitude"] - near_lon) < 0.01)
                    ]
                    if not sub_df.empty:
                        latest_row = sub_df.sort_values("date").iloc[-1]
                        
                        wind_deg_val = latest_row.get("wind_direction")
                        wind_cardinal = wind_bearing_to_cardinal(float(wind_deg_val)) if pd.notna(wind_deg_val) else None
                        
                        temp_val = latest_row.get("temperature_mean")
                        humidity_val = latest_row.get("humidity")
                        pressure_val = latest_row.get("pressure")
                        rainfall_val = latest_row.get("rainfall")
                        wind_speed_val = latest_row.get("wind_speed")
                        blh_val = latest_row.get("boundary_layer_height")
                        
                        return {
                            "available": True,
                            "temperature": float(round(temp_val, 1)) if pd.notna(temp_val) else None,
                            "humidity": float(round(humidity_val, 1)) if pd.notna(humidity_val) else None,
                            "pressure": float(round(pressure_val, 1)) if pd.notna(pressure_val) else None,
                            "rainfall": float(round(rainfall_val, 2)) if pd.notna(rainfall_val) else None,
                            "wind_speed": float(round(wind_speed_val, 1)) if pd.notna(wind_speed_val) else None,
                            "wind_direction": wind_cardinal,
                            "boundary_layer_height": float(round(blh_val, 1)) if pd.notna(blh_val) else None
                        }
            except Exception as e:
                logger.error(f"Error querying ERA5 weather parameters: {e}")

        # 2. Search fused dataset
        if self._fused_df is not None and self._kdtree is not None:
            try:
                dist, idx = self._kdtree.query([lat, lon])
                if dist < 0.15:
                    unique_coords = self._fused_df[["latitude", "longitude"]].drop_duplicates().values
                    near_lat, near_lon = unique_coords[idx]
                    
                    sub_df = self._fused_df[
                        (np.abs(self._fused_df["latitude"] - near_lat) < 0.01) &
                        (np.abs(self._fused_df["longitude"] - near_lon) < 0.01)
                    ]
                    if not sub_df.empty:
                        latest_row = sub_df.sort_values("date" if "date" in self._fused_df.columns else self._fused_df.columns[0]).iloc[-1]
                        
                        wind_deg_val = latest_row.get("wind_direction")
                        wind_cardinal = wind_bearing_to_cardinal(float(wind_deg_val)) if pd.notna(wind_deg_val) else None
                        
                        temp_val = latest_row.get("temperature_mean")
                        humidity_val = latest_row.get("humidity")
                        pressure_val = latest_row.get("pressure")
                        rainfall_val = latest_row.get("rainfall")
                        wind_speed_val = latest_row.get("wind_speed")
                        blh_val = latest_row.get("boundary_layer_height")
                        
                        return {
                            "available": True,
                            "temperature": float(round(temp_val, 1)) if pd.notna(temp_val) else None,
                            "humidity": float(round(humidity_val, 1)) if pd.notna(humidity_val) else None,
                            "pressure": float(round(pressure_val, 1)) if pd.notna(pressure_val) else None,
                            "rainfall": float(round(rainfall_val, 2)) if pd.notna(rainfall_val) else None,
                            "wind_speed": float(round(wind_speed_val, 1)) if pd.notna(wind_speed_val) else None,
                            "wind_direction": wind_cardinal,
                            "boundary_layer_height": float(round(blh_val, 1)) if pd.notna(blh_val) else None
                        }
            except Exception as e:
                logger.error(f"Error querying fused weather parameters: {e}")

        # 3. If unavailable
        return {"available": False}

    def _load_eval_metrics(self):
        if os.path.exists(EVAL_JSON):
            with open(EVAL_JSON, "r") as f:
                self._eval_metrics = json.load(f)

    def _resolve_location(self, location: str,
                          lat: Optional[float],
                          lon: Optional[float]) -> dict:
        # 1. Direct lat/lon string  e.g. "18.52,73.85"
        if "," in location and lat is None:
            try:
                parts = location.split(",")
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return {"name": f"{lat:.4f}°N {lon:.4f}°E",
                        "state": "Unknown",
                        "latitude": lat, "longitude": lon}
            except ValueError:
                pass

        # 2. Lat/lon passed explicitly
        if lat is not None and lon is not None:
            display = location.strip().title() if location.strip() else f"{lat:.3f},{lon:.3f}"
            return {"name": display, "state": "Unknown",
                    "latitude": lat, "longitude": lon}

        # 3. Named lookup
        key = location.lower().strip()
        for db_key, (db_lat, db_lon, db_name, db_state) in LOCATION_DB.items():
            if db_key in key or key in db_key:
                return {"name": db_name, "state": db_state,
                        "latitude": db_lat, "longitude": db_lon}

        # 4. Unknown city — India centroid fallback
        logger.warning(f"Location '{location}' not found in geocoding DB. Using India centroid.")
        raise ValueError(
            f"City '{location}' not found. "
            "Provide a known Indian city/state name or explicit latitude,longitude."
        )

    def _build_sequence(self, lat: float, lon: float) -> np.ndarray:
        """Return a (1, 7, n_features) raw (unscaled) numpy array."""
        feat_names = self._model_svc.feature_names
        n_feat = len(feat_names) if feat_names else 21

        if self._fused_df is None or self._kdtree is None:
            # Fallback: zero sequence when fused dataset is unavailable
            logger.warning("Fused dataset unavailable — returning zero-filled sequence.")
            return np.zeros((1, SEQUENCE_LENGTH, n_feat), dtype=np.float32)

        # Find the nearest recorded station coordinates
        _, idx = self._kdtree.query([lat, lon])
        unique_coords = self._fused_df[["latitude", "longitude"]].drop_duplicates().values
        near_lat, near_lon = unique_coords[idx]

        station_df = self._fused_df[
            (np.abs(self._fused_df["latitude"]  - near_lat) < 0.01) &
            (np.abs(self._fused_df["longitude"] - near_lon) < 0.01)
        ].sort_values("date" if "date" in self._fused_df.columns else self._fused_df.columns[0])

        # Select available feature columns
        available = [f for f in feat_names if f in station_df.columns]
        missing   = [f for f in feat_names if f not in station_df.columns]

        if missing:
            logger.debug(f"Missing features filled with 0: {missing}")

        rows = station_df[available].tail(SEQUENCE_LENGTH).values.astype(np.float32)

        # Pad if fewer than 7 rows are available
        if rows.shape[0] < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - rows.shape[0], len(available)), dtype=np.float32)
            rows = np.vstack([pad, rows])

        # Insert zero columns for missing features in the correct position
        full = np.zeros((SEQUENCE_LENGTH, n_feat), dtype=np.float32)
        for col_i, fn in enumerate(feat_names):
            if fn in available:
                src_i = available.index(fn)
                full[:, col_i] = rows[:, src_i]

        return full.reshape(1, SEQUENCE_LENGTH, n_feat)

    def _build_response(self, loc_meta: dict, preds: dict,
                        raw_seq: np.ndarray,
                        confidence: float, contribution: dict, weather: dict) -> dict:
        aqi   = preds["AQI"]
        pm25  = preds["PM25"]
        hcho  = preds["hcho_hotspot_probability"]

        category, recommendation = classify_aqi(aqi)
        hcho_risk    = classify_hcho(hcho)

        # Pull environmental proxies from last row of sequence
        last_row = raw_seq[0, -1, :]                 # shape (n_feat,)
        feat_names = self._model_svc.feature_names

        def _feat(name: str, default: float = 0.0) -> float:
            try:
                return float(last_row[feat_names.index(name)])
            except (ValueError, IndexError):
                return default

        aod_val    = _feat("AOD", 0.3)
        wind_deg   = _feat("wind_direction", 0.0)
        fire_count = _feat("fire_count", 0.0)

        aerosol_level  = classify_aerosol(aod_val)
        fire_influence = classify_fire(fire_count)
        wind_dir       = wind_bearing_to_cardinal(wind_deg)

        model_meta = self._model_svc.metadata
        return {
            "location": {
                "name":      loc_meta["name"],
                "state":     loc_meta["state"],
                "latitude":  loc_meta["latitude"],
                "longitude": loc_meta["longitude"],
            },
            "prediction": {
                "AQI":                      round(aqi, 1),
                "category":                 category,
                "PM25":                     round(pm25, 1),
                "HCHO_probability":         round(hcho, 4),
                "hcho_hotspot_probability": round(hcho, 4),
                "hcho_column":             round(preds.get("hcho_column", 0.0), 6),
                "hcho_column_density":     round(preds.get("hcho_column", 0.0), 6),
                "HCHO_risk":                hcho_risk,
                "hcho_risk":                hcho_risk,
                "confidence_score":         confidence,
            },
            "environment": {
                "fire_influence":    fire_influence,
                "wind_transport":    wind_dir,
                "aerosol_level":     aerosol_level,
                "available":         weather.get("available", False),
                "temperature":       weather.get("temperature"),
                "humidity":          weather.get("humidity"),
                "pressure":          weather.get("pressure"),
                "rainfall":          weather.get("rainfall"),
                "wind_speed":        weather.get("wind_speed"),
                "wind_direction":    weather.get("wind_direction"),
                "boundary_layer_height": weather.get("boundary_layer_height"),
            },
            "explainability": {
                "feature_contribution": contribution,
            },
            "recommendation": recommendation,
            "metadata": {
                "model_version":   model_meta.get("model_version", "v1.0.0"),
                "dataset_version": self._dataset_version,
                "prediction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            },
            "satellite_features": {
                "NO2": _feat("NO2", 0.0),
                "SO2": _feat("SO2", 0.0),
                "CO": _feat("CO", 0.0),
                "O3": _feat("O3", 0.0),
                "HCHO": _feat("HCHO", 0.0)
            }
        }

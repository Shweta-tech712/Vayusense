"""
backend/explainability/explain_service.py

Computes feature-group contribution percentages from a raw input sequence.

Architecture note
-----------------
The current implementation is a rule-based "permutation-importance proxy"
derived from domain knowledge:
  • Satellite gases  (NO2, SO2, CO, O3, HCHO)
  • Aerosol          (AOD, humidity_corrected_AOD, aod_boundary_layer_ratio)
  • Weather          (temp, humidity, rainfall, wind, BLH, ventilation_index)
  • Fire             (fire_count, FRP, fire_severity, transport_influence_score)
  • Temporal/Lag     (season, day_of_year, lags)

This design is SHAP-ready: replace _rule_based_contribution() with a call to
shap.DeepExplainer or shap.GradientExplainer once the full training dataset
is available.
"""
import os
import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger("explain_service")

# Feature group membership (must match feature_names order in feature_names.json)
FEATURE_GROUPS: Dict[str, List[str]] = {
    "Satellite": ["NO2", "SO2", "CO", "O3", "HCHO"],
    "AOD":       ["AOD", "humidity_corrected_AOD", "aod_boundary_layer_ratio", "ventilation_index"],
    "Weather":   ["temperature_mean", "humidity", "rainfall", "wind_speed",
                  "wind_direction", "boundary_layer_height"],
    "Fire":      ["fire_count", "FRP", "fire_severity_index", "transport_influence_score"],
    "Temporal":  ["season", "day_of_year", "AQI_lag_1", "AQI_lag_3", "PM25_lag_1"],
}

BACKGROUND_PATH = os.path.join(
    os.path.dirname(__file__), "background_samples.npy"
)


class ExplainService:
    """Computes feature contribution percentages for a single prediction."""

    def __init__(self, feature_names: List[str]):
        self._feature_names = feature_names
        self._group_indices: Dict[str, List[int]] = self._build_group_indices()
        self._background: np.ndarray | None = self._load_background()

    # ---------- public API ----------

    def explain(self, raw_sequence: np.ndarray) -> Dict[str, float]:
        """
        Parameters
        ----------
        raw_sequence : np.ndarray  shape (1, seq_len, n_features)  **unscaled**

        Returns
        -------
        dict  group_name → percentage (float, sums to 100)
        """
        # Use the mean of the temporal axis to get a representative feature vector
        mean_vec = raw_sequence[0].mean(axis=0)   # shape (n_features,)
        contributions = self._rule_based_contribution(mean_vec)
        return contributions

    # ---------- private helpers ----------

    def _build_group_indices(self) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for group, names in FEATURE_GROUPS.items():
            indices = [i for i, fn in enumerate(self._feature_names) if fn in names]
            if indices:
                mapping[group] = indices
        return mapping

    def _load_background(self):
        if os.path.exists(BACKGROUND_PATH):
            bg = np.load(BACKGROUND_PATH)
            logger.info(f"ExplainService: loaded background samples {bg.shape}")
            return bg
        logger.warning("ExplainService: background_samples.npy not found. SHAP will not be available.")
        return None

    def _rule_based_contribution(self, mean_vec: np.ndarray) -> Dict[str, float]:
        """
        Estimate contribution as: group_mean_abs_value × group_weight_prior.

        Prior weights (domain-derived, from published AQI literature):
          Satellite: 0.30  (primary pollutant tracers)
          AOD:       0.25  (best PM2.5 proxy from satellite)
          Weather:   0.25  (boundary layer drives dispersion)
          Fire:      0.15  (episodic but high-impact)
          Temporal:  0.05  (stationarity / lag signal)
        """
        PRIORS = {
            "Satellite": 0.30,
            "AOD":       0.25,
            "Weather":   0.25,
            "Fire":      0.15,
            "Temporal":  0.05,
        }

        raw_scores: Dict[str, float] = {}
        for group, indices in self._group_indices.items():
            group_vals = mean_vec[indices]
            # l1 magnitude scaled by prior
            magnitude = float(np.mean(np.abs(group_vals))) if len(group_vals) else 0.0
            raw_scores[group] = magnitude * PRIORS.get(group, 0.05)

        total = sum(raw_scores.values()) or 1.0
        percentages = {g: round((v / total) * 100, 1) for g, v in raw_scores.items()}

        # Ensure all five groups are present in output even if some features are absent
        for g in FEATURE_GROUPS:
            if g not in percentages:
                percentages[g] = 0.0

        return percentages

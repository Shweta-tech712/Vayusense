import os
import sys
import unittest
import numpy as np

# Ensure root directory is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService

class TestPredictionSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize and load model
        ModelService.instance().load()
        cls.pred_service = PredictionService()

    def test_delhi_high_pollution(self):
        # Delhi is historically highly polluted
        res = self.pred_service.predict("Delhi")
        aqi = res["prediction"]["AQI"]
        pm25 = res["prediction"]["PM25"]
        category = res["prediction"]["category"]
        
        print(f"\nDelhi prediction: AQI={aqi}, PM2.5={pm25}, Category={category}")
        
        # AQI should be in realistic high-range
        self.assertTrue(50 < aqi <= 500, f"Delhi AQI {aqi} is not realistic.")
        self.assertTrue(25 < pm25 <= 999, f"Delhi PM2.5 {pm25} is not realistic.")
        self.assertIn(category, ["Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])

    def test_kochi_clean_region(self):
        # Kochi is historically cleaner
        res = self.pred_service.predict("Kochi")
        aqi = res["prediction"]["AQI"]
        pm25 = res["prediction"]["PM25"]
        category = res["prediction"]["category"]
        
        print(f"\nKochi prediction: AQI={aqi}, PM2.5={pm25}, Category={category}")
        
        # AQI should be in realistic lower range compared to Delhi
        self.assertTrue(aqi < 250, f"Kochi AQI {aqi} is unexpectedly severe.")
        self.assertIn(category, ["Good", "Satisfactory", "Moderate", "Poor"])

    def test_hcho_separation(self):
        # Verify HCHO outputs are separated
        res = self.pred_service.predict("Mumbai")
        pred = res["prediction"]
        
        self.assertIn("hcho_hotspot_probability", pred)
        self.assertIn("hcho_column", pred)
        self.assertTrue(0.0 <= pred["hcho_hotspot_probability"] <= 1.0)
        self.assertTrue(pred["hcho_column"] >= 0.0)
        # Check matching relation: hcho_column = hcho_hotspot_probability * 0.004
        self.assertAlmostEqual(pred["hcho_column"], pred["hcho_hotspot_probability"] * 0.004, places=6)

if __name__ == "__main__":
    unittest.main()

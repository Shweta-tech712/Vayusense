import os
import sys
import unittest
import numpy as np

# Ensure root directory is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService

class TestPredictionValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ModelService.instance().load()
        cls.pred_service = PredictionService()

    def test_indian_cities_ranges(self):
        cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Kochi', 'Kolkata', 'Chennai']
        
        for city in cities:
            print(f"\nValidating predictions for {city}...")
            res = self.pred_service.predict(city)
            pred = res["prediction"]
            
            aqi = pred["AQI"]
            pm25 = pred["PM25"]
            hcho_prob = pred["hcho_hotspot_probability"]
            hcho_col = pred["hcho_column_density"]
            
            print(f"[{city}] AQI={aqi}, PM2.5={pm25}, HCHO Prob={hcho_prob}, HCHO Column={hcho_col}")
            
            # Assert no negative values
            self.assertTrue(aqi >= 0.0, f"{city} AQI is negative: {aqi}")
            self.assertTrue(pm25 >= 0.0, f"{city} PM2.5 is negative: {pm25}")
            self.assertTrue(hcho_prob >= 0.0, f"{city} HCHO probability is negative: {hcho_prob}")
            self.assertTrue(hcho_col >= 0.0, f"{city} HCHO column density is negative: {hcho_col}")
            
            # Assert logical ranges
            self.assertTrue(aqi <= 500.0, f"{city} AQI {aqi} exceeds 500")
            self.assertTrue(pm25 <= 999.0, f"{city} PM2.5 {pm25} exceeds 999")
            self.assertTrue(hcho_prob <= 1.0, f"{city} HCHO probability {hcho_prob} exceeds 1.0")

if __name__ == "__main__":
    unittest.main()

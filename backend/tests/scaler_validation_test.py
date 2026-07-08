import os
import sys
import unittest
import numpy as np

# Ensure root directory is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService

class TestScalerValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ModelService.instance().load()
        cls.pred_service = PredictionService()

    def test_scaler_transformation(self):
        # We can directly pass a zero sequence to the model prediction flow
        # feature dimension is 21 or based on config
        n_features = len(ModelService.instance().feature_names)
        dummy_seq = np.zeros((1, 7, n_features), dtype=np.float32)
        
        # Get raw prediction by manually querying model
        raw_preds = ModelService.instance()._model.predict(dummy_seq, verbose=0)
        raw_aqi = float(raw_preds[0].flatten()[0])
        raw_pm25 = float(raw_preds[1].flatten()[0])
        
        # Get final inverse scaled prediction
        final_preds = ModelService.instance().predict(dummy_seq)
        final_aqi = final_preds["AQI"]
        final_pm25 = final_preds["PM25"]
        hcho_prob = final_preds["hcho_hotspot_probability"]
        
        print(f"\n[Raw Outputs] AQI: {raw_aqi:.4f}, PM2.5: {raw_pm25:.4f}")
        print(f"[Final Outputs] AQI: {final_aqi:.1f}, PM2.5: {final_pm25:.1f}")
        
        # Assert raw does not equal final
        self.assertNotEqual(raw_aqi, final_aqi)
        self.assertNotEqual(raw_pm25, final_pm25)
        
        # Assert final output ranges
        self.assertTrue(0 <= final_aqi <= 500, f"AQI {final_aqi} is out of bounds [0, 500]")
        self.assertTrue(0 <= final_pm25 <= 999, f"PM2.5 {final_pm25} is out of bounds [0, 999]")
        self.assertTrue(0.0 <= hcho_prob <= 1.0, f"HCHO probability {hcho_prob} is out of bounds [0, 1]")

if __name__ == "__main__":
    unittest.main()

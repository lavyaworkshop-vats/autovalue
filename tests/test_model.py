"""AutoValue AI - Model Pipeline Unit Tests
"""
import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import ModelService
from backend.schemas import VehicleInput

class TestModelService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ModelService.get_instance()

    def test_model_loaded(self):
        self.assertIsNotNone(self.service.model)
        self.assertIsNotNone(self.service.preprocessor)
        self.assertGreater(len(self.service.feature_names), 50)

    def test_single_prediction_valid(self):
        sample = VehicleInput(
            year=2019,
            odometer=45000,
            manufacturer='toyota',
            condition='excellent',
            cylinders='4 cylinders',
            fuel='gas',
            title_status='clean',
            transmission='automatic',
            drive='4wd',
            type='SUV',
            state='ca'
        )
        val = self.service.predict_valuation(sample)
        self.assertIsInstance(val, float)
        self.assertGreater(val, 5000.0)
        self.assertLess(val, 100000.0)

    def test_depreciation_with_mileage(self):
        low_mil = VehicleInput(year=2018, odometer=20000, manufacturer='ford')
        high_mil = VehicleInput(year=2018, odometer=160000, manufacturer='ford')
        
        pred_low = self.service.predict_valuation(low_mil)
        pred_high = self.service.predict_valuation(high_mil)
        
        # Vehicle with 20k miles must be worth more than 160k miles
        self.assertGreater(pred_low, pred_high)

    def test_deal_evaluation(self):
        sample = VehicleInput(
            year=2018,
            odometer=50000,
            manufacturer='ford',
            asking_price=50000.0  # Clearly overpriced
        )
        res = self.service.evaluate_deal(sample)
        self.assertEqual(res.deal_classification, "Overpriced")
        self.assertEqual(res.deal_badge_color, "rose")

if __name__ == '__main__':
    unittest.main()

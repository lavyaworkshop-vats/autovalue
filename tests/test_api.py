"""AutoValue AI - FastAPI Route Integration Tests (Direct Invocation)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import health_check, get_metadata, get_model_info, predict_price, simulate_depreciation
from backend.schemas import VehicleInput, SimulationRequest

class TestAPIRoutes(unittest.TestCase):
    def test_health_check(self):
        res = health_check()
        self.assertEqual(res["status"], "healthy")
        self.assertEqual(res["service"], "AutoValue AI")

    def test_metadata_endpoint(self):
        res = get_metadata()
        self.assertIn("toyota", res.manufacturers)
        self.assertIn("SUV", res.body_types)
        self.assertIn("clean", res.title_statuses)

    def test_model_info_endpoint(self):
        res = get_model_info()
        self.assertIn("Ridge", res.architecture)
        self.assertGreater(res.metrics["R2_Score"], 0.8)

    def test_predict_endpoint(self):
        vehicle = VehicleInput(
            year=2020,
            odometer=35000,
            manufacturer="honda",
            condition="excellent",
            cylinders="4 cylinders",
            fuel="gas",
            title_status="clean",
            transmission="automatic",
            drive="fwd",
            type="sedan",
            state="ca",
            asking_price=22000.0
        )
        res = predict_price(vehicle)
        self.assertGreater(res.estimated_fair_value, 1000.0)
        self.assertIn(res.deal_classification, ["Fair Price", "Overpriced", "Undervalued (Great Deal!)"])
        self.assertGreater(len(res.key_drivers), 0)

    def test_simulate_endpoint(self):
        vehicle = VehicleInput(
            year=2018,
            odometer=50000,
            manufacturer="ford",
            condition="excellent"
        )
        req = SimulationRequest(base_vehicle=vehicle)
        res = simulate_depreciation(req)
        self.assertGreater(len(res.mileage_curve), 3)
        self.assertGreater(len(res.condition_curve), 3)
        # Verify first mileage point (10k mi) is higher than last point (200k mi)
        self.assertGreater(res.mileage_curve[0].predicted_price, res.mileage_curve[-1].predicted_price)

if __name__ == '__main__':
    unittest.main()

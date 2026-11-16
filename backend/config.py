"""AutoValue AI - Backend Configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "ridge_model.joblib")
MFR_COUNTS_PATH = os.path.join(MODEL_DIR, "mfr_counts.joblib")

# Fallback paths to project root
if not os.path.exists(PREPROCESSOR_PATH):
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
    MODEL_PATH = os.path.join(BASE_DIR, "ridge_model.joblib")
    MFR_COUNTS_PATH = os.path.join(BASE_DIR, "mfr_counts.joblib")

APP_TITLE = "AutoValue AI"
APP_DESCRIPTION = "Intelligent Used-Vehicle Pricing & Market Valuation API"
APP_VERSION = "2.0.0"

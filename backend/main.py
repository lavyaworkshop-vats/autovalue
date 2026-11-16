"""AutoValue AI - FastAPI Application Entrypoint
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .config import FRONTEND_DIR, APP_TITLE, APP_DESCRIPTION, APP_VERSION
from .schemas import (
    VehicleInput, PredictionResponse,
    SimulationRequest, SimulationResponse,
    MetadataResponse, ModelInfoResponse
)
from .inference import ModelService

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# Enable CORS for local development and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event: warm up the model
@app.on_event("startup")
def startup_event():
    print("[FastAPI] Initializing AutoValue AI Model Service...")
    ModelService.get_instance()
    print("[FastAPI] Ready to receive valuation requests.")

# API Endpoints
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION
    }

@app.get("/api/metadata", response_model=MetadataResponse)
def get_metadata():
    service = ModelService.get_instance()
    return service.get_metadata()

@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    service = ModelService.get_instance()
    return service.get_model_info()

@app.post("/api/predict", response_model=PredictionResponse)
def predict_price(vehicle: VehicleInput):
    try:
        service = ModelService.get_instance()
        return service.evaluate_deal(vehicle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/simulate", response_model=SimulationResponse)
def simulate_depreciation(req: SimulationRequest):
    try:
        service = ModelService.get_instance()
        return service.simulate_depreciation(req.base_vehicle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

# Mount static files and serve Single Page App
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_ui():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "AutoValue AI API is running. UI not found.",
        "docs_url": "/docs"
    }

"""AutoValue AI - Pydantic Request & Response Schemas
"""
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

class VehicleInput(BaseModel):
    year: int = Field(2018, ge=1980, le=2025, description="Manufacturing year of the vehicle")
    odometer: float = Field(50000.0, ge=100.0, le=350000.0, description="Odometer mileage reading in miles")
    manufacturer: str = Field("ford", description="Vehicle manufacturer (e.g. toyota, ford, bmw)")
    condition: str = Field("excellent", description="Vehicle cosmetic & mechanical condition")
    cylinders: str = Field("6 cylinders", description="Engine cylinder configuration")
    fuel: str = Field("gas", description="Engine fuel type (gas, diesel, hybrid, electric)")
    title_status: str = Field("clean", description="Vehicle title status (clean, rebuilt, salvage, etc.)")
    transmission: str = Field("automatic", description="Transmission type (automatic, manual)")
    drive: str = Field("4wd", description="Drivetrain type (4wd, fwd, rwd)")
    type: str = Field("SUV", description="Vehicle body type (sedan, SUV, pickup, coupe, etc.)")
    state: str = Field("ca", description="US State code (ca, tx, fl, ny, etc.)")
    lat: float = Field(37.77, description="Latitude coordinate")
    long: float = Field(-122.42, description="Longitude coordinate")
    asking_price: Optional[float] = Field(None, ge=100.0, description="Optional asking/listing price for deal evaluation")
    margin_pct: float = Field(0.15, ge=0.01, le=0.50, description="Tolerance percentage for fair deal classification")

class KeyDriver(BaseModel):
    factor: str
    impact: str
    coefficient: float
    description: str

class PredictionResponse(BaseModel):
    estimated_fair_value: float
    fair_value_range: Tuple[float, float]
    asking_price: Optional[float]
    deal_classification: str
    difference_from_fair_value: float
    deal_badge_color: str
    deal_summary: str
    key_drivers: List[KeyDriver]

class SimulationRequest(BaseModel):
    base_vehicle: VehicleInput

class MileageDataPoint(BaseModel):
    odometer: int
    predicted_price: float

class ConditionDataPoint(BaseModel):
    condition: str
    predicted_price: float

class SimulationResponse(BaseModel):
    base_price: float
    mileage_curve: List[MileageDataPoint]
    condition_curve: List[ConditionDataPoint]

class MetadataResponse(BaseModel):
    manufacturers: List[str]
    conditions: List[str]
    cylinders: List[str]
    fuel_types: List[str]
    title_statuses: List[str]
    transmissions: List[str]
    drives: List[str]
    body_types: List[str]
    states: List[str]

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    regularization_alpha: float
    metrics: Dict[str, Any]
    features_count: int
    numerical_features: List[str]
    categorical_features: List[str]

"""
Pydantic schema definitions for AgriSense AI API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionInputSchema(BaseModel):
    nitrogen: float = Field(..., ge=0, le=200, description="Nitrogen content in soil (kg/ha)", example=90)
    phosphorus: float = Field(..., ge=0, le=200, description="Phosphorus content in soil (kg/ha)", example=42)
    potassium: float = Field(..., ge=0, le=250, description="Potassium content in soil (kg/ha)", example=43)
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in °C", example=24.5)
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)", example=80.0)
    ph: float = Field(..., ge=0, le=14, description="Soil pH value", example=6.5)
    rainfall: float = Field(..., ge=0, le=5000, description="Annual/seasonal rainfall in mm", example=220.0)
    soil_type: str = Field(default="Loamy", description="Soil texture type", example="Loamy")
    season: str = Field(default="Kharif", description="Cropping season", example="Kharif")
    region: str = Field(default="South India", description="Geographic region", example="South India")
    soil_moisture: Optional[float] = Field(default=50.0, ge=0, le=100, description="Soil moisture level (%)", example=70.0)
    irrigation: Optional[str] = Field(default="Available", description="Irrigation availability", example="Available")
    sunlight: Optional[float] = Field(default=8.0, ge=0, le=24, description="Daily sunlight hours", example=8.0)

class BatchPredictionInputSchema(BaseModel):
    inputs: List[PredictionInputSchema]

class TopRecommendationSchema(BaseModel):
    crop: str
    confidence_percentage: float
    probability: float

class ExplanationSchema(BaseModel):
    recommended_crop: str
    summary: str
    contributing_factors: List[str]
    warnings: List[str]
    disclaimer: str

class PredictionResponseSchema(BaseModel):
    success: bool
    recommended_crop: Optional[str] = None
    confidence: Optional[float] = None
    top_3_recommendations: Optional[List[TopRecommendationSchema]] = None
    explanation: Optional[ExplanationSchema] = None
    input_parameters: Optional[Dict[str, Any]] = None
    validation_errors: Optional[List[str]] = None

class FeedbackInputSchema(BaseModel):
    prediction_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5, description="Satisfaction rating from 1 to 5")
    comments: Optional[str] = None

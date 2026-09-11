"""
Crop prediction engine with strict agronomic input validation and probability scoring.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Tuple, List

from src.utils import (
    setup_logger, MODEL_PATH, PIPELINE_PATH, LABEL_ENCODER_PATH,
    AGRONOMIC_BOUNDS, VALID_SOIL_TYPES, VALID_SEASONS, VALID_REGIONS, VALID_IRRIGATION
)
from src.explain import generate_crop_explanation

logger = setup_logger("predict")

# Cache models in memory
_MODEL = None
_PIPELINE = None
_LABEL_ENCODER = None

def load_prediction_artifacts():
    global _MODEL, _PIPELINE, _LABEL_ENCODER
    if _MODEL is None:
        logger.info("Loading ML model artifacts into memory...")
        _MODEL = joblib.load(MODEL_PATH)
        _PIPELINE = joblib.load(PIPELINE_PATH)
        _LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)

def validate_farmer_inputs(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates input parameters against physical agronomic bounds and valid choices.
    
    Returns:
        Tuple of (is_valid: bool, error_messages: List[str])
    """
    errors = []
    
    # Numerical validation
    for key, bounds in AGRONOMIC_BOUNDS.items():
        if key in input_data and input_data[key] is not None:
            val = float(input_data[key])
            if val < bounds["min"] or val > bounds["max"]:
                errors.append(
                    f"Value for '{key}' ({val} {bounds['unit']}) is outside expected range ({bounds['min']} to {bounds['max']} {bounds['unit']}). Please verify."
                )
                
    # Categorical validation
    soil = input_data.get("soil_type")
    if soil and soil not in VALID_SOIL_TYPES:
        errors.append(f"Invalid soil_type '{soil}'. Supported: {', '.join(VALID_SOIL_TYPES)}")
        
    season = input_data.get("season")
    if season and season not in VALID_SEASONS:
        errors.append(f"Invalid season '{season}'. Supported: {', '.join(VALID_SEASONS)}")
        
    region = input_data.get("region")
    if region and region not in VALID_REGIONS:
        errors.append(f"Invalid region '{region}'. Supported: {', '.join(VALID_REGIONS)}")
        
    irrigation = input_data.get("irrigation")
    if irrigation and irrigation not in VALID_IRRIGATION:
        errors.append(f"Invalid irrigation '{irrigation}'. Supported: {', '.join(VALID_IRRIGATION)}")
        
    return len(errors) == 0, errors

def predict_crop(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction pipeline. Validates input, transforms features, predicts top 3 crops with probabilities,
    and attaches agronomic explanation.
    
    Args:
        input_data: Dictionary of farmer inputs
        
    Returns:
        Structured prediction result dictionary
    """
    load_prediction_artifacts()
    
    # 1. Validate Input
    is_valid, validation_errors = validate_farmer_inputs(input_data)
    if not is_valid:
        return {
            "success": False,
            "error_type": "VALIDATION_ERROR",
            "validation_errors": validation_errors,
            "recommended_crop": None,
            "confidence": 0.0,
            "top_3_recommendations": [],
            "explanation": None
        }
        
    # 2. Format into Single-Row DataFrame
    row_dict = {
        "N": float(input_data.get("nitrogen", 50)),
        "P": float(input_data.get("phosphorus", 50)),
        "K": float(input_data.get("potassium", 50)),
        "temperature": float(input_data.get("temperature", 25.0)),
        "humidity": float(input_data.get("humidity", 60.0)),
        "ph": float(input_data.get("ph", 6.5)),
        "rainfall": float(input_data.get("rainfall", 500.0)),
        "soil_type": str(input_data.get("soil_type", "Loamy")),
        "season": str(input_data.get("season", "Kharif")),
        "region": str(input_data.get("region", "North India")),
        "soil_moisture": float(input_data.get("soil_moisture", 50.0)),
        "irrigation": str(input_data.get("irrigation", "Available")),
        "sunlight": float(input_data.get("sunlight", 8.0))
    }
    
    df_input = pd.DataFrame([row_dict])
    
    # 3. Transform Input using saved Pipeline
    X_processed = _PIPELINE.transform(df_input)
    
    # 4. Model Prediction & Probabilities
    if hasattr(_MODEL, "predict_proba"):
        probs = _MODEL.predict_proba(X_processed)[0]
    else:
        # Fallback for models without predict_proba
        pred_idx = _MODEL.predict(X_processed)[0]
        probs = np.zeros(len(_LABEL_ENCODER.classes_))
        probs[pred_idx] = 1.0
        
    # Get Top 3 Predictions
    top_indices = np.argsort(probs)[::-1][:3]
    top_crops = []
    
    for idx in top_indices:
        crop_name = _LABEL_ENCODER.inverse_transform([idx])[0]
        prob_pct = round(float(probs[idx]) * 100.0, 2)
        top_crops.append({
            "crop": crop_name,
            "confidence_percentage": prob_pct,
            "probability": round(float(probs[idx]), 4)
        })
        
    recommended_crop = top_crops[0]["crop"]
    best_confidence = top_crops[0]["confidence_percentage"]
    
    # 5. Generate Agronomic Rationale
    explanation = generate_crop_explanation(recommended_crop, input_data, top_crops)
    
    return {
        "success": True,
        "recommended_crop": recommended_crop,
        "confidence": best_confidence,
        "top_3_recommendations": top_crops,
        "explanation": explanation,
        "input_parameters": input_data
    }

if __name__ == "__main__":
    test_input = {
        "nitrogen": 90,
        "phosphorus": 42,
        "potassium": 43,
        "temperature": 24.5,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 2200,
        "soil_type": "Loamy",
        "season": "Kharif",
        "region": "South India",
        "soil_moisture": 70,
        "irrigation": "Available",
        "sunlight": 8
    }
    res = predict_crop(test_input)
    print("Sample Prediction Result:")
    print(res)

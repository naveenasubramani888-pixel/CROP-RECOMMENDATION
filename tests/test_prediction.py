"""
Integration tests for prediction pipeline and explanation engine.
"""

import pytest
from src.predict import predict_crop

def test_prediction_output_structure():
    sample_input = {
        "nitrogen": 90, "phosphorus": 42, "potassium": 43,
        "temperature": 24.5, "humidity": 80, "ph": 6.5, "rainfall": 2200,
        "soil_type": "Loamy", "season": "Kharif", "region": "South India",
        "soil_moisture": 70, "irrigation": "Available", "sunlight": 8
    }
    res = predict_crop(sample_input)
    assert res["success"] is True
    assert "recommended_crop" in res
    assert res["confidence"] > 0
    assert len(res["top_3_recommendations"]) == 3
    assert "explanation" in res

def test_invalid_input_prediction_rejection():
    invalid_input = {
        "nitrogen": -50,  # Negative N
        "temperature": 25.0
    }
    res = predict_crop(invalid_input)
    assert res["success"] is False
    assert res["error_type"] == "VALIDATION_ERROR"

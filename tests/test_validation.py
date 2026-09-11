"""
Unit tests for input validation engine.
"""

import pytest
from src.predict import validate_farmer_inputs

def test_valid_input_passes():
    valid_data = {
        "nitrogen": 90, "phosphorus": 42, "potassium": 43,
        "temperature": 24.5, "humidity": 80, "ph": 6.5, "rainfall": 220,
        "soil_type": "Loamy", "season": "Kharif", "region": "South India",
        "soil_moisture": 70, "irrigation": "Available", "sunlight": 8
    }
    is_valid, errors = validate_farmer_inputs(valid_data)
    assert is_valid is True
    assert len(errors) == 0

def test_invalid_temperature_fails():
    invalid_data = {
        "temperature": 120.0,  # Extreme temp out of bounds (-10 to 60)
        "nitrogen": 50, "phosphorus": 50, "potassium": 50,
        "humidity": 50, "ph": 6.5, "rainfall": 500
    }
    is_valid, errors = validate_farmer_inputs(invalid_data)
    assert is_valid is False
    assert any("temperature" in err for err in errors)

def test_invalid_soil_type_fails():
    invalid_data = {
        "soil_type": "Martian Soil",
        "nitrogen": 50, "phosphorus": 50, "potassium": 50,
        "temperature": 25, "humidity": 50, "ph": 6.5, "rainfall": 500
    }
    is_valid, errors = validate_farmer_inputs(invalid_data)
    assert is_valid is False
    assert any("soil_type" in err for err in errors)

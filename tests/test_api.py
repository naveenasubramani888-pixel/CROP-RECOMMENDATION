"""
Integration tests for FastAPI REST API endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "AgriSense AI" in data["app_name"]

def test_crops_endpoint():
    response = client.get("/crops")
    assert response.status_code == 200
    data = response.json()
    assert data["total_crops"] == 20
    assert "Rice" in data["crops"]

def test_predict_endpoint_success():
    payload = {
        "nitrogen": 90, "phosphorus": 42, "potassium": 43,
        "temperature": 24.5, "humidity": 80, "ph": 6.5, "rainfall": 2200,
        "soil_type": "Loamy", "season": "Kharif", "region": "South India",
        "soil_moisture": 70, "irrigation": "Available", "sunlight": 8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["recommended_crop"] is not None

def test_predict_endpoint_validation_error():
    payload = {
        "nitrogen": 90,
        "phosphorus": 42,
        "potassium": 9999  # Out of bounds
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

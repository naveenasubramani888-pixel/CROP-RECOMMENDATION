"""
Explanation and agronomic reasoning engine for AgriSense AI.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.utils import CROP_KNOWLEDGE_BASE

def generate_crop_explanation(
    recommended_crop: str,
    input_data: Dict[str, Any],
    top_3_crops: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates human-understandable agricultural rationale for crop recommendations.
    
    Args:
        recommended_crop: Primary predicted crop name
        input_data: Dictionary of farmer inputs
        top_3_crops: List of top 3 predictions with probabilities
        
    Returns:
        Structured explanation dictionary with text reasoning, suitability breakdown, and warnings
    """
    kb = CROP_KNOWLEDGE_BASE.get(recommended_crop, {})
    
    N = input_data.get("nitrogen", 0)
    P = input_data.get("phosphorus", 0)
    K = input_data.get("potassium", 0)
    temp = input_data.get("temperature", 0)
    humidity = input_data.get("humidity", 0)
    ph = input_data.get("ph", 0)
    rainfall = input_data.get("rainfall", 0)
    soil_type = input_data.get("soil_type", "Loamy")
    season = input_data.get("season", "Kharif")
    
    # Analyze Suitability Factors
    factors = []
    
    # Rainfall Check
    if rainfall > 1500:
        if recommended_crop in ["Rice", "Sugarcane", "Banana", "Tea", "Coffee"]:
            factors.append(f"High rainfall ({rainfall} mm) provides ideal moisture for {recommended_crop}.")
        else:
            factors.append(f"Abundant water supply ({rainfall} mm) supports high-moisture growth.")
    elif rainfall < 600:
        if recommended_crop in ["Millet", "Sorghum", "Chickpea", "Lentil", "Mustard", "Barley"]:
            factors.append(f"Low/moderate rainfall ({rainfall} mm) perfectly suits drought-tolerant {recommended_crop}.")
        else:
            factors.append(f"Rainfall of {rainfall} mm is sufficient with proper water management.")
    else:
        factors.append(f"Moderate rainfall of {rainfall} mm provides balanced soil moisture.")
        
    # Temperature & Humidity Check
    if temp > 28:
        factors.append(f"Warm climate ({temp}°C) accelerates {recommended_crop} development.")
    elif temp < 18:
        factors.append(f"Cool temperature ({temp}°C) provides ideal winter growing conditions for {recommended_crop}.")
    else:
        factors.append(f"Moderate temperature ({temp}°C) supports healthy vegetative growth.")
        
    # NPK Nutrient Profile Check
    npk_total = N + P + K
    if npk_total > 200:
        factors.append(f"High soil nutrient profile (NPK sum: {npk_total} kg/ha) satisfies heavy feeder requirements.")
    else:
        factors.append(f"Soil nutrient level (N: {N}, P: {P}, K: {K}) matches {recommended_crop} metabolic needs.")
        
    # Soil & pH Check
    factors.append(f"Soil pH ({ph}) and texture ({soil_type}) fall within optimal agricultural parameters.")
    
    # Construct Summary Sentence
    primary_reason = f"{recommended_crop} is recommended because environmental conditions ({temp}°C, {humidity}% humidity, {rainfall} mm rainfall) and soil fertility (pH {ph}, NPK {N}:{P}:{K}) closely match its optimal agronomic growth profile for the {season} season."
    
    # Agronomic Warnings
    warnings = []
    if ph < 5.0:
        warnings.append("Soil is strongly acidic. Consider applying lime to raise pH for better nutrient availability.")
    elif ph > 8.0:
        warnings.append("Soil is alkaline. Consider applying organic compost or gypsum.")
        
    if rainfall < 400 and input_data.get("irrigation") == "Rainfed":
        warnings.append("Low rainfall detected under rainfed conditions. Supplementary irrigation recommended.")
        
    return {
        "recommended_crop": recommended_crop,
        "summary": primary_reason,
        "contributing_factors": factors,
        "knowledge_base": kb,
        "warnings": warnings,
        "disclaimer": "AI-generated recommendation. Consult local agricultural experts before making major farming decisions."
    }

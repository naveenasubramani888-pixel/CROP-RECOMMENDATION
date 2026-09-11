"""
Feature engineering transformers and domain calculations for AgriSense AI.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AgriculturalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for generating domain-specific agricultural features.
    """
    
    def __init__(self, include_ratios: bool = True):
        self.include_ratios = include_ratios
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_df = X.copy()
        
        # 1. Total NPK
        N = X_df["N"].values
        P = X_df["P"].values
        K = X_df["K"].values
        
        X_df["NPK_total"] = N + P + K
        
        if self.include_ratios:
            # Prevent division by zero
            X_df["NPK_N_P_ratio"] = N / (P + 1e-5)
            X_df["NPK_N_K_ratio"] = N / (K + 1e-5)
            X_df["NPK_P_K_ratio"] = P / (K + 1e-5)
            
        # 2. Water Availability Index
        # Rain + soil_moisture + humidity component
        rainfall = X_df["rainfall"].values
        humidity = X_df["humidity"].values
        moisture = X_df["soil_moisture"].values
        
        # Normalize component scores roughly
        rain_score = np.clip(rainfall / 2500.0, 0, 1.5)
        hum_score = np.clip(humidity / 100.0, 0, 1.0)
        moist_score = np.clip(moisture / 100.0, 0, 1.0)
        
        irrigation_bonus = np.where(X_df["irrigation"] == "Available", 0.2, 
                           np.where(X_df["irrigation"] == "Drip Irrigation", 0.25,
                           np.where(X_df["irrigation"] == "Canal Irrigation", 0.2, 0.0)))
        
        X_df["water_availability_index"] = round_series(rain_score * 0.4 + hum_score * 0.3 + moist_score * 0.3 + irrigation_bonus)
        
        # 3. Soil Fertility Score
        npk_total = X_df["NPK_total"].values
        ph = X_df["ph"].values
        
        npk_score = np.clip(npk_total / 300.0, 0, 1.2)
        ph_penalty = np.abs(ph - 6.5) / 3.5  # Ideal pH near 6.5
        ph_score = np.clip(1.0 - ph_penalty, 0.1, 1.0)
        
        X_df["soil_fertility_score"] = round_series(npk_score * 0.7 + ph_score * 0.3)
        
        # 4. Climate Suitability Index
        temp = X_df["temperature"].values
        sunlight = X_df["sunlight"].values
        
        temp_score = np.where((temp >= 20) & (temp <= 32), 1.0, np.where((temp >= 12) & (temp <= 38), 0.7, 0.4))
        sun_score = np.clip(sunlight / 12.0, 0.1, 1.0)
        
        X_df["climate_suitability_index"] = round_series(temp_score * 0.6 + sun_score * 0.4)
        
        return X_df

def round_series(arr):
    return np.round(arr, 4)

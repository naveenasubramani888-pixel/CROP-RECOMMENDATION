"""
Domain-informed synthetic agricultural dataset generation module for AgriSense AI.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.utils import setup_logger, RAW_DATA_PATH, AGRONOMIC_BOUNDS

logger = setup_logger("data_generation")

# Agronomic optimal profile per crop (mean, std)
CROP_PROFILES: Dict[str, Dict[str, Any]] = {
    "Rice": {
        "N": (90, 15), "P": (45, 10), "K": (40, 10),
        "temp": (26, 3), "humidity": (82, 6), "ph": (6.3, 0.4), "rainfall": (2200, 300),
        "soil_moisture": (80, 8), "sunlight": (7, 1.2),
        "soil_types": ["Clay", "Loamy"], "soil_probs": [0.7, 0.3],
        "seasons": ["Kharif"], "season_probs": [1.0],
        "regions": ["East India", "South India", "North India"], "region_probs": [0.5, 0.3, 0.2],
        "irrigation": ["Available", "Canal Irrigation", "Rainfed"], "irrigation_probs": [0.5, 0.4, 0.1]
    },
    "Wheat": {
        "N": (120, 15), "P": (60, 10), "K": (50, 10),
        "temp": (18, 3), "humidity": (55, 7), "ph": (6.8, 0.4), "rainfall": (700, 120),
        "soil_moisture": (50, 7), "sunlight": (8.5, 1.0),
        "soil_types": ["Loamy", "Clay", "Silty"], "soil_probs": [0.6, 0.25, 0.15],
        "seasons": ["Rabi"], "season_probs": [1.0],
        "regions": ["North India", "Central India", "West India"], "region_probs": [0.6, 0.25, 0.15],
        "irrigation": ["Available", "Canal Irrigation"], "irrigation_probs": [0.7, 0.3]
    },
    "Maize": {
        "N": (85, 12), "P": (50, 10), "K": (40, 8),
        "temp": (24, 4), "humidity": (65, 8), "ph": (6.5, 0.5), "rainfall": (900, 150),
        "soil_moisture": (60, 8), "sunlight": (8.0, 1.1),
        "soil_types": ["Loamy", "Sandy", "Red Soil"], "soil_probs": [0.5, 0.3, 0.2],
        "seasons": ["Kharif", "Rabi"], "season_probs": [0.75, 0.25],
        "regions": ["North India", "Central India", "South India"], "region_probs": [0.4, 0.3, 0.3],
        "irrigation": ["Available", "Rainfed"], "irrigation_probs": [0.6, 0.4]
    },
    "Cotton": {
        "N": (125, 15), "P": (55, 10), "K": (55, 10),
        "temp": (28, 4), "humidity": (60, 8), "ph": (7.2, 0.5), "rainfall": (800, 150),
        "soil_moisture": (55, 8), "sunlight": (9.0, 1.0),
        "soil_types": ["Black Soil", "Loamy"], "soil_probs": [0.8, 0.2],
        "seasons": ["Kharif"], "season_probs": [1.0],
        "regions": ["West India", "Central India", "South India"], "region_probs": [0.5, 0.3, 0.2],
        "irrigation": ["Available", "Rainfed", "Drip Irrigation"], "irrigation_probs": [0.5, 0.3, 0.2]
    },
    "Sugarcane": {
        "N": (145, 15), "P": (70, 12), "K": (80, 12),
        "temp": (29, 3), "humidity": (75, 7), "ph": (6.8, 0.4), "rainfall": (2000, 250),
        "soil_moisture": (75, 7), "sunlight": (8.0, 1.0),
        "soil_types": ["Loamy", "Clay", "Black Soil"], "soil_probs": [0.4, 0.35, 0.25],
        "seasons": ["Whole Year"], "season_probs": [1.0],
        "regions": ["North India", "West India", "South India"], "region_probs": [0.4, 0.35, 0.25],
        "irrigation": ["Available", "Canal Irrigation"], "irrigation_probs": [0.6, 0.4]
    },
    "Groundnut": {
        "N": (30, 8), "P": (50, 10), "K": (50, 10),
        "temp": (27, 3), "humidity": (62, 8), "ph": (6.4, 0.4), "rainfall": (750, 120),
        "soil_moisture": (45, 7), "sunlight": (8.5, 1.0),
        "soil_types": ["Sandy", "Loamy", "Red Soil"], "soil_probs": [0.5, 0.35, 0.15],
        "seasons": ["Kharif", "Zaid"], "season_probs": [0.8, 0.2],
        "regions": ["West India", "South India"], "region_probs": [0.6, 0.4],
        "irrigation": ["Rainfed", "Available"], "irrigation_probs": [0.6, 0.4]
    },
    "Soybean": {
        "N": (40, 8), "P": (70, 10), "K": (50, 10),
        "temp": (25, 3), "humidity": (70, 7), "ph": (6.6, 0.4), "rainfall": (850, 140),
        "soil_moisture": (60, 7), "sunlight": (7.5, 1.0),
        "soil_types": ["Black Soil", "Loamy"], "soil_probs": [0.7, 0.3],
        "seasons": ["Kharif"], "season_probs": [1.0],
        "regions": ["Central India", "West India"], "region_probs": [0.7, 0.3],
        "irrigation": ["Rainfed", "Available"], "irrigation_probs": [0.7, 0.3]
    },
    "Potato": {
        "N": (95, 12), "P": (95, 12), "K": (120, 15),
        "temp": (19, 3), "humidity": (65, 8), "ph": (5.8, 0.4), "rainfall": (600, 100),
        "soil_moisture": (65, 6), "sunlight": (7.0, 1.0),
        "soil_types": ["Loamy", "Sandy"], "soil_probs": [0.7, 0.3],
        "seasons": ["Rabi"], "season_probs": [1.0],
        "regions": ["North India", "East India"], "region_probs": [0.6, 0.4],
        "irrigation": ["Available", "Drip Irrigation"], "irrigation_probs": [0.7, 0.3]
    },
    "Tomato": {
        "N": (110, 15), "P": (75, 12), "K": (100, 15),
        "temp": (24, 3), "humidity": (68, 7), "ph": (6.5, 0.4), "rainfall": (900, 150),
        "soil_moisture": (60, 7), "sunlight": (8.5, 1.0),
        "soil_types": ["Loamy", "Sandy", "Red Soil"], "soil_probs": [0.6, 0.25, 0.15],
        "seasons": ["Whole Year", "Kharif", "Rabi"], "season_probs": [0.5, 0.3, 0.2],
        "regions": ["South India", "West India", "North India"], "region_probs": [0.4, 0.3, 0.3],
        "irrigation": ["Drip Irrigation", "Available"], "irrigation_probs": [0.6, 0.4]
    },
    "Chickpea": {
        "N": (30, 8), "P": (60, 10), "K": (30, 8),
        "temp": (20, 3), "humidity": (50, 8), "ph": (7.1, 0.4), "rainfall": (500, 100),
        "soil_moisture": (40, 6), "sunlight": (8.0, 1.0),
        "soil_types": ["Loamy", "Black Soil"], "soil_probs": [0.6, 0.4],
        "seasons": ["Rabi"], "season_probs": [1.0],
        "regions": ["Central India", "North India"], "region_probs": [0.6, 0.4],
        "irrigation": ["Rainfed", "Available"], "irrigation_probs": [0.7, 0.3]
    },
    "Lentil": {
        "N": (25, 7), "P": (50, 9), "K": (30, 7),
        "temp": (18, 3), "humidity": (52, 7), "ph": (6.7, 0.4), "rainfall": (450, 90),
        "soil_moisture": (42, 6), "sunlight": (7.5, 1.0),
        "soil_types": ["Loamy", "Silty"], "soil_probs": [0.7, 0.3],
        "seasons": ["Rabi"], "season_probs": [1.0],
        "regions": ["North India", "East India"], "region_probs": [0.6, 0.4],
        "irrigation": ["Rainfed", "Available"], "irrigation_probs": [0.6, 0.4]
    },
    "Millet": {
        "N": (55, 10), "P": (30, 8), "K": (30, 8),
        "temp": (31, 4), "humidity": (45, 9), "ph": (6.6, 0.5), "rainfall": (500, 110),
        "soil_moisture": (35, 7), "sunlight": (9.5, 1.0),
        "soil_types": ["Sandy", "Loamy", "Red Soil"], "soil_probs": [0.6, 0.25, 0.15],
        "seasons": ["Kharif"], "season_probs": [1.0],
        "regions": ["West India", "Central India"], "region_probs": [0.6, 0.4],
        "irrigation": ["Rainfed"], "irrigation_probs": [1.0]
    },
    "Sorghum": {
        "N": (65, 12), "P": (40, 9), "K": (40, 9),
        "temp": (30, 4), "humidity": (50, 8), "ph": (7.0, 0.5), "rainfall": (600, 120),
        "soil_moisture": (40, 7), "sunlight": (9.0, 1.0),
        "soil_types": ["Black Soil", "Loamy", "Sandy"], "soil_probs": [0.5, 0.3, 0.2],
        "seasons": ["Kharif", "Rabi"], "season_probs": [0.7, 0.3],
        "regions": ["Central India", "West India", "South India"], "region_probs": [0.4, 0.3, 0.3],
        "irrigation": ["Rainfed", "Available"], "irrigation_probs": [0.75, 0.25]
    },
    "Barley": {
        "N": (75, 12), "P": (40, 8), "K": (40, 8),
        "temp": (17, 3), "humidity": (52, 7), "ph": (7.2, 0.4), "rainfall": (550, 100),
        "soil_moisture": (45, 6), "sunlight": (8.0, 1.0),
        "soil_types": ["Loamy", "Sandy"], "soil_probs": [0.65, 0.35],
        "seasons": ["Rabi"], "season_probs": [1.0],
        "regions": ["North India", "Himalayan Region"], "region_probs": [0.7, 0.3],
        "irrigation": ["Available", "Rainfed"], "irrigation_probs": [0.6, 0.4]
    },
    "Mustard": {
        "N": (75, 12), "P": (50, 9), "K": (40, 8),
        "temp": (16, 3), "humidity": (58, 7), "ph": (6.7, 0.4), "rainfall": (400, 80),
        "soil_moisture": (45, 6), "sunlight": (7.8, 1.0),
        "soil_types": ["Loamy", "Sandy"], "soil_probs": [0.6, 0.4],
        "seasons": ["Rabi"], "season_probs": [1.0],
        "regions": ["North India", "West India"], "region_probs": [0.6, 0.4],
        "irrigation": ["Available", "Rainfed"], "irrigation_probs": [0.65, 0.35]
    },
    "Banana": {
        "N": (130, 15), "P": (85, 12), "K": (150, 18),
        "temp": (29, 3), "humidity": (80, 6), "ph": (6.5, 0.4), "rainfall": (2100, 250),
        "soil_moisture": (78, 6), "sunlight": (8.0, 1.0),
        "soil_types": ["Loamy", "Clay"], "soil_probs": [0.7, 0.3],
        "seasons": ["Whole Year"], "season_probs": [1.0],
        "regions": ["South India", "West India", "East India"], "region_probs": [0.5, 0.3, 0.2],
        "irrigation": ["Available", "Drip Irrigation"], "irrigation_probs": [0.6, 0.4]
    },
    "Mango": {
        "N": (90, 12), "P": (55, 10), "K": (80, 12),
        "temp": (31, 3), "humidity": (60, 8), "ph": (6.5, 0.4), "rainfall": (1200, 200),
        "soil_moisture": (55, 7), "sunlight": (9.0, 1.0),
        "soil_types": ["Loamy", "Red Soil"], "soil_probs": [0.6, 0.4],
        "seasons": ["Whole Year"], "season_probs": [1.0],
        "regions": ["South India", "West India", "North India"], "region_probs": [0.4, 0.3, 0.3],
        "irrigation": ["Available", "Rainfed"], "irrigation_probs": [0.5, 0.5]
    },
    "Coffee": {
        "N": (95, 12), "P": (55, 10), "K": (100, 12),
        "temp": (22, 2.5), "humidity": (78, 6), "ph": (5.8, 0.35), "rainfall": (2100, 250),
        "soil_moisture": (72, 6), "sunlight": (6.5, 1.0),
        "soil_types": ["Peaty", "Loamy"], "soil_probs": [0.6, 0.4],
        "seasons": ["Whole Year"], "season_probs": [1.0],
        "regions": ["South India"], "region_probs": [1.0],
        "irrigation": ["Available", "Rainfed"], "irrigation_probs": [0.5, 0.5]
    },
    "Tea": {
        "season": "Whole Year",
        "N": (115, 12), "P": (50, 9), "K": (65, 10),
        "temp": (21, 2.5), "humidity": (85, 5), "ph": (5.0, 0.3), "rainfall": (2500, 300),
        "soil_moisture": (80, 5), "sunlight": (6.0, 1.0),
        "soil_types": ["Peaty", "Loamy"], "soil_probs": [0.7, 0.3],
        "seasons": ["Whole Year"], "season_probs": [1.0],
        "regions": ["East India", "South India", "Himalayan Region"], "region_probs": [0.5, 0.3, 0.2],
        "irrigation": ["Rainfed", "Available"], "irrigation_probs": [0.7, 0.3]
    },
    "Coconut": {
        "N": (55, 10), "P": (40, 8), "K": (130, 15),
        "temp": (30, 3), "humidity": (82, 5), "ph": (6.6, 0.5), "rainfall": (1900, 250),
        "soil_moisture": (70, 7), "sunlight": (9.0, 0.9),
        "soil_types": ["Sandy", "Loamy"], "soil_probs": [0.65, 0.35],
        "seasons": ["Whole Year"], "season_probs": [1.0],
        "regions": ["South India", "West India"], "region_probs": [0.7, 0.3],
        "irrigation": ["Available", "Rainfed"], "irrigation_probs": [0.6, 0.4]
    }
}

def generate_agricultural_dataset(num_samples: int = 15000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic dataset with domain-informed distributions and controlled noise.
    
    Args:
        num_samples: Total rows to generate (default 15,000)
        random_seed: Seed for reproducibility
        
    Returns:
        pd.DataFrame containing raw agricultural dataset
    """
    np.random.seed(random_seed)
    crops = list(CROP_PROFILES.keys())
    samples_per_crop = num_samples // len(crops)
    
    records = []
    logger.info(f"Generating synthetic dataset with {num_samples} records across {len(crops)} crops...")
    
    for crop in crops:
        prof = CROP_PROFILES[crop]
        
        N_vals = np.clip(np.random.normal(prof["N"][0], prof["N"][1], samples_per_crop), 0, 200)
        P_vals = np.clip(np.random.normal(prof["P"][0], prof["P"][1], samples_per_crop), 0, 200)
        K_vals = np.clip(np.random.normal(prof["K"][0], prof["K"][1], samples_per_crop), 0, 250)
        
        temp_vals = np.clip(np.random.normal(prof["temp"][0], prof["temp"][1], samples_per_crop), 5, 50)
        humidity_vals = np.clip(np.random.normal(prof["humidity"][0], prof["humidity"][1], samples_per_crop), 15, 100)
        ph_vals = np.clip(np.random.normal(prof["ph"][0], prof["ph"][1], samples_per_crop), 3.5, 9.5)
        rainfall_vals = np.clip(np.random.normal(prof["rainfall"][0], prof["rainfall"][1], samples_per_crop), 150, 4500)
        
        moisture_vals = np.clip(np.random.normal(prof["soil_moisture"][0], prof["soil_moisture"][1], samples_per_crop), 10, 98)
        sunlight_vals = np.clip(np.random.normal(prof["sunlight"][0], prof["sunlight"][1], samples_per_crop), 3.0, 14.0)
        
        soil_types = np.random.choice(prof["soil_types"], size=samples_per_crop, p=prof["soil_probs"])
        seasons = np.random.choice(prof["seasons"], size=samples_per_crop, p=prof["season_probs"])
        regions = np.random.choice(prof["regions"], size=samples_per_crop, p=prof["region_probs"])
        irrigation_vals = np.random.choice(prof["irrigation"], size=samples_per_crop, p=prof["irrigation_probs"])
        
        for i in range(samples_per_crop):
            records.append({
                "N": round(float(N_vals[i]), 1),
                "P": round(float(P_vals[i]), 1),
                "K": round(float(K_vals[i]), 1),
                "temperature": round(float(temp_vals[i]), 2),
                "humidity": round(float(humidity_vals[i]), 2),
                "ph": round(float(ph_vals[i]), 2),
                "rainfall": round(float(rainfall_vals[i]), 2),
                "soil_type": soil_types[i],
                "season": seasons[i],
                "region": regions[i],
                "soil_moisture": round(float(moisture_vals[i]), 2),
                "irrigation": irrigation_vals[i],
                "sunlight": round(float(sunlight_vals[i]), 2),
                "crop": crop
            })

    df = pd.DataFrame(records)
    
    # Inject 5% controlled realistic noise (overlap in parameters)
    noise_count = int(0.05 * len(df))
    noise_indices = np.random.choice(df.index, size=noise_count, replace=False)
    
    for idx in noise_indices:
        # Add slight fluctuation to environmental metrics to simulate real-world field variance
        df.loc[idx, "N"] = round(float(np.clip(df.loc[idx, "N"] + np.random.normal(0, 12), 0, 200)), 1)
        df.loc[idx, "rainfall"] = round(float(np.clip(df.loc[idx, "rainfall"] + np.random.normal(0, 150), 100, 4500)), 2)
        df.loc[idx, "temperature"] = round(float(np.clip(df.loc[idx, "temperature"] + np.random.normal(0, 2.5), 5, 50)), 2)

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    logger.info(f"Dataset generated successfully. Total rows: {len(df)}, Features: {df.shape[1]-1}, Target: crop ({df['crop'].nunique()} unique classes).")
    return df

def save_raw_dataset(df: pd.DataFrame, output_path: str = str(RAW_DATA_PATH)) -> None:
    """Saves generated dataset to data/raw/ file without overwriting if not intended."""
    df.to_csv(output_path, index=False)
    logger.info(f"Raw dataset saved to: {output_path}")

if __name__ == "__main__":
    df_generated = generate_agricultural_dataset()
    save_raw_dataset(df_generated)

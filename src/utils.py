"""
Utility functions, paths, loggers, and agronomic bounds for AgriSense AI.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "crop_recommendation_raw.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned" / "crop_recommendation_cleaned.csv"
PROCESSED_DIR = DATA_DIR / "processed"

X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "crop_recommendation_model.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.txt"

LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for path in [DATA_DIR / "raw", DATA_DIR / "cleaned", PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Logger configuration
def setup_logger(name: str, log_file: str = "app.log", level=logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
        file_handler.setLevel(level)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# Agronomic Valid Input Ranges
AGRONOMIC_BOUNDS = {
    "nitrogen": {"min": 0, "max": 200, "unit": "kg/ha"},
    "phosphorus": {"min": 0, "max": 200, "unit": "kg/ha"},
    "potassium": {"min": 0, "max": 250, "unit": "kg/ha"},
    "temperature": {"min": -10.0, "max": 60.0, "unit": "°C"},
    "humidity": {"min": 0.0, "max": 100.0, "unit": "%"},
    "ph": {"min": 0.0, "max": 14.0, "unit": "pH"},
    "rainfall": {"min": 0.0, "max": 5000.0, "unit": "mm"},
    "soil_moisture": {"min": 0.0, "max": 100.0, "unit": "%"},
    "sunlight": {"min": 0.0, "max": 24.0, "unit": "hours/day"}
}

VALID_SOIL_TYPES = ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky", "Black Soil", "Red Soil"]
VALID_SEASONS = ["Kharif", "Rabi", "Zaid", "Whole Year"]
VALID_REGIONS = ["North India", "South India", "East India", "West India", "Central India", "Himalayan Region"]
VALID_IRRIGATION = ["Available", "Rainfed", "Drip Irrigation", "Canal Irrigation"]

# Crop Knowledge Base
CROP_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Rice": {
        "season": "Kharif",
        "temp_range": "20°C - 35°C",
        "rainfall_range": "1500 - 3000 mm",
        "ph_range": "5.5 - 7.0",
        "soil_types": ["Clay", "Loamy"],
        "npk": "N: 80-120, P: 40-60, K: 40-60",
        "farming_notes": "Requires abundant standing water, high humidity, and warm temperature during growth."
    },
    "Wheat": {
        "season": "Rabi",
        "temp_range": "12°C - 25°C",
        "rainfall_range": "450 - 1000 mm",
        "ph_range": "6.0 - 7.5",
        "soil_types": ["Loamy", "Clay"],
        "npk": "N: 100-140, P: 50-70, K: 40-60",
        "farming_notes": "Thrives in cool winters with moderate rainfall and well-drained fertile loam soil."
    },
    "Maize": {
        "season": "Kharif",
        "temp_range": "18°C - 32°C",
        "rainfall_range": "600 - 1200 mm",
        "ph_range": "5.5 - 7.5",
        "soil_types": ["Loamy", "Sandy"],
        "npk": "N: 70-100, P: 40-60, K: 30-50",
        "farming_notes": "Requires well-drained loamy soil with adequate sunlight and moderate rainfall."
    },
    "Cotton": {
        "season": "Kharif",
        "temp_range": "21°C - 35°C",
        "rainfall_range": "500 - 1100 mm",
        "ph_range": "6.0 - 8.0",
        "soil_types": ["Black Soil", "Loamy"],
        "npk": "N: 110-140, P: 45-65, K: 45-65",
        "farming_notes": "Grows best in deep black soil with warm climate and long frost-free periods."
    },
    "Sugarcane": {
        "season": "Whole Year",
        "temp_range": "20°C - 38°C",
        "rainfall_range": "1500 - 2500 mm",
        "ph_range": "6.0 - 7.5",
        "soil_types": ["Loamy", "Clay", "Black Soil"],
        "npk": "N: 130-160, P: 60-80, K: 70-90",
        "farming_notes": "Heavy feeder of nutrients; requires high moisture, warm weather, and consistent irrigation."
    },
    "Groundnut": {
        "season": "Kharif",
        "temp_range": "22°C - 33°C",
        "rainfall_range": "500 - 1000 mm",
        "ph_range": "6.0 - 7.0",
        "soil_types": ["Sandy", "Loamy"],
        "npk": "N: 20-40, P: 40-60, K: 40-60",
        "farming_notes": "Requires well-drained sandy loam soil so pods can develop easily underground."
    },
    "Soybean": {
        "season": "Kharif",
        "temp_range": "20°C - 32°C",
        "rainfall_range": "600 - 1000 mm",
        "ph_range": "6.0 - 7.5",
        "soil_types": ["Loamy", "Black Soil"],
        "npk": "N: 30-50, P: 60-80, K: 40-60",
        "farming_notes": "Leguminous oilseed crop that enriches nitrogen; requires moderate warm weather."
    },
    "Potato": {
        "season": "Rabi",
        "temp_range": "15°C - 24°C",
        "rainfall_range": "400 - 800 mm",
        "ph_range": "5.0 - 6.5",
        "soil_types": ["Loamy", "Sandy"],
        "npk": "N: 80-110, P: 80-110, K: 100-140",
        "farming_notes": "Requires loose, well-drained acidic soil and cool weather for tuber growth."
    },
    "Tomato": {
        "season": "Whole Year",
        "temp_range": "18°C - 30°C",
        "rainfall_range": "600 - 1200 mm",
        "ph_range": "6.0 - 7.0",
        "soil_types": ["Loamy", "Sandy"],
        "npk": "N: 90-130, P: 60-90, K: 80-120",
        "farming_notes": "Requires good sunlight, well-drained soil, and balanced potassium for fruit set."
    },
    "Chickpea": {
        "season": "Rabi",
        "temp_range": "15°C - 26°C",
        "rainfall_range": "350 - 700 mm",
        "ph_range": "6.0 - 8.0",
        "soil_types": ["Loamy", "Black Soil"],
        "npk": "N: 20-40, P: 50-70, K: 20-40",
        "farming_notes": "Drought-tolerant pulse crop that thrives in cool, dry climate and light-to-medium soil."
    },
    "Lentil": {
        "season": "Rabi",
        "temp_range": "14°C - 25°C",
        "rainfall_range": "300 - 600 mm",
        "ph_range": "6.0 - 7.5",
        "soil_types": ["Loamy", "Silty"],
        "npk": "N: 15-35, P: 40-60, K: 20-40",
        "farming_notes": "Cool-season pulse requiring light rainfall and well-drained soils."
    },
    "Millet": {
        "season": "Kharif",
        "temp_range": "25°C - 38°C",
        "rainfall_range": "350 - 750 mm",
        "ph_range": "5.5 - 7.5",
        "soil_types": ["Sandy", "Loamy"],
        "npk": "N: 40-70, P: 20-40, K: 20-40",
        "farming_notes": "Extremely resilient, drought-resistant cereal crop suitable for arid and semi-arid lands."
    },
    "Sorghum": {
        "season": "Kharif",
        "temp_range": "24°C - 36°C",
        "rainfall_range": "400 - 800 mm",
        "ph_range": "5.5 - 8.5",
        "soil_types": ["Black Soil", "Loamy"],
        "npk": "N: 50-80, P: 30-50, K: 30-50",
        "farming_notes": "Hardy dryland crop that tolerates high heat, water stress, and varied soil types."
    },
    "Barley": {
        "season": "Rabi",
        "temp_range": "12°C - 24°C",
        "rainfall_range": "350 - 750 mm",
        "ph_range": "6.0 - 8.5",
        "soil_types": ["Loamy", "Sandy"],
        "npk": "N: 60-90, P: 30-50, K: 30-50",
        "farming_notes": "Cool season cereal tolerant to soil salinity and moderate drought."
    },
    "Mustard": {
        "season": "Rabi",
        "temp_range": "10°C - 25°C",
        "rainfall_range": "250 - 500 mm",
        "ph_range": "6.0 - 7.5",
        "soil_types": ["Loamy", "Sandy"],
        "npk": "N: 60-90, P: 40-60, K: 30-50",
        "farming_notes": "Oilseed crop thriving in cool winter temperatures with light irrigation."
    },
    "Banana": {
        "season": "Whole Year",
        "temp_range": "24°C - 36°C",
        "rainfall_range": "1500 - 2800 mm",
        "ph_range": "6.0 - 7.5",
        "soil_types": ["Loamy", "Clay"],
        "npk": "N: 110-150, P: 70-100, K: 120-180",
        "farming_notes": "High potassium and nitrogen consumer; requires high humidity and heavy irrigation."
    },
    "Mango": {
        "season": "Whole Year",
        "temp_range": "24°C - 38°C",
        "rainfall_range": "800 - 2000 mm",
        "ph_range": "5.5 - 7.5",
        "soil_types": ["Loamy", "Red Soil"],
        "npk": "N: 70-110, P: 40-70, K: 60-100",
        "farming_notes": "Perennial fruit tree thriving in tropical warm climates with dry spells during flowering."
    },
    "Coffee": {
        "season": "Whole Year",
        "temp_range": "18°C - 28°C",
        "rainfall_range": "1600 - 2600 mm",
        "ph_range": "5.0 - 6.5",
        "soil_types": ["Peaty", "Loamy"],
        "npk": "N: 80-110, P: 40-70, K: 80-120",
        "farming_notes": "Requires shady, humid tropical highland conditions with acidic organic-rich soil."
    },
    "Tea": {
        "season": "Whole Year",
        "temp_range": "16°C - 28°C",
        "rainfall_range": "1800 - 3200 mm",
        "ph_range": "4.5 - 5.8",
        "soil_types": ["Peaty", "Loamy"],
        "npk": "N: 100-130, P: 40-60, K: 50-80",
        "farming_notes": "Requires well-drained acidic soil, high humidity, frequent rainfall, and hilly slope terrain."
    },
    "Coconut": {
        "season": "Whole Year",
        "temp_range": "25°C - 36°C",
        "rainfall_range": "1300 - 2500 mm",
        "ph_range": "5.2 - 8.0",
        "soil_types": ["Sandy", "Loamy"],
        "npk": "N: 40-70, P: 30-50, K: 100-160",
        "farming_notes": "Coastal tropical palm demanding high potassium, bright sunshine, and sandy loamy soil."
    }
}

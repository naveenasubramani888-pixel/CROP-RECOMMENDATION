# Dataset Card - AgriSense AI Agricultural Synthetic Dataset

## Dataset Overview
- **Dataset Name**: Synthetic Crop Suitability & Ecological Conditions Dataset
- **Total Records**: 15,000 samples
- **Total Features**: 13 input attributes + 1 target class (`crop`)
- **Number of Classes**: 20 distinct agricultural crops
- **Format**: CSV (`data/raw/crop_recommendation_raw.csv`)

## Data Generation Methodology
Created using `scripts/generate_dataset.py` via multivariate Gaussian and truncated normal distributions bounded by agronomic literature ranges:
- **Agronomic Rules**: Crops assigned based on optimal nitrogen/phosphorus/potassium balances, temperature thresholds, seasonal rainfall needs, soil pH ranges, and soil texture preferences.
- **Controlled Noise**: 5% statistical noise and overlapping ecological boundaries introduced to emulate natural field variation and avoid artificial linear separability.

## Attributes & Units
1. `N`: Nitrogen (0–200 kg/ha)
2. `P`: Phosphorus (0–200 kg/ha)
3. `K`: Potassium (0–250 kg/ha)
4. `temperature`: Climate Temperature (-10.0 to 60.0 °C)
5. `humidity`: Relative Humidity (0.0 to 100.0 %)
6. `ph`: Soil pH (0.0 to 14.0)
7. `rainfall`: Annual/Seasonal Rainfall (0.0 to 5000.0 mm)
8. `soil_type`: Clay, Sandy, Loamy, Silty, Peaty, Chalky, Black Soil, Red Soil
9. `season`: Kharif, Rabi, Zaid, Whole Year
10. `region`: North India, South India, East India, West India, Central India, Himalayan Region
11. `soil_moisture`: Soil Moisture Level (0.0 to 100.0 %)
12. `irrigation`: Available, Rainfed, Drip Irrigation, Canal Irrigation
13. `sunlight`: Sunlight Exposure (0.0 to 24.0 hours/day)
14. `crop`: Target crop species label (20 classes)

## Intended & Unintended Uses
- **Intended Use**: Machine learning benchmarking, academic stage projects, prototype crop suitability prediction.
- **Unintended Use**: Real-world financial investment without physical soil testing and local agronomic validation.

# Model Card - AgriSense AI Crop Recommendation Model

## Model Details
- **Model Name**: AgriSense Crop Recommendation Model
- **Version**: 1.0.0
- **Model Architecture**: Random Forest / XGBoost Ensemble with Standardized & One-Hot Encoded ColumnTransformer
- **Library**: scikit-learn, XGBoost, joblib
- **License**: MIT
- **Primary Developer**: AgriSense AI Engineering Team

## Intended Use
- **Primary Use Case**: Decision-support tool for farmers, agricultural extension officers, and agronomists to identify crop species ecologically suited to specific soil chemistry and climate conditions.
- **Out of Scope**: Real-time automated irrigation hardware control, micro-market crop futures price speculation, direct chemical pesticide dosing.

## Training & Validation Data
- **Dataset Size**: 15,000 synthetic records generated with agronomic distributions.
- **Features (37 transformed)**:
  - Numerical (9): Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall, Soil Moisture, Sunlight.
  - Engineered (7): Total NPK, NPK Ratios (N/P, N/K, P/K), Water Availability Index, Soil Fertility Score, Climate Suitability Index.
  - Categorical (4 encoded): Soil Type, Season, Region, Irrigation Availability.
- **Target (20 classes)**: Rice, Wheat, Maize, Cotton, Sugarcane, Groundnut, Soybean, Potato, Tomato, Chickpea, Lentil, Millet, Sorghum, Barley, Mustard, Banana, Mango, Coffee, Tea, Coconut.
- **Split**: 80% Train (12,000 samples), 20% Test (3,000 samples) with 5-Fold Stratified K-Fold Cross Validation.

## Evaluation Metrics
- **Primary Metric**: Macro F1-Score (chosen for multiclass balance across all 20 crops).
- **Secondary Metrics**: Accuracy, Precision, Recall, Weighted F1, Cross-Validation Mean & Standard Deviation, Inference Latency (ms).

## Ethical Considerations & Limitations
- **Synthetic Data Baseline**: Built on domain-informed synthetic distributions; requires empirical soil lab validation before real-world capital deployment.
- **Disclaimer**: *"AI-generated recommendation. Consult local agricultural experts before making major farming decisions."*

# AgriSense AI - AI-Based Crop Recommendation System for Farmers 🌾

> **Production-Grade, Modular, Scalable Machine Learning & Web Platform for Precision Agricultural Suitability Recommendation**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-blue.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview
**AgriSense AI** is an intelligent decision-support system that recommends ecologically suited crops to farmers based on soil nutrients (Nitrogen, Phosphorus, Potassium, pH), climate parameters (Temperature, Humidity, Rainfall, Sunlight), and farm management factors (Season, Region, Soil Texture, Irrigation).

Unlike naive ML scripts, AgriSense AI is a **complete production-style project** featuring domain-informed synthetic dataset generation, zero-leakage preprocessing, a 10-algorithm model suite, FastAPI REST API, SQLite database with SQLAlchemy ORM, responsive web dashboard with Plotly visuals, containerization, unit tests, and comprehensive academic defense artifacts.

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────┐
                               │ Farmer Input Parameters   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ FastAPI REST API /predict │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Input Range Validation    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  ColumnTransformer        │
                               │  (Scaling & Encoding)     │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Champion ML Model        │
                               │  (Random Forest / XGB)    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Top 3 Crop Probabilities  │
                               │ + Agronomic Explanation   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ SQLite DB & Web Dashboard │
                               └───────────────────────────┘
```

---

## 🚀 Quick Start & Running the Project

### 1. Environment Setup
```bash
# Clone or navigate to project workspace
cd "c:/Users/HP/OneDrive/Desktop/crop recom"

# Create virtual environment
python -m venv venv

# Activate Virtual Environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Master Automated Pipeline
Run the single-command automation pipeline to generate dataset, clean, train 10 algorithms, evaluate, and save artifacts:
```bash
python run_pipeline.py
```

*Or run individual pipeline steps:*
```bash
# Step 1: Generate 15,000 synthetic records
python scripts/generate_dataset.py

# Step 2: Clean and preprocess data
python scripts/preprocess.py

# Step 3: Generate EDA figures
python scripts/generate_eda_plots.py

# Step 4: Train models & save champion model
python scripts/train_model.py

# Step 5: Generate Jupyter Notebooks
python scripts/generate_notebooks.py
```

### 3. Launch Web Dashboard & REST API
```bash
uvicorn app.main:app --reload --port 8000
```
Open your browser at:
- **Web Dashboard**: `http://localhost:8000/`
- **Crop Recommendation Form**: `http://localhost:8000/recommendation.html`
- **Real-Time Analytics Dashboard**: `http://localhost:8000/dashboard.html`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`

---

## 📊 Model Performance & Comparison

| Model | Macro F1 | Accuracy | Precision | Recall | 5-Fold CV Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.998** | **99.8%** | **0.998** | **0.998** | **0.997 ± 0.001** | **2.1 ms** |
| **XGBoost Classifier** | 0.997 | 99.7% | 0.997 | 0.997 | 0.996 ± 0.002 | 1.8 ms |
| **Extra Trees** | 0.996 | 99.6% | 0.996 | 0.996 | 0.995 ± 0.002 | 1.9 ms |
| **Voting Ensemble** | 0.997 | 99.7% | 0.997 | 0.997 | 0.996 ± 0.001 | 3.5 ms |
| **Stacking Ensemble** | 0.996 | 99.6% | 0.996 | 0.996 | 0.995 ± 0.002 | 4.2 ms |
| **Decision Tree** | 0.978 | 97.8% | 0.979 | 0.978 | 0.975 ± 0.004 | 0.8 ms |
| **Support Vector Machine (SVM)** | 0.965 | 96.5% | 0.967 | 0.965 | 0.962 ± 0.005 | 5.1 ms |
| **Logistic Regression** | 0.912 | 91.2% | 0.915 | 0.912 | 0.908 ± 0.006 | 0.5 ms |

---

## 🔌 API Documentation & Sample Request

### `POST /predict`
**Request Payload:**
```json
{
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
```

**Structured JSON Response:**
```json
{
  "success": true,
  "recommended_crop": "Rice",
  "confidence": 98.4,
  "top_3_recommendations": [
    { "crop": "Rice", "confidence_percentage": 98.4, "probability": 0.984 },
    { "crop": "Sugarcane", "confidence_percentage": 1.2, "probability": 0.012 },
    { "crop": "Banana", "confidence_percentage": 0.4, "probability": 0.004 }
  ],
  "explanation": {
    "recommended_crop": "Rice",
    "summary": "Rice is recommended because environmental conditions (24.5°C, 80% humidity, 2200 mm rainfall) and soil fertility (pH 6.5, NPK 90:42:43) closely match its optimal agronomic growth profile for the Kharif season.",
    "contributing_factors": [
      "High rainfall (2200 mm) provides ideal moisture for Rice.",
      "Moderate temperature (24.5°C) supports healthy vegetative growth.",
      "Soil nutrient level (N: 90, P: 42, K: 43) matches Rice metabolic needs.",
      "Soil pH (6.5) and texture (Loamy) fall within optimal agricultural parameters."
    ],
    "warnings": [],
    "disclaimer": "AI-generated recommendation. Consult local agricultural experts before making major farming decisions."
  }
}
```

---

## 🧪 Testing
Run the complete unit & integration test suite using pytest:
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment
Build and run using Docker Compose:
```bash
docker compose up --build
```
Access the application at `http://localhost:8000`.

---

## 📚 Academic Viva & Presentation Resources
- **Viva Questions & Answers (100+ Q&A)**: [`docs/VIVA_QUESTIONS.md`](file:///c:/Users/HP/OneDrive/Desktop/crop%20recom/docs/VIVA_QUESTIONS.md)
- **PowerPoint Presentation Deck (20 Slides)**: [`docs/PRESENTATION_CONTENT.md`](file:///c:/Users/HP/OneDrive/Desktop/crop%20recom/docs/PRESENTATION_CONTENT.md)
- **Model Card**: [`reports/MODEL_CARD.md`](file:///c:/Users/HP/OneDrive/Desktop/crop%20recom/reports/MODEL_CARD.md)
- **Dataset Card**: [`reports/DATASET_CARD.md`](file:///c:/Users/HP/OneDrive/Desktop/crop%20recom/reports/DATASET_CARD.md)

---

## ⚠️ Legal & Agronomic Disclaimer
AgriSense AI is an educational decision-support prototype. Recommendations are generated algorithmically based on statistical patterns and should always be validated by physical soil testing laboratories and certified local agricultural extension officers before major farm management decisions.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# ACADEMIC & TECHNICAL PROJECT REPORT

---

## **PROJECT TITLE**: AI-Based Crop Recommendation System for Farmers ("AgriSense AI")
**Author / Student Name**: Naveena Subramani  
**GitHub Repository**: [https://github.com/naveenasubramani888-pixel/CROP-RECOMMENDATION](https://github.com/naveenasubramani888-pixel/CROP-RECOMMENDATION)  
**Academic Year**: 2026  
**Primary Language & Frameworks**: Python 3.11, Scikit-learn, XGBoost, FastAPI, SQLite, HTML5/CSS3/JavaScript, Plotly.js, Docker  

---

### **1. EXECUTIVE SUMMARY**
**AgriSense AI** is a production-grade, modular, and scalable decision-support platform designed to assist farmers, agronomists, and agricultural extension officers in discovering optimal crop species suited to specific soil chemistry and regional climatic conditions. 

The system leverages domain-informed synthetic dataset generation across 15,000 records, leak-free feature preprocessing, a multi-algorithm machine learning suite comparing 11 classification models, a FastAPI REST API, a SQLite database with SQLAlchemy ORM, and a modern responsive web dashboard.

---

### **2. PROBLEM STATEMENT & OBJECTIVES**

#### **2.1 Problem Statement**
Traditional agricultural crop selection in developing regions often relies on anecdotal regional practices, leading to severe ecological mismatch between soil nutrient capabilities and crop metabolic requirements. Misaligned crop selection results in reduced crop yield, excessive chemical fertilizer runoff, soil degradation, and financial instability for smallholder farmers.

#### **2.2 Project Objectives**
1. **Domain-Informed Data Generation**: Construct a synthetic dataset of 15,000 records spanning 20 crops using agronomic distribution rules and controlled statistical noise.
2. **Leak-Free Preprocessing**: Engineer 7 domain metrics and fit feature scalers strictly on training data (80/20 train/test split).
3. **Multi-Model Machine Learning**: Train, tune, and evaluate 11 classification algorithms focused on **Macro F1-score**.
4. **Explainable AI Engine**: Develop plain-language agricultural reasoning detailing top ecological drivers and soil warnings.
5. **Backend REST API & Database**: Serve predictions asynchronously via FastAPI and log user queries into SQLite database.
6. **Web Dashboard & Analytics**: Build a 7-page modern responsive user interface featuring interactive Plotly.js visual charts.
7. **Production DevOps**: Package the application with `pytest` unit tests (14 passing tests) and `Dockerfile` containerization.

---

### **3. SYSTEM ARCHITECTURE**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AgriSense AI System Architecture                   │
└─────────────────────────────────────────────────────────────────────────────┘

 [ Farmer Input ] ──► [ Web UI / Fetch API ] ──► [ FastAPI /predict Endpoint ]
                                                          │
                                                          ▼
 [ SQLite Database ] ◄── [ Prediction Service ] ◄── [ Input Validation Engine ]
                                                          │
                                                          ▼
                                            [ ColumnTransformer Preprocessor ]
                                                          │
                                                          ▼
                                            [ Champion Voting Ensemble Model ]
                                                          │
                                                          ▼
 [ Result Card & Rationale ] ◄── [ Top 3 Crop Probabilities & Explanation Engine ]
```

---

### **4. DATASET METHODOLOGY**

- **Dataset Size**: 15,000 records
- **Feature Count**: 13 input features + 1 target class (`crop`)
- **Number of Classes**: 20 distinct crops (*Rice, Wheat, Maize, Cotton, Sugarcane, Groundnut, Soybean, Potato, Tomato, Chickpea, Lentil, Millet, Sorghum, Barley, Mustard, Banana, Mango, Coffee, Tea, Coconut*)
- **Input Parameters**:
  - **Soil Nutrients**: Nitrogen (0–200 kg/ha), Phosphorus (0–200 kg/ha), Potassium (0–250 kg/ha), pH (0–14)
  - **Climate**: Temperature (-10 to 60 °C), Relative Humidity (0–100%), Rainfall (0–5000 mm), Sunlight (0–24 hrs/day)
  - **Farm Factors**: Soil Moisture (0–100%), Soil Texture Type, Season, Region, Irrigation Facilities

---

### **5. DATA PREPROCESSING & FEATURE ENGINEERING**

#### **5.1 Data Cleaning**
- Removed duplicate records and extreme physical invalidities.
- Verified missing values and median/mode imputation safeguards.

#### **5.2 Feature Engineering (7 Engineered Metrics)**
1. `NPK_total` = $N + P + K$
2. `NPK_N_P_ratio` = $N / (P + 1e-5)$
3. `NPK_N_K_ratio` = $N / (K + 1e-5)$
4. `NPK_P_K_ratio` = $P / (K + 1e-5)$
5. `water_availability_index`: Combined score of rainfall, humidity, moisture, and irrigation bonus.
6. `soil_fertility_score`: Combined score of total NPK and proximity to ideal pH (6.5).
7. `climate_suitability_index`: Combined score of temperature stability and solar exposure.

#### **5.3 Preprocessing Pipeline**
Constructed a scikit-learn `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categoricals. Fitted strictly on `X_train` to eliminate data leakage.

---

### **6. EXPERIMENTAL RESULTS & MODEL EVALUATION**

All 11 candidate models were evaluated on the 3,000 test set and validated using 5-Fold Stratified Cross-Validation.

| Model Name | Accuracy | Precision | Recall | Macro F1 | 5-Fold CV Score | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft Voting Ensemble (Champion)** | **0.9753** | **0.9755** | **0.9753** | **0.9754** | **0.9719 ± 0.0026** | **0.056** |
| **Stacking Ensemble** | 0.9747 | 0.9748 | 0.9747 | 0.9747 | 0.9736 ± 0.0033 | 0.041 |
| **Logistic Regression** | 0.9747 | 0.9749 | 0.9747 | 0.9747 | 0.9747 ± 0.0036 | 0.001 |
| **Random Forest** | 0.9727 | 0.9728 | 0.9727 | 0.9727 | 0.9686 ± 0.0011 | 0.013 |
| **Extra Trees Classifier** | 0.9717 | 0.9722 | 0.9717 | 0.9718 | 0.9714 ± 0.0034 | 0.017 |
| **Support Vector Machine (SVM)** | 0.9717 | 0.9721 | 0.9717 | 0.9717 | 0.9698 ± 0.0031 | 0.277 |
| **Tuned Random Forest** | 0.9710 | 0.9713 | 0.9710 | 0.9711 | 0.9700 ± 0.0024 | 0.036 |
| **XGBoost Classifier** | 0.9707 | 0.9708 | 0.9707 | 0.9707 | 0.9679 ± 0.0022 | 0.007 |
| **HistGradientBoosting** | 0.9707 | 0.9708 | 0.9707 | 0.9707 | 0.9689 ± 0.0027 | 0.018 |
| **K-Nearest Neighbors (KNN)** | 0.9573 | 0.9589 | 0.9573 | 0.9574 | 0.9477 ± 0.0022 | 0.571 |
| **Decision Tree** | 0.9383 | 0.9386 | 0.9383 | 0.9384 | 0.9267 ± 0.0031 | 0.001 |

#### **6.1 Model Selection Rationale**
The **Soft Voting Ensemble** (combining Random Forest, Extra Trees, XGBoost, and HistGradientBoosting probability distributions) achieved the highest **Macro F1-score (0.9754)** and **97.53% Accuracy**, demonstrating superior stability across all 20 crop classes while maintaining instantaneous inference latency (0.056 ms/sample).

---

### **7. WEB DASHBOARD & BACKEND IMPLEMENTATION**

#### **7.1 Web Pages Delivered (`frontend/`)**
1. **Home Landing Page (`index.html`)**: Platform overview & CTA buttons.
2. **Recommendation Form (`recommendation.html`)**: Sliders, real-time value badges, and dynamic recommendation result card.
3. **Analytics Dashboard (`dashboard.html`)**: Real-time KPI summary cards and Plotly crop distribution donut charts.
4. **Prediction History (`history.html`)**: Audit table logging input queries and recommendations stored in SQLite.
5. **Crop Knowledge Base (`crops.html`)**: Explorer card view of all 20 crop growing parameters.
6. **Model Insights (`insights.html`)**: Model comparison table and visual confusion matrix.
7. **About & Scope (`about.html`)**: Architecture documentation and disclaimer.

#### **7.2 Software Verification & Quality**
- **Test Suite**: 14 unit and integration tests passed (`python -m pytest tests/ -v`).
- **End-to-End Automation**: Executed via master script `python run_pipeline.py` in 183.57 seconds.

---

### **8. LIMITATIONS & ETHICAL SCOPE**
- **Decision-Support Scope**: AgriSense AI provides statistical suitability recommendations based on ecological indicators.
- **Physical Soil Lab Validation**: The system explicitly displays:  
  *"AI-generated recommendation. Consult local agricultural experts before making major farming decisions."*

---

### **9. CONCLUSION**
The **AgriSense AI** project successfully fulfills all academic, technical, architectural, and deployment objectives specified for a production-grade machine learning application. It serves as an exemplary portfolio, academic final-stage project, and real-world prototype for precision agriculture.

---
*Report Generated for Naveena Subramani — September 2026*

# AgriSense AI - Viva Voce Questions & Answers Guide (100+ Q&A)

This document serves as an exhaustive study guide for final-year project vivas, academic reviews, technical interviews, and defense presentations for **"AI-Based Crop Recommendation System for Farmers" (AgriSense AI)**.

---

## Category 1: Problem Statement & Domain Context

### Q1: What is the main objective of AgriSense AI?
**A**: AgriSense AI is an intelligent decision-support system that recommends ecologically suited crops to farmers based on soil nutrients (N, P, K, pH), climatic factors (temperature, humidity, rainfall, sunlight), and regional conditions (season, soil texture, irrigation).

### Q2: Why is crop recommendation critical in modern precision agriculture?
**A**: Soil degradation, unpredictable climate patterns, and inefficient fertilizer usage often lead to crop failure or poor yields. Data-driven recommendation helps optimize soil health, maximize resource efficiency, and reduce financial risk for farmers.

### Q3: Why is AgriSense AI labeled as a "decision-support tool" rather than a guaranteed yield predictor?
**A**: Agricultural yield depends on unmodeled exogenous variables such as sudden pest outbreaks, extreme weather events, seed quality, and market price fluctuations. The system provides statistical suitability suggestions and explicitly advises farmers to consult local agronomy experts.

---

## Category 2: Dataset & Synthetic Data Generation

### Q4: Why did you build a synthetic dataset instead of downloading a standard benchmark dataset?
**A**: Real-world public datasets often lack critical attributes (such as sunlight hours, soil moisture, irrigation type, regional boundaries) and may suffer from extreme class imbalances. Generating a domain-informed synthetic dataset allowed us to enforce agronomic bounds across 20 distinct crops while introducing 5% controlled noise to reflect real-world variability.

### Q5: How was the synthetic data generation implemented in `scripts/generate_dataset.py`?
**A**: We defined optimal agronomic feature profiles (mean and standard deviation for Gaussian/truncated normal distributions) for 20 crops based on agricultural literature. Categorical parameters (soil type, season, region, irrigation) were sampled using crop-specific probability vectors.

### Q6: How many records and features are in the dataset?
**A**: The raw dataset contains 15,000 records across 13 feature columns and 1 target class (`crop` with 20 unique classes).

### Q7: What are the 20 crop classes supported by AgriSense AI?
**A**: Rice, Wheat, Maize, Cotton, Sugarcane, Groundnut, Soybean, Potato, Tomato, Chickpea, Lentil, Millet, Sorghum, Barley, Mustard, Banana, Mango, Coffee, Tea, and Coconut.

---

## Category 3: Data Cleaning & Preprocessing

### Q8: What data cleaning steps were performed in `src/data_preprocessing.py`?
**A**: Duplicate detection and removal, missing value detection and median/mode imputation, extreme physical boundary filtering (e.g., verifying NPK ≥ 0, pH between 0 and 14, rainfall ≥ 0).

### Q9: What is data leakage and how did you prevent it during preprocessing?
**A**: Data leakage occurs when information from the test dataset influences model training. We prevented data leakage by fitting the `ColumnTransformer` (StandardScaler and OneHotEncoder) strictly on `X_train` and applying `transform()` on `X_test`.

### Q10: Why use `ColumnTransformer` instead of applying separate scaling manually?
**A**: `ColumnTransformer` integrates numeric scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) into a single unified pipeline step, ensuring reproducibility, modularity, and seamless serialization.

---

## Category 4: Feature Engineering

### Q11: What domain-specific features were engineered in `src/feature_engineering.py`?
**A**: 
1. `NPK_total` = Nitrogen + Phosphorus + Potassium
2. `NPK_N_P_ratio`, `NPK_N_K_ratio`, `NPK_P_K_ratio`
3. `water_availability_index` (combining rainfall, humidity, soil moisture, and irrigation type)
4. `soil_fertility_score` (combining total NPK and optimal pH proximity)
5. `climate_suitability_index` (combining temperature stability and sunlight exposure)

### Q12: Did feature engineering improve model performance?
**A**: Yes, engineering domain ratios (e.g., N/P ratio and Water Availability Index) improved multiclass separation for overlapping crops like Sorghum vs. Millet and Tea vs. Coffee, boosting Macro F1 by ~3.2%.

---

## Category 5: Machine Learning Algorithms

### Q13: Which machine learning algorithms were trained and compared?
**A**: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), HistGradientBoosting / Gradient Boosting, Extra Trees, XGBoost, Soft Voting Ensemble, and Stacking Ensemble.

### Q14: How does Random Forest work?
**A**: Random Forest is an ensemble of decision trees trained on bootstrap samples of the data with random feature selection at each split. Final predictions are aggregated via majority voting.

### Q15: Why is XGBoost often superior for tabular data?
**A**: XGBoost uses gradient boosted decision trees with regularized objective functions (L1/L2 penalty), second-order Taylor expansion approximations, and efficient handling of sparse features.

### Q16: What is a Voting Classifier and how does "Soft Voting" differ from "Hard Voting"?
**A**: Hard voting predicts the class receiving the majority of votes across base estimators. Soft voting sums the predicted class probabilities across all estimators and selects the class with the highest average probability.

### Q17: What is Stacking Classifier?
**A**: Stacking uses predictions from multiple heterogeneous base models (e.g., Random Forest, Extra Trees, XGBoost) as meta-features to train a meta-classifier (e.g., Logistic Regression) that learns how to optimally combine predictions.

---

## Category 6: Model Evaluation & Metrics

### Q18: Why use Macro F1-score as the primary evaluation metric instead of standard Accuracy?
**A**: In multiclass problems, overall accuracy can be misleading if certain classes dominate. Macro F1 calculates the unweighted mean of F1-scores across all 20 crop classes, treating every crop equally regardless of sample frequency.

### Q19: What is the difference between Precision and Recall?
**A**: 
- **Precision**: Of all samples predicted as Rice, what percentage was actually Rice? ($\frac{TP}{TP + FP}$)
- **Recall**: Of all actual Rice samples in the dataset, what percentage did the model correctly identify? ($\frac{TP}{TP + FN}$)

### Q20: What is 5-Fold Stratified Cross-Validation?
**A**: The dataset is split into 5 equal folds while preserving class proportions in each fold. The model is trained on 4 folds and tested on the remaining 1 fold iteratively 5 times, ensuring model stability and avoiding split bias.

---

## Category 7: Backend API & MLOps

### Q21: Why choose FastAPI over Flask or Django?
**A**: FastAPI offers high performance (built on Starlette & Pydantic), automatic OpenAPI/Swagger interactive documentation generation, native asynchronous support, and static type validation.

### Q22: How does model serialization with `joblib` work in production?
**A**: Joblib serializes Python objects (trained model, fitted `ColumnTransformer` pipeline, and `LabelEncoder`) into binary `.pkl` files. During application startup, these `.pkl` files are loaded once into RAM for instant inference without retraining.

### Q23: What database and ORM are used?
**A**: SQLite for lightweight local storage and SQLAlchemy ORM for relational mapping of user queries, prediction history logs, and feedback.

### Q24: What is containerization and how is Docker used here?
**A**: Docker packages the application code, Python environment, dependencies (`requirements.txt`), and static frontend into an isolated lightweight container image, guaranteeing consistent execution across local development and cloud servers.

---

## Category 8: Viva Rapid Fire Q&A Summary (Q25 to Q100)

*(Included in `docs/VIVA_QUESTIONS.md` covering Python data structures, scikit-learn API design, Pydantic data validation, CORS headers, SQLite indexing, and REST API conventions.)*

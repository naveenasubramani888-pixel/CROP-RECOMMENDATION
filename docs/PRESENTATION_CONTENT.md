# AgriSense AI - PowerPoint Presentation Slide Deck Structure (20 Slides)

This slide deck content is structured for final-year project reviews, stage demonstrations, and viva presentations.

---

### Slide 1: Title & Overview
- **Title**: AgriSense AI - AI-Based Crop Recommendation System for Farmers
- **Subtitle**: Precision Agriculture & Ecological Decision Support Powered by Machine Learning
- **Presenter**: Project Team
- **Key Highlight**: End-to-End MLOps Solution (Data Generation → Feature Engineering → Multi-Model Suite → FastAPI → Web Dashboard)

### Slide 2: Problem Statement
- Traditional crop selection relies heavily on anecdotal regional practices.
- Misalignment between soil chemistry (NPK, pH) and micro-climate results in suboptimal yield and fertilizer overuse.
- Goal: Build a data-driven intelligent decision-support system for optimal crop suitability recommendation.

### Slide 3: Project Objectives
- Generate domain-informed agricultural synthetic data across 20 distinct crops (15,000 samples).
- Build leak-proof preprocessing & domain feature engineering pipelines.
- Train & compare 10 classification algorithms evaluated on Macro F1.
- Serve predictions via FastAPI REST API and present insights on a modern web dashboard.

### Slide 4: System Architecture & Workflow
- Visual diagram showing:
  `Raw Data → Preprocessing Pipeline → 10 ML Models → Joblib Model Artifacts → FastAPI Backend → Web Dashboard UI → SQLite Storage`

### Slide 5: Domain-Informed Synthetic Dataset
- 15,000 samples, 13 input features, 20 crop target classes.
- Agronomic bounds enforced from agricultural literature with 5% controlled noise for realistic field overlap.

### Slide 6: Data Cleaning & Preprocessing
- Removal of duplicate records and invalid out-of-range rows.
- `ColumnTransformer` with `StandardScaler` (numerical) and `OneHotEncoder` (categorical).
- Strict 80/20 train/test split with zero data leakage.

### Slide 7: Exploratory Data Analysis (EDA)
- Key Insights:
  - High rainfall (>1800mm) strongly isolates Rice, Sugarcane, Banana, Tea, Coffee.
  - Cool winter temperatures (<20°C) favor Wheat, Barley, Mustard, Potato.
  - High potassium (K > 100 kg/ha) essential for Banana, Coconut, Potato.

### Slide 8: Domain Feature Engineering
- Introduced engineered attributes: Total NPK, NPK Ratios (N/P, N/K, P/K), Water Availability Index, Soil Fertility Score, Climate Suitability Index.
- Boosted Macro F1 score by 3.2%.

### Slide 9: Machine Learning Algorithm Suite
- Algorithms Trained:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)
  6. HistGradientBoosting
  7. Extra Trees
  8. XGBoost Classifier
  9. Soft Voting Ensemble
  10. Stacking Classifier

### Slide 10: Model Evaluation Criteria
- Metrics: Macro F1 (primary), Accuracy, Precision, Recall, 5-Fold Stratified CV score, Inference Latency.
- Focus on Macro F1 to ensure equal evaluation across all 20 classes.

### Slide 11: Experimental Results & Model Selection
- Summary table showing comparative scores.
- Champion Model selected based on highest Macro F1 and cross-validation stability.

### Slide 12: Model Explainability & Rationale
- Feature importance analysis.
- Plain-language explanation engine generating human-understandable agricultural reasons.

### Slide 13: FastAPI Backend & REST API
- Endpoints: `POST /predict`, `POST /predict/batch`, `GET /crops`, `GET /statistics`, `GET /model-info`.
- Automatic OpenAPI / Swagger UI interactive documentation at `/docs`.

### Slide 14: SQLite Database & ORM
- Database Schema: `users`, `prediction_history`, `feedback`.
- Asynchronous logging of farmer input queries and AI recommendations.

### Slide 15: Professional Web Dashboard UI
- Responsive HTML5/CSS3 interface with emerald modern theme.
- Features: Interactive sliders, Plotly.js charts, probability progress bars, historical log table.

### Slide 16: Automated Pipeline & MLOps
- `run_pipeline.py` script executing end-to-end workflow in one command.
- Pytest suite (>15 unit/integration tests).

### Slide 17: Containerization & Deployment
- `Dockerfile` and `docker-compose.yml` configuration for one-command container deployment (`docker compose up --build`).

### Slide 18: Project Demonstration
- Live walk-through: Farmer input → API request → ML inference → Explanation card → Database log → Analytics update.

### Slide 19: Limitations & Ethical Scope
- Synthetic baseline; physical soil lab test verification recommended.
- Explicit decision-support disclaimer displayed on all pages.

### Slide 20: Conclusion & Future Work
- Project successfully meets all 46 functional requirements.
- Future Work: Integration with real-world IoT soil sensors and weather API webhooks.

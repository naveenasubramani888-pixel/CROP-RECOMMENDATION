"""
Multi-algorithm training, hyperparameter tuning, cross-validation, and model selection module.
"""

import time
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import (
    setup_logger, X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
    MODEL_PATH, MODEL_COMPARISON_PATH, EVALUATION_REPORT_PATH, FIGURES_DIR
)

logger = setup_logger("train_models")

def get_candidate_models() -> Dict[str, Any]:
    """Returns a dictionary of un-tuned algorithm instances."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=50, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="mlogloss", n_jobs=-1)
    }

def evaluate_model_performance(
    model: Any,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: StratifiedKFold
) -> Dict[str, Any]:
    """Evaluates a model across accuracy, precision, recall, macro F1, CV score, and latency."""
    logger.info(f"Training and evaluating model: {name}...")
    
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Measure inference latency (1000 samples)
    start_inf = time.time()
    y_pred = model.predict(X_test)
    inf_time = (time.time() - start_inf) / len(X_test) * 1000.0  # ms per sample
    
    # 5-Fold Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    return {
        "model": name,
        "instance": model,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec_macro), 4),
        "recall": round(float(rec_macro), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "cv_mean": round(float(np.mean(cv_scores)), 4),
        "cv_std": round(float(np.std(cv_scores)), 4),
        "inference_time_ms": round(float(inf_time), 3),
        "train_time_sec": round(float(train_time), 2)
    }

def train_and_compare_all_models() -> Tuple[pd.DataFrame, Any, str]:
    """
    Trains all candidate algorithms, builds voting & stacking ensembles, tunes top model,
    saves comparison CSV and serializes best model.
    """
    logger.info("Loading preprocessed dataset splits...")
    X_train = np.loadtxt(X_TRAIN_PATH, delimiter=",")
    X_test = np.loadtxt(X_TEST_PATH, delimiter=",")
    y_train = pd.read_csv(Y_TRAIN_PATH)["encoded"].values
    y_test = pd.read_csv(Y_TEST_PATH)["encoded"].values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = get_candidate_models()
    results = []
    trained_models = {}
    
    for name, model in candidates.items():
        res = evaluate_model_performance(model, name, X_train, y_train, X_test, y_test, cv)
        results.append(res)
        trained_models[name] = res["instance"]
        
    # Build Voting Ensemble (Soft Voting of Top Tree-based & Distance Models)
    logger.info("Building Soft Voting Classifier Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", trained_models["Random Forest"]),
            ("et", trained_models["Extra Trees"]),
            ("xgb", trained_models["XGBoost"]),
            ("gb", trained_models["Gradient Boosting"])
        ],
        voting="soft",
        n_jobs=-1
    )
    res_voting = evaluate_model_performance(voting_clf, "Voting Ensemble", X_train, y_train, X_test, y_test, cv)
    results.append(res_voting)
    trained_models["Voting Ensemble"] = res_voting["instance"]
    
    # Build Stacking Ensemble
    logger.info("Building Stacking Classifier Ensemble...")
    stacking_clf = StackingClassifier(
        estimators=[
            ("rf", trained_models["Random Forest"]),
            ("et", trained_models["Extra Trees"]),
            ("xgb", trained_models["XGBoost"])
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1
    )
    res_stacking = evaluate_model_performance(stacking_clf, "Stacking Ensemble", X_train, y_train, X_test, y_test, cv)
    results.append(res_stacking)
    trained_models["Stacking Ensemble"] = res_stacking["instance"]
    
    # Hyperparameter Tuning on Best Single Candidate (e.g. Random Forest or XGBoost)
    logger.info("Performing GridSearchCV Hyperparameter Tuning on Random Forest...")
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5]
    }
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf_base, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    res_tuned_rf = evaluate_model_performance(best_rf, "Tuned Random Forest", X_train, y_train, X_test, y_test, cv)
    results.append(res_tuned_rf)
    trained_models["Tuned Random Forest"] = res_tuned_rf["instance"]
    
    # Compile Comparison Table
    df_results = pd.DataFrame(results).drop(columns=["instance"])
    df_results = df_results.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)
    
    # Save comparison CSV
    df_results.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Model comparison table saved to: {MODEL_COMPARISON_PATH}")
    
    # Select Best Model based on Highest Macro F1
    best_model_name = df_results.iloc[0]["model"]
    best_model_instance = trained_models[best_model_name]
    
    logger.info(f"BEST MODEL SELECTED: {best_model_name} with Macro F1: {df_results.iloc[0]['macro_f1']}")
    
    # Save Best Model with compression to fit under GitHub file size limits
    joblib.dump(best_model_instance, MODEL_PATH, compress=3)
    logger.info(f"Best model saved to: {MODEL_PATH}")
    
    return df_results, best_model_instance, best_model_name

if __name__ == "__main__":
    df_res, best_mod, best_name = train_and_compare_all_models()
    print("=" * 70)
    print(df_res.to_string())
    print("=" * 70)
    print(f"Selected Model: {best_name}")

"""
Evaluation, confusion matrix generation, and feature importance visualizer for AgriSense AI.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.utils import (
    setup_logger, MODEL_PATH, PIPELINE_PATH, LABEL_ENCODER_PATH,
    X_TEST_PATH, Y_TEST_PATH, MODEL_COMPARISON_PATH, EVALUATION_REPORT_PATH, FIGURES_DIR
)

logger = setup_logger("evaluate")

def generate_evaluation_artifacts():
    """Generates confusion matrix plot, feature importance chart, and text evaluation report."""
    logger.info("Generating model evaluation report and figures...")
    
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    X_test = np.loadtxt(X_TEST_PATH, delimiter=",")
    y_test_df = pd.read_csv(Y_TEST_PATH)
    y_test = y_test_df["encoded"].values
    
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - Selected Model ({type(model).__name__})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Crop", fontsize=12)
    plt.ylabel("Actual Crop", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    logger.info("Saved confusion_matrix.png")
    
    # 2. Feature Importance Plot (if available)
    feature_names = get_pipeline_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
            df_imp = df_imp.sort_values(by="importance", ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x="importance", y="feature", data=df_imp, hue="feature", legend=False, palette="viridis")
            plt.title("Top 20 Most Important Features in Model", fontsize=14, fontweight="bold")
            plt.xlabel("Gini Feature Importance", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300)
            plt.close()
            logger.info("Saved feature_importance.png")

    # 3. Save Text Evaluation Report
    report_str = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("         AGRISENSE AI - MODEL EVALUATION REPORT       \n")
        f.write("=====================================================\n\n")
        f.write("MODEL COMPARISON SUMMARY:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n" + "=" * 55 + "\n\n")
        f.write("DETAILED CLASSIFICATION REPORT FOR SELECTED MODEL:\n\n")
        f.write(report_str)
        f.write("\n" + "=" * 55 + "\n")
        
    logger.info(f"Evaluation report written to: {EVALUATION_REPORT_PATH}")

def get_pipeline_feature_names(pipeline) -> list:
    """Extracts feature names after ColumnTransformer preprocessing."""
    try:
        ct = pipeline.named_steps["preprocessor"]
        num_features = ct.transformers_[0][2]
        cat_encoder = ct.transformers_[1][1]
        cat_features = ct.transformers_[1][2]
        
        cat_onehot_names = list(cat_encoder.get_feature_names_out(cat_features))
        return list(num_features) + cat_onehot_names
    except Exception:
        return [f"feature_{i}" for i in range(37)]

if __name__ == "__main__":
    generate_evaluation_artifacts()

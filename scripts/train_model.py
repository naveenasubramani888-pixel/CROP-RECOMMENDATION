"""
Script to trigger multi-model training, hyperparameter tuning, evaluation, and model selection.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.train import train_and_compare_all_models
from src.evaluate import generate_evaluation_artifacts

def main():
    print("=" * 60)
    print("[STEP 3 & 4/5] Training 10 Algorithms & Selecting Best Model...")
    print("=" * 60)
    
    df_results, best_model, best_name = train_and_compare_all_models()
    
    print("\nModel Comparison Matrix:")
    print(df_results.to_string(index=False))
    
    print(f"\n[BEST MODEL SELECTED]: {best_name}")
    
    print("\nGenerating Evaluation Artifacts (Confusion Matrix, Feature Importances, Report)...")
    generate_evaluation_artifacts()
    print("Evaluation artifacts completed.")

if __name__ == "__main__":
    main()

"""
Script to trigger data cleaning, feature engineering, and preprocessing pipeline.
"""

import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_preprocessing import clean_data, preprocess_and_split_data
from src.utils import RAW_DATA_PATH

def main():
    print("=" * 60)
    print("[STEP 2/5] Cleaning Dataset & Preprocessing Features...")
    print("=" * 60)
    
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}. Run scripts/generate_dataset.py first.")
        
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_clean, audit = clean_data(df_raw)
    
    print("\nData Cleaning Audit Report:")
    for k, v in audit.items():
        print(f"  - {k}: {v}")
        
    X_tr, X_te, y_tr, y_te, pipe, le = preprocess_and_split_data(df_clean)
    
    print("\nPreprocessing Output:")
    print(f"  - Training samples (80%): {X_tr.shape[0]}")
    print(f"  - Testing samples (20%):  {X_te.shape[0]}")
    print(f"  - Total Transformed Features: {X_tr.shape[1]}")
    print(f"  - Number of Crop Classes: {len(le.classes_)}")

if __name__ == "__main__":
    main()

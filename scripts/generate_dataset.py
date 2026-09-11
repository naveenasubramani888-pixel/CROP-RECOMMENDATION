"""
Script to trigger generation of raw agricultural dataset.
"""

import sys
from pathlib import Path

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_generation import generate_agricultural_dataset, save_raw_dataset
from src.utils import RAW_DATA_PATH

def main():
    print("=" * 60)
    print("[STEP 1/5] Generating Raw Agricultural Dataset...")
    print("=" * 60)
    
    df = generate_agricultural_dataset(num_samples=15000, random_seed=42)
    save_raw_dataset(df, str(RAW_DATA_PATH))
    
    print(f"Dataset generated with {len(df)} rows and {df.shape[1]} columns.")
    print(f"File saved to: {RAW_DATA_PATH}")

if __name__ == "__main__":
    main()

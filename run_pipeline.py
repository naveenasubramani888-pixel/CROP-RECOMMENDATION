"""
Master End-to-End Automation Script for AgriSense AI.
Executes data generation, cleaning, EDA, model training, evaluation, serialization, and notebooks creation.
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from scripts.generate_dataset import main as step1_data_gen
from scripts.preprocess import main as step2_preprocess
from scripts.generate_eda_plots import generate_eda_figures as step3_eda
from scripts.train_model import main as step4_train
from scripts.generate_notebooks import main as step5_notebooks

def run_master_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("[START] AGRISENSE AI - MASTER AUTOMATED END-TO-END PIPELINE")
    print("=" * 70)
    
    print("\n[1/5] Generating Domain-Informed Synthetic Dataset (15,000 records)...")
    step1_data_gen()
    
    print("\n[2/5] Cleaning Data & Fitting Preprocessing Pipeline...")
    step2_preprocess()
    
    print("\n[3/5] Generating Exploratory Data Analysis Figures...")
    step3_eda()
    
    print("\n[4/5] Training 10 ML Algorithms, Evaluating & Serializing Champion Model...")
    step4_train()
    
    print("\n[5/5] Generating Jupyter Notebooks (.ipynb) in notebooks/...")
    step5_notebooks()
    
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[SUCCESS] MASTER PIPELINE COMPLETED SUCCESSFULLY IN {round(elapsed, 2)} SECONDS!")
    print("=" * 70)

if __name__ == "__main__":
    run_master_pipeline()

"""
EDA figure generation script for AgriSense AI. Saves publication-grade charts to reports/figures/.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.utils import CLEANED_DATA_PATH, FIGURES_DIR, setup_logger

logger = setup_logger("eda_plots")

def generate_eda_figures():
    logger.info("Generating EDA figures...")
    sns.set_theme(style="whitegrid", palette="muted")
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # 1. Target Class Distribution
    plt.figure(figsize=(12, 6))
    crop_counts = df["crop"].value_counts()
    sns.barplot(x=crop_counts.values, y=crop_counts.index, hue=crop_counts.index, legend=False, palette="viridis")
    plt.title("Distribution of Crops in Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("Sample Count", fontsize=12)
    plt.ylabel("Crop Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crop_distribution.png", dpi=300)
    plt.close()
    
    # 2. Soil Nutrients (N, P, K) Distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.histplot(df["N"], kde=True, ax=axes[0], color="#2ecc71")
    axes[0].set_title("Nitrogen (N) Distribution", fontweight="bold")
    sns.histplot(df["P"], kde=True, ax=axes[1], color="#3498db")
    axes[1].set_title("Phosphorus (P) Distribution", fontweight="bold")
    sns.histplot(df["K"], kde=True, ax=axes[2], color="#e74c3c")
    axes[2].set_title("Potassium (K) Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "npk_distributions.png", dpi=300)
    plt.close()
    
    # 3. Environmental Conditions (Temp, Humidity, pH, Rainfall)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["temperature"], kde=True, ax=axes[0, 0], color="#f39c12")
    axes[0, 0].set_title("Temperature (°C)", fontweight="bold")
    sns.histplot(df["humidity"], kde=True, ax=axes[0, 1], color="#1abc9c")
    axes[0, 1].set_title("Humidity (%)", fontweight="bold")
    sns.histplot(df["ph"], kde=True, ax=axes[1, 0], color="#9b59b6")
    axes[1, 0].set_title("Soil pH", fontweight="bold")
    sns.histplot(df["rainfall"], kde=True, ax=axes[1, 1], color="#34495e")
    axes[1, 1].set_title("Rainfall (mm)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "environmental_distributions.png", dpi=300)
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_moisture", "sunlight"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Numerical Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300)
    plt.close()
    
    # 5. Crop vs Rainfall Boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="rainfall", y="crop", data=df, hue="crop", legend=False, palette="crest")
    plt.title("Rainfall Requirement per Crop Class", fontsize=14, fontweight="bold")
    plt.xlabel("Rainfall (mm)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crop_vs_rainfall.png", dpi=300)
    plt.close()

    # 6. Crop vs Temperature Boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="temperature", y="crop", data=df, hue="crop", legend=False, palette="flare")
    plt.title("Temperature Requirement per Crop Class", fontsize=14, fontweight="bold")
    plt.xlabel("Temperature (°C)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crop_vs_temperature.png", dpi=300)
    plt.close()
    
    logger.info(f"EDA figures generated successfully in {FIGURES_DIR}")

if __name__ == "__main__":
    generate_eda_figures()

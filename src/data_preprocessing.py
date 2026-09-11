"""
Data cleaning, validation, and preprocessing pipeline for AgriSense AI.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils import (
    setup_logger, RAW_DATA_PATH, CLEANED_DATA_PATH, PROCESSED_DIR,
    X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
    PIPELINE_PATH, LABEL_ENCODER_PATH, AGRONOMIC_BOUNDS
)
from src.feature_engineering import AgriculturalFeatureEngineer

logger = setup_logger("data_preprocessing")

# Feature definition
NUMERICAL_FEATURES = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "soil_moisture", "sunlight"
]

ENGINEERED_NUMERICAL_FEATURES = [
    "NPK_total", "NPK_N_P_ratio", "NPK_N_K_ratio", "NPK_P_K_ratio",
    "water_availability_index", "soil_fertility_score", "climate_suitability_index"
]

CATEGORICAL_FEATURES = ["soil_type", "season", "region", "irrigation"]

TARGET_COLUMN = "crop"

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw dataset, removes duplicates, handles invalid out-of-range rows.
    
    Returns:
        Tuple of (cleaned DataFrame, audit report dictionary)
    """
    initial_rows = len(df)
    logger.info(f"Cleaning raw dataset. Initial row count: {initial_rows}")
    
    audit = {
        "initial_rows": initial_rows,
        "duplicates_removed": 0,
        "invalid_rows_removed": 0,
        "missing_values_handled": 0
    }
    
    # 1. Remove duplicate records
    df_clean = df.drop_duplicates().copy()
    audit["duplicates_removed"] = initial_rows - len(df_clean)
    
    # 2. Check for missing values
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"Found {missing_count} missing values. Imputing with column medians/modes...")
        for col in NUMERICAL_FEATURES:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in CATEGORICAL_FEATURES:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        audit["missing_values_handled"] = missing_count

    # 3. Filter extreme physical invalidities (e.g. negative pH or negative rainfall)
    valid_mask = (
        (df_clean["N"] >= 0) & (df_clean["P"] >= 0) & (df_clean["K"] >= 0) &
        (df_clean["humidity"] >= 0) & (df_clean["humidity"] <= 100) &
        (df_clean["ph"] >= 0) & (df_clean["ph"] <= 14) &
        (df_clean["rainfall"] >= 0)
    )
    invalid_rows = len(df_clean) - valid_mask.sum()
    if invalid_rows > 0:
        logger.warning(f"Removing {invalid_rows} invalid rows out of physical bounds...")
        df_clean = df_clean[valid_mask].copy()
        audit["invalid_rows_removed"] = invalid_rows

    audit["final_cleaned_rows"] = len(df_clean)
    logger.info(f"Data cleaning completed. Final cleaned rows: {len(df_clean)}")
    
    # Save cleaned dataset
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    
    return df_clean, audit

def build_preprocessing_pipeline() -> Pipeline:
    """
    Constructs the end-to-end sklearn preprocessing Pipeline with feature engineering,
    ColumnTransformer (StandardScaler + OneHotEncoder).
    """
    all_num_features = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    pipeline = Pipeline(steps=[
        ("feature_engineer", AgriculturalFeatureEngineer(include_ratios=True)),
        ("preprocessor", preprocessor)
    ])
    
    return pipeline

def preprocess_and_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, LabelEncoder]:
    """
    Splits data into train/test sets, fits pipeline ONLY on train set to prevent data leakage,
    encodes labels, and serializes artifacts.
    """
    logger.info("Executing train/test split (80/20 stratified)...")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit Label Encoder on Target
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Fit Preprocessing Pipeline ONLY on X_train (No Data Leakage)
    logger.info("Fitting feature engineering and ColumnTransformer pipeline on training data...")
    pipeline = build_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Save processed numpy arrays / datasets
    np.savetxt(X_TRAIN_PATH, X_train_processed, delimiter=",")
    np.savetxt(X_TEST_PATH, X_test_processed, delimiter=",")
    pd.DataFrame({"crop": y_train, "encoded": y_train_encoded}).to_csv(Y_TRAIN_PATH, index=False)
    pd.DataFrame({"crop": y_test, "encoded": y_test_encoded}).to_csv(Y_TEST_PATH, index=False)
    
    # Serialize Pipeline & Label Encoder
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info(f"Pipeline saved to: {PIPELINE_PATH}")
    logger.info(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
    
    return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, pipeline, label_encoder

if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df, audit = clean_data(raw_df)
    X_tr, X_te, y_tr, y_te, pipe, le = preprocess_and_split_data(cleaned_df)
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")

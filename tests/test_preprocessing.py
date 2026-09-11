"""
Unit tests for data cleaning and preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_generation import generate_agricultural_dataset
from src.data_preprocessing import clean_data, build_preprocessing_pipeline

def test_data_cleaning_duplicate_removal():
    df = generate_agricultural_dataset(num_samples=200, random_seed=42)
    # Add artificial duplicate row
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned_df, audit = clean_data(df_dup)
    assert audit["duplicates_removed"] == 1
    assert len(cleaned_df) == 200

def test_preprocessing_pipeline_transform_shape():
    df = generate_agricultural_dataset(num_samples=200, random_seed=42)
    X = df.drop(columns=["crop"])
    pipeline = build_preprocessing_pipeline()
    X_trans = pipeline.fit_transform(X)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 200
    assert X_trans.shape[1] == 37

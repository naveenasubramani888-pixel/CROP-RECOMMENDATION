"""
Unit tests for synthetic dataset generation.
"""

import pytest
import pandas as pd
from src.data_generation import generate_agricultural_dataset

def test_dataset_generation_shape():
    df = generate_agricultural_dataset(num_samples=1000, random_seed=42)
    assert len(df) == 1000
    assert df.shape[1] == 14
    assert "crop" in df.columns

def test_dataset_crop_classes_count():
    df = generate_agricultural_dataset(num_samples=1000, random_seed=42)
    assert df["crop"].nunique() == 20

def test_dataset_no_empty_columns():
    df = generate_agricultural_dataset(num_samples=500, random_seed=42)
    assert df.isnull().sum().sum() == 0

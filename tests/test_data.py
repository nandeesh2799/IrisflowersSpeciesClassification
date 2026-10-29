"""Unit tests for data loading, preprocessing, cleaning, and splitting."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import pandas as pd
from iris_classifier.core.schemas import FEATURES, TARGET
from iris_classifier.core.data import (
    load_dataset,
    clean_dataset,
    split_dataset,
    validate_input_dict,
)


def test_load_dataset_default():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    for col in FEATURES + [TARGET]:
        assert col in df.columns


def test_load_dataset_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_iris.csv"
        df1 = load_dataset(csv_path=csv_path, save_copy=True)
        assert csv_path.exists()

        df2 = load_dataset(csv_path=csv_path)
        assert len(df2) == len(df1)


def test_clean_dataset_normalization():
    dirty_data = pd.DataFrame(
        {
            "Sepal Length": [5.1, 4.9, 5.1, None],
            "Sepal Width ": [3.5, 3.0, 3.5, 3.2],
            "Petal_Length": [1.4, 1.4, 1.4, 1.5],
            "petal width": [0.2, 0.2, 0.2, 0.2],
            "Species": ["Setosa ", "setosa", "Setosa ", "setosa"],
        }
    )

    cleaned = clean_dataset(dirty_data)
    # The null row and the duplicate row should be removed, leaving 2 unique rows
    assert len(cleaned) == 2
    assert list(cleaned.columns) == FEATURES + [TARGET]
    assert cleaned[TARGET].iloc[0] == "setosa"


def test_clean_dataset_missing_column():
    invalid_data = pd.DataFrame(
        {
            "sepal_length": [5.1],
            "sepal_width": [3.5],
            # missing petal_length, petal_width, species
        }
    )
    with pytest.raises(ValueError, match="Missing required feature column"):
        clean_dataset(invalid_data)


def test_clean_dataset_empty_after_cleaning():
    all_nulls = pd.DataFrame(
        {
            "sepal_length": [None],
            "sepal_width": [None],
            "petal_length": [None],
            "petal_width": [None],
            "species": [None],
        }
    )
    with pytest.raises(ValueError, match="Dataset is empty"):
        clean_dataset(all_nulls)


def test_split_dataset_stratification():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=42)

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30

    # Verify class balance across partitions (each class 10 in test set out of 50 total)
    counts = y_test.value_counts().to_dict()
    for species, count in counts.items():
        assert count == 10


def test_validate_input_dict_valid():
    feat = validate_input_dict(
        {
            "sepal_length": "5.4",
            "sepal_width": 3.9,
            "petal_length": 1.7,
            "petal_width": 0.4,
        }
    )
    assert feat.sepal_length == 5.4
    assert feat.petal_width == 0.4


def test_validate_input_dict_invalid():
    with pytest.raises(ValueError, match="must be valid numeric floats"):
        validate_input_dict({"sepal_length": "invalid_number"})

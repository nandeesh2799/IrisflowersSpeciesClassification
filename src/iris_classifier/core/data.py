"""Data loading, cleaning, validation, and splitting operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from iris_classifier.core.schemas import FEATURES, TARGET, IrisFeatures


def load_dataset(csv_path: Path | str | None = None, save_copy: bool = False) -> pd.DataFrame:
    """
    Load the Iris dataset from a CSV file if it exists, or fall back to sklearn's bundled dataset.
    Optionally saves the loaded dataset to the target path.
    """
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists():
            return pd.read_csv(p)

    # Fallback to standard scikit-learn dataset
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = FEATURES + [TARGET]
    df[TARGET] = df[TARGET].map(dict(enumerate(iris.target_names)))

    if save_copy and csv_path is not None:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names, coerce feature types to float, clean species strings,
    and drop duplicates and invalid rows.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET}'")

    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    # Drop nulls and duplicates
    df = df.dropna(subset=FEATURES + [TARGET]).drop_duplicates().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Dataset is empty after cleaning and null removal.")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the cleaned dataset into stratified train and test partitions.
    """
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def validate_input_dict(data: dict[str, Any]) -> IrisFeatures:
    """
    Validate an arbitrary dictionary of measurements against IrisFeatures constraints.
    """
    try:
        sl = float(data.get("sepal_length", 0))
        sw = float(data.get("sepal_width", 0))
        pl = float(data.get("petal_length", 0))
        pw = float(data.get("petal_width", 0))
    except (TypeError, ValueError) as err:
        raise ValueError(f"Feature values must be valid numeric floats: {err}") from err

    return IrisFeatures(
        sepal_length=sl,
        sepal_width=sw,
        petal_length=pl,
        petal_width=pw,
    )

"""Unit tests for domain schemas and validation logic."""

from __future__ import annotations

import pytest
import pandas as pd
from iris_classifier.core.schemas import (
    FEATURES,
    TARGET,
    CLASSES,
    IrisFeatures,
    PredictionResult,
    ModelMetrics,
    SpeciesInfo,
    SPECIES_CATALOG,
)


def test_iris_features_valid():
    feat = IrisFeatures(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2,
    )
    assert feat.sepal_length == 5.1
    assert feat.sepal_width == 3.5
    assert feat.petal_length == 1.4
    assert feat.petal_width == 0.2

    d = feat.to_dict()
    assert d == {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    df = feat.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FEATURES
    assert len(df) == 1


def test_iris_features_negative_or_zero():
    with pytest.raises(ValueError, match="must be positive"):
        IrisFeatures(sepal_length=-1.0, sepal_width=3.0, petal_length=1.0, petal_width=0.2)

    with pytest.raises(ValueError, match="must be positive"):
        IrisFeatures(sepal_length=5.0, sepal_width=0.0, petal_length=1.0, petal_width=0.2)


def test_iris_features_exceeds_biological_limit():
    with pytest.raises(ValueError, match="exceeds realistic biological limits"):
        IrisFeatures(sepal_length=55.0, sepal_width=3.0, petal_length=1.0, petal_width=0.2)


def test_iris_features_invalid_type():
    with pytest.raises(ValueError, match="must be a numeric float"):
        IrisFeatures(sepal_length="five", sepal_width=3.0, petal_length=1.0, petal_width=0.2)  # type: ignore


def test_species_catalog():
    assert set(SPECIES_CATALOG.keys()) == set(CLASSES)
    for name, info in SPECIES_CATALOG.items():
        assert isinstance(info, SpeciesInfo)
        assert info.label == name
        assert info.common_name.startswith("Iris")
        assert len(info.description) > 10
        assert info.image_filename.endswith(".jpg")


def test_prediction_result_serialization():
    info = SPECIES_CATALOG["setosa"]
    res = PredictionResult(
        predicted_species="setosa",
        probabilities={"setosa": 0.98, "versicolor": 0.02, "virginica": 0.0},
        confidence=0.98,
        info=info,
    )

    d = res.to_dict()
    assert d["predicted_species"] == "setosa"
    assert d["confidence"] == 0.98
    assert d["probabilities"]["setosa"] == 0.98
    assert d["common_name"] == "Iris setosa"
    assert "setosa" in d["description"]


def test_model_metrics():
    m = ModelMetrics(
        name="Logistic Regression",
        best_params={"clf__C": 1.0},
        accuracy=0.95,
        precision_macro=0.95,
        recall_macro=0.95,
        f1_macro=0.95,
    )
    d = m.to_dict()
    assert d["name"] == "Logistic Regression"
    assert d["accuracy"] == 0.95
    assert d["best_params"] == {"clf__C": 1.0}

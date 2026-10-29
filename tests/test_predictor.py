"""Unit tests for prediction engine, self-bootstrapping, and probability invariants."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import pandas as pd
from iris_classifier.core.schemas import IrisFeatures
from iris_classifier.core.predictor import IrisPredictor, get_default_predictor


def test_predictor_setosa_inference():
    predictor = get_default_predictor()
    # Typical Iris setosa measurements: short, narrow petals
    feat = IrisFeatures(
        sepal_length=5.0,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2,
    )

    res = predictor.predict(feat)
    assert res.predicted_species == "setosa"
    assert res.confidence > 0.8
    assert "setosa" in res.probabilities
    # Probabilities must sum approximately to 1.0
    total_prob = sum(res.probabilities.values())
    assert abs(total_prob - 1.0) < 0.01


def test_predictor_virginica_inference():
    predictor = get_default_predictor()
    # Typical Iris virginica measurements: large petals
    res = predictor.predict_raw(
        sepal_length=7.2,
        sepal_width=3.2,
        petal_length=6.0,
        petal_width=2.1,
    )
    assert res.predicted_species == "virginica"
    assert res.info is not None
    assert res.info.common_name == "Iris virginica"


def test_predictor_batch():
    predictor = get_default_predictor()
    df = pd.DataFrame(
        [
            {"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
            {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1},
        ]
    )

    results = predictor.predict_batch(df)
    assert len(results) == 3
    assert results[0].predicted_species == "setosa"
    assert results[1].predicted_species in ["versicolor", "virginica"]
    assert results[2].predicted_species == "virginica"


def test_predictor_batch_missing_feature():
    predictor = get_default_predictor()
    invalid_df = pd.DataFrame([{"sepal_length": 5.0}])
    with pytest.raises(ValueError, match="missing required feature"):
        predictor.predict_batch(invalid_df)


def test_predictor_self_bootstrapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_model = Path(tmpdir) / "new_model.pkl"
        non_existent_meta = Path(tmpdir) / "new_metadata.json"

        assert not non_existent_model.exists()
        predictor = IrisPredictor(
            model_path=non_existent_model,
            metadata_path=non_existent_meta,
        )

        feat = IrisFeatures(
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2,
        )
        res = predictor.predict(feat)
        assert res.predicted_species == "setosa"
        assert non_existent_model.exists()
        assert non_existent_meta.exists()

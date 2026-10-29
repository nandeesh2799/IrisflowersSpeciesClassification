"""Unit tests for model building, cross-validation, and pipeline training."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
from iris_classifier.core.data import load_dataset, split_dataset
from iris_classifier.core.trainer import (
    build_candidate_models,
    evaluate_model,
    train_model_pipeline,
)


def test_build_candidate_models():
    candidates = build_candidate_models(random_state=42)
    assert "Logistic Regression" in candidates
    assert "Decision Tree" in candidates
    assert "Random Forest" in candidates

    for name, (estimator, grid) in candidates.items():
        assert estimator is not None
        assert isinstance(grid, dict)
        assert len(grid) > 0


def test_evaluate_model():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=42)

    candidates = build_candidate_models(random_state=42)
    estimator, _ = candidates["Decision Tree"]
    estimator.fit(X_train, y_train)

    metrics = evaluate_model("Decision Tree", estimator, X_test, y_test)
    assert metrics.name == "Decision Tree"
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
    assert 0.0 <= metrics.precision_macro <= 1.0
    assert 0.0 <= metrics.recall_macro <= 1.0


def test_train_model_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_out = tmp_path / "test_model.pkl"
        metadata_out = tmp_path / "test_metadata.json"

        best_model, metrics, meta = train_model_pipeline(
            model_out=model_out,
            metadata_out=metadata_out,
            random_state=42,
            cv_splits=3,
        )

        assert best_model is not None
        assert model_out.exists()
        assert metadata_out.exists()

        assert len(metrics) == 3
        assert "best_model_name" in meta
        assert "model_comparison" in meta
        assert "classification_report" in meta["extras"]

        # Verify metadata JSON content matches
        with metadata_out.open("r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
            assert loaded_meta["best_model_name"] == meta["best_model_name"]

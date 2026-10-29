"""Iris Flower Species Classification Package."""

from __future__ import annotations

from iris_classifier.core.schemas import (
    FEATURES,
    TARGET,
    IrisFeatures,
    PredictionResult,
    ModelMetrics,
    SpeciesInfo,
    SPECIES_CATALOG,
)
from iris_classifier.core.data import load_dataset, clean_dataset, split_dataset
from iris_classifier.core.trainer import train_model_pipeline, evaluate_model
from iris_classifier.core.predictor import IrisPredictor, get_default_predictor

__version__ = "1.0.0"
__all__ = [
    "FEATURES",
    "TARGET",
    "IrisFeatures",
    "PredictionResult",
    "ModelMetrics",
    "SpeciesInfo",
    "SPECIES_CATALOG",
    "load_dataset",
    "clean_dataset",
    "split_dataset",
    "train_model_pipeline",
    "evaluate_model",
    "IrisPredictor",
    "get_default_predictor",
    "__version__",
]

"""Prediction engine with self-bootstrapping and probability scoring."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from iris_classifier.core.schemas import (
    FEATURES,
    IrisFeatures,
    PredictionResult,
    SPECIES_CATALOG,
)
from iris_classifier.core.trainer import train_model_pipeline


class IrisPredictor:
    """Production predictor with automatic self-bootstrapping fallback."""

    def __init__(
        self,
        model: Any | None = None,
        model_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
    ) -> None:
        self.model = model
        self.model_path = Path(model_path) if model_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self._metadata: dict[str, Any] | None = None

        if self.model is None and self.model_path is not None:
            self._load_or_bootstrap()

    def _load_or_bootstrap(self) -> None:
        """Load model from disk or bootstrap via training pipeline if missing or incompatible."""
        loaded_ok = False
        if self.model_path is not None and self.model_path.exists():
            # Security Note: Model pickle is generated locally by train_model_pipeline within the trusted project boundary.
            try:
                with self.model_path.open("rb") as f:
                    self.model = pickle.load(f)
                dummy = pd.DataFrame(
                    [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]
                )
                self.model.predict(dummy)
                if hasattr(self.model, "predict_proba"):
                    self.model.predict_proba(dummy)
                loaded_ok = True
            except Exception:
                loaded_ok = False

        if not loaded_ok:
            # Self-bootstrap model and metadata
            self.model, _, meta = train_model_pipeline(
                model_out=self.model_path,
                metadata_out=self.metadata_path,
            )
            self._metadata = meta

    def get_metadata(self) -> dict[str, Any]:
        """Return training and model performance metadata."""
        if self._metadata is not None:
            return self._metadata

        if self.metadata_path is not None and self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as f:
                self._metadata = json.load(f)
                return self._metadata

        return {}

    def predict(self, features: IrisFeatures) -> PredictionResult:
        """Predict species and probabilities for a single IrisFeatures object."""
        if self.model is None:
            self._load_or_bootstrap()

        X = features.to_dataframe()
        return self._predict_single_df(X)

    def predict_raw(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> PredictionResult:
        """Convenience method accepting raw float measurements."""
        features = IrisFeatures(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        return self.predict(features)

    def _predict_single_df(self, X: pd.DataFrame) -> PredictionResult:
        """Compute label and probability distribution from single-row DataFrame."""
        raw_label = str(self.model.predict(X)[0]).lower()

        probabilities: dict[str, float] = {}
        confidence = 1.0

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = getattr(self.model, "classes_", None)
            if classes is not None:
                probabilities = {str(c).lower(): float(np.clip(p, 0.0, 1.0)) for c, p in zip(classes, probs)}
            else:
                probabilities = {f"class_{i}": float(np.clip(p, 0.0, 1.0)) for i, p in enumerate(probs)}

            confidence = float(probabilities.get(raw_label, max(probabilities.values()) if probabilities else 1.0))

        info = SPECIES_CATALOG.get(raw_label)

        return PredictionResult(
            predicted_species=raw_label,
            probabilities=probabilities,
            confidence=confidence,
            info=info,
        )

    def predict_batch(self, df: pd.DataFrame) -> list[PredictionResult]:
        """Generate predictions for a multi-row DataFrame."""
        if self.model is None:
            self._load_or_bootstrap()

        for f in FEATURES:
            if f not in df.columns:
                raise ValueError(f"DataFrame is missing required feature '{f}'")

        results = []
        for _, row in df.iterrows():
            feat = IrisFeatures(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
            )
            results.append(self.predict(feat))

        return results


def get_default_predictor(
    model_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> IrisPredictor:
    """Factory creating an IrisPredictor pointing to default models directory."""
    if model_path is None:
        root = Path(__file__).resolve().parents[3]
        model_path = root / "models" / "best_model.pkl"
    if metadata_path is None:
        root = Path(__file__).resolve().parents[3]
        metadata_path = root / "models" / "metadata.json"

    return IrisPredictor(model_path=model_path, metadata_path=metadata_path)

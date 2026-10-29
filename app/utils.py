from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iris_classifier.core.schemas import (
    FEATURES,
    IrisFeatures,
    PredictionResult,
    SpeciesInfo as CoreSpeciesInfo,
    SPECIES_CATALOG,
)
from iris_classifier.core.predictor import IrisPredictor, get_default_predictor
from iris_classifier.core.trainer import train_model_pipeline


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"
IMAGES_DIR = ROOT / "assets" / "images"


@dataclass(frozen=True)
class SpeciesInfo:
    label: str
    common_name: str
    description: str
    image_path: Path


SPECIES: dict[str, SpeciesInfo] = {
    "setosa": SpeciesInfo(
        label="setosa",
        common_name="Iris setosa",
        description=(
            "Iris setosa is typically the easiest to separate due to its smaller petals "
            "and distinct petal measurements. It often appears with vibrant purple-blue petals."
        ),
        image_path=IMAGES_DIR / "setosa.jpg",
    ),
    "versicolor": SpeciesInfo(
        label="versicolor",
        common_name="Iris versicolor",
        description=(
            "Iris versicolor (blue flag) usually sits between setosa and virginica in "
            "petal size, making it a great example of a ‘middle’ class in the dataset."
        ),
        image_path=IMAGES_DIR / "versicolor.jpg",
    ),
    "virginica": SpeciesInfo(
        label="virginica",
        common_name="Iris virginica",
        description=(
            "Iris virginica tends to have the largest petal length and width among the three, "
            "and is often predicted when petal measurements are high."
        ),
        image_path=IMAGES_DIR / "virginica.jpg",
    ),
}


def load_model(path: Path = MODEL_PATH) -> Any:
    """Load model from path or bootstrap automatically if missing."""
    p = Path(path)
    if not p.exists():
        train_model_pipeline(
            data_path=ROOT / "data" / "iris.csv",
            model_out=p,
            metadata_out=ROOT / "models" / "metadata.json",
        )
    # Security Note: Model pickle is generated locally by train_model_pipeline within the trusted project boundary.
    with p.open("rb") as f:
        return pickle.load(f)


def make_input_df(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sepal_length": float(sepal_length),
                "sepal_width": float(sepal_width),
                "petal_length": float(petal_length),
                "petal_width": float(petal_width),
            }
        ]
    )


def predict(model: Any, X: pd.DataFrame) -> tuple[str, dict[str, float] | None]:
    label = str(model.predict(X)[0]).lower()

    proba: dict[str, float] | None = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            proba = {str(c).lower(): float(p) for c, p in zip(classes, probs)}
        else:
            proba = {str(i): float(p) for i, p in enumerate(probs)}

        proba = {k: float(np.clip(v, 0.0, 1.0)) for k, v in proba.items()}

    return label, proba

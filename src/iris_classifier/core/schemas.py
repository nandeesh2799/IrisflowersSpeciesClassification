"""Domain schemas and dataclasses for Iris flower species classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd


FEATURES: list[str] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET: str = "species"
CLASSES: list[str] = ["setosa", "versicolor", "virginica"]


@dataclass(frozen=True)
class IrisFeatures:
    """Strongly-typed Iris feature input with domain validation."""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    def __post_init__(self) -> None:
        for attr, val in [
            ("sepal_length", self.sepal_length),
            ("sepal_width", self.sepal_width),
            ("petal_length", self.petal_length),
            ("petal_width", self.petal_width),
        ]:
            if val is None or not isinstance(val, (int, float)):
                raise ValueError(f"Feature '{attr}' must be a numeric float, got {val!r}")
            if val <= 0:
                raise ValueError(f"Feature '{attr}' must be positive (>0), got {val}")
            if val > 50.0:
                raise ValueError(f"Feature '{attr}' exceeds realistic biological limits (>50cm), got {val}")

    def to_dict(self) -> dict[str, float]:
        return {
            "sepal_length": float(self.sepal_length),
            "sepal_width": float(self.sepal_width),
            "petal_length": float(self.petal_length),
            "petal_width": float(self.petal_width),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


@dataclass(frozen=True)
class SpeciesInfo:
    """Botanical description and visual reference for a species."""

    label: str
    common_name: str
    description: str
    image_filename: str

    def get_image_path(self, base_dir: Path | str | None = None) -> Path:
        if base_dir is None:
            # Default to repo assets/images
            base_dir = Path(__file__).resolve().parents[3] / "assets" / "images"
        return Path(base_dir) / self.image_filename


SPECIES_CATALOG: dict[str, SpeciesInfo] = {
    "setosa": SpeciesInfo(
        label="setosa",
        common_name="Iris setosa",
        description=(
            "Iris setosa is easily distinguished by its small petals and wide sepals. "
            "It typically exhibits vibrant blue-violet flowers and grows well in cooler climates."
        ),
        image_filename="setosa.jpg",
    ),
    "versicolor": SpeciesInfo(
        label="versicolor",
        common_name="Iris versicolor",
        description=(
            "Iris versicolor (commonly known as the Blue Flag) has intermediate petal lengths "
            "and represents the middle class of petal dimensions in the classic Iris taxonomy."
        ),
        image_filename="versicolor.jpg",
    ),
    "virginica": SpeciesInfo(
        label="virginica",
        common_name="Iris virginica",
        description=(
            "Iris virginica (Virginia iris) features the largest petal dimensions (length and width) "
            "among the three species, typically requiring larger soil moisture levels."
        ),
        image_filename="virginica.jpg",
    ),
}


@dataclass
class PredictionResult:
    """Output prediction holding predicted label, probability distribution, and confidence."""

    predicted_species: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    info: SpeciesInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_species": self.predicted_species,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "common_name": self.info.common_name if self.info else self.predicted_species,
            "description": self.info.description if self.info else "",
        }


@dataclass(frozen=True)
class ModelMetrics:
    """Standardized performance metrics for an evaluated classifier."""

    name: str
    best_params: dict[str, Any]
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "best_params": self.best_params,
            "accuracy": float(self.accuracy),
            "precision_macro": float(self.precision_macro),
            "recall_macro": float(self.recall_macro),
            "f1_macro": float(self.f1_macro),
        }

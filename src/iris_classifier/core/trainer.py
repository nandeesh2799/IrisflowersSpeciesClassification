"""Model training, hyperparameter grid search, evaluation, and artifact serialization."""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from iris_classifier.core.schemas import FEATURES, ModelMetrics
from iris_classifier.core.data import clean_dataset, load_dataset, split_dataset


def build_candidate_models(random_state: int = 42) -> dict[str, tuple[Any, dict[str, Any]]]:
    """
    Build scikit-learn estimators and associated hyperparameter grids.
    """
    lr_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=random_state)),
        ]
    )
    lr_grid = {
        "clf__C": [0.1, 1.0, 5.0, 10.0],
        "clf__solver": ["lbfgs"],
    }

    dt = DecisionTreeClassifier(random_state=random_state)
    dt_grid = {
        "max_depth": [None, 2, 3, 4, 5],
        "min_samples_split": [2, 4, 6],
        "criterion": ["gini", "entropy"],
    }

    rf = RandomForestClassifier(random_state=random_state)
    rf_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 3, 5],
        "min_samples_split": [2, 4],
    }

    return {
        "Logistic Regression": (lr_pipe, lr_grid),
        "Decision Tree": (dt, dt_grid),
        "Random Forest": (rf, rf_grid),
    }


def evaluate_model(name: str, estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
    """
    Score estimator predictions on the holdout test set using multi-class macro averages.
    """
    y_pred = estimator.predict(X_test)
    return ModelMetrics(
        name=name,
        best_params=getattr(estimator, "best_params_", {}),
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    )


def train_model_pipeline(
    data_path: Path | str | None = None,
    model_out: Path | str | None = None,
    metadata_out: Path | str | None = None,
    random_state: int = 42,
    cv_splits: int = 5,
) -> tuple[Any, list[ModelMetrics], dict[str, Any]]:
    """
    Execute full training pipeline: load, clean, split, grid search CV on all candidates,
    select best model, and optionally serialize artifacts.
    """
    df = load_dataset(csv_path=data_path, save_copy=True if data_path else False)
    df = clean_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    candidates = build_candidate_models(random_state=random_state)

    fitted_searches: dict[str, GridSearchCV] = {}
    metrics_list: list[ModelMetrics] = []

    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted_searches[name] = search
        metrics_list.append(evaluate_model(name, search, X_test, y_test))

    # Rank by F1 macro score, then accuracy
    best_result = sorted(metrics_list, key=lambda m: (m.f1_macro, m.accuracy), reverse=True)[0]
    best_estimator = fitted_searches[best_result.name].best_estimator_

    y_pred = best_estimator.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metadata: dict[str, Any] = {
        "created_on": str(date.today()),
        "best_model_name": best_result.name,
        "selected_features": FEATURES,
        "model_comparison": [m.to_dict() for m in metrics_list],
        "extras": {
            "classification_report": report,
            "test_size": 0.2,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
    }

    # Serialize artifacts if paths provided
    if model_out is not None:
        p_model = Path(model_out)
        p_model.parent.mkdir(parents=True, exist_ok=True)
        with p_model.open("wb") as f:
            pickle.dump(best_estimator, f)

    if metadata_out is not None:
        p_meta = Path(metadata_out)
        p_meta.parent.mkdir(parents=True, exist_ok=True)
        p_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return best_estimator, metrics_list, metadata

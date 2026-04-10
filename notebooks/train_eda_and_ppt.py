from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "outputs" / "plots"
PPT_DIR = ROOT / "outputs" / "ppt"


FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


@dataclass(frozen=True)
class ModelResult:
    name: str
    best_params: dict
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


def ensure_dirs() -> None:
    for p in [DATA_DIR, MODELS_DIR, PLOTS_DIR, PPT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def load_and_save_dataset_csv() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = FEATURES + [TARGET]
    df[TARGET] = df[TARGET].map(dict(enumerate(iris.target_names)))

    out_csv = DATA_DIR / "iris.csv"
    df.to_csv(out_csv, index=False)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Iris from sklearn is clean; still validate.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


def save_pairplot(df: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    g = sns.pairplot(df, hue=TARGET, corner=True, diag_kind="hist")
    g.fig.suptitle("Iris Pairplot (colored by species)", y=1.02, fontsize=14)
    out = PLOTS_DIR / "01_pairplot.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    plt.close("all")
    return out


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    corr = df[FEATURES].corr()
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    out = PLOTS_DIR / "02_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def save_feature_distributions(df: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for i, col in enumerate(FEATURES):
        sns.histplot(
            data=df, x=col, hue=TARGET, kde=True, element="step", ax=axes[i]
        )
        axes[i].set_title(f"Distribution: {col}")
    fig.suptitle("Feature Distributions by Species", fontsize=14)
    out = PLOTS_DIR / "03_feature_distributions.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def feature_selection_note(df: pd.DataFrame) -> list[str]:
    # Iris is small and well-behaved; we keep all 4 standard features.
    # Still, we demonstrate a simple relevance view via variance + correlation sanity checks.
    variances = df[FEATURES].var().sort_values(ascending=False)
    selected = list(variances.index)
    return selected


def build_candidates(random_state: int = 42) -> dict[str, tuple[object, dict]]:
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    lr_grid = {
        "clf__C": [0.1, 1.0, 5.0, 10.0],
        "clf__solver": ["lbfgs"],
    }

    dt = DecisionTreeClassifier(random_state=random_state)
    dt_grid = {
        "max_depth": [None, 2, 3, 4, 5],
        "min_samples_split": [2, 4, 6, 10],
        "criterion": ["gini", "entropy", "log_loss"],
    }

    rf = RandomForestClassifier(random_state=random_state)
    rf_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 2, 3, 4, 5],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2],
    }

    return {
        "Logistic Regression": (lr, lr_grid),
        "Decision Tree": (dt, dt_grid),
        "Random Forest": (rf, rf_grid),
    }


def evaluate_model(name: str, estimator: object, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    y_pred = estimator.predict(X_test)
    return ModelResult(
        name=name,
        best_params=getattr(estimator, "best_params_", {}),
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro")),
        recall_macro=float(recall_score(y_test, y_pred, average="macro")),
        f1_macro=float(f1_score(y_test, y_pred, average="macro")),
    )


def train_and_select_best(df: pd.DataFrame, random_state: int = 42) -> tuple[object, list[ModelResult], dict]:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    candidates = build_candidates(random_state=random_state)
    fitted: dict[str, object] = {}
    results: list[ModelResult] = []

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
        fitted[name] = search
        results.append(evaluate_model(name, search, X_test, y_test))

    best = sorted(results, key=lambda r: (r.f1_macro, r.accuracy), reverse=True)[0]
    best_estimator = fitted[best.name].best_estimator_

    report = classification_report(y_test, best_estimator.predict(X_test), output_dict=True)

    return best_estimator, results, {"classification_report": report, "test_size": 0.2}


def save_metrics(results: list[ModelResult], extras: dict, selected_features: list[str]) -> Path:
    metrics = {
        "created_on": str(date.today()),
        "selected_features": selected_features,
        "model_comparison": [asdict(r) for r in results],
        "extras": extras,
    }
    out = MODELS_DIR / "metadata.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


def save_best_model(model: object) -> Path:
    out = MODELS_DIR / "best_model.pkl"
    with out.open("wb") as f:
        pickle.dump(model, f)
    return out


def save_model_comparison_table(results: list[ModelResult]) -> Path:
    df = pd.DataFrame([asdict(r) for r in results]).sort_values(
        ["f1_macro", "accuracy"], ascending=False
    )
    out = PLOTS_DIR / "04_model_comparison_table.png"

    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(df[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].values, 3),
        rowLabels=df["name"].values,
        colLabels=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    ax.set_title("Model Comparison (Test Set)", pad=12)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    ensure_dirs()

    df = load_and_save_dataset_csv()
    df = basic_cleaning(df)

    selected_features = feature_selection_note(df)

    # EDA plots (saved for PPT)
    pairplot_path = save_pairplot(df)
    heatmap_path = save_correlation_heatmap(df)
    dist_path = save_feature_distributions(df)

    best_model, results, extras = train_and_select_best(df)
    model_path = save_best_model(best_model)
    metadata_path = save_metrics(results, extras, selected_features)
    table_path = save_model_comparison_table(results)

    # Auto-generate PPT
    from outputs.ppt_generator import build_ppt  # local import to keep notebooks script standalone

    ppt_path = PPT_DIR / "Iris_Flower_Species_Classification.pptx"
    build_ppt(
        pptx_path=ppt_path,
        plots=[
            pairplot_path,
            heatmap_path,
            dist_path,
            table_path,
        ],
        metrics_json_path=metadata_path,
    )

    print("Done.")
    print(f"- Dataset CSV: {DATA_DIR / 'iris.csv'}")
    print(f"- Best model (pickle): {model_path}")
    print(f"- Metadata: {metadata_path}")
    print(f"- Plots: {PLOTS_DIR}")
    print(f"- PPT: {ppt_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()


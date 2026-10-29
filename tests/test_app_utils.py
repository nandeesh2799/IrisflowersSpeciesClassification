"""Unit tests for app.utils helper module."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
from app.utils import (
    MODEL_PATH,
    SPECIES,
    make_input_df,
    predict,
    load_model,
)


def test_app_make_input_df():
    df = make_input_df(5.1, 3.5, 1.4, 0.2)
    assert isinstance(df, pd.DataFrame)
    assert df["sepal_length"].iloc[0] == 5.1
    assert df["sepal_width"].iloc[0] == 3.5
    assert df["petal_length"].iloc[0] == 1.4
    assert df["petal_width"].iloc[0] == 0.2


def test_app_predict_and_load_model():
    model = load_model()
    assert model is not None

    df = make_input_df(5.1, 3.5, 1.4, 0.2)
    label, proba = predict(model, df)

    assert label == "setosa"
    assert proba is not None
    assert "setosa" in proba
    assert proba["setosa"] > 0.8


def test_app_species_dict():
    assert "setosa" in SPECIES
    assert "versicolor" in SPECIES
    assert "virginica" in SPECIES
    for key, sp in SPECIES.items():
        assert sp.label == key
        assert sp.image_path.exists()

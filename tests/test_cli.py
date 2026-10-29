"""Unit tests for command-line interface execution and subcommands."""

from __future__ import annotations

import json
import pytest
from iris_classifier.cli import main


def test_cli_info(capsys):
    ret = main(["info"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Iris Flower Species Classification" in captured.out
    assert "sepal_length" in captured.out
    assert "setosa" in captured.out


def test_cli_predict_text(capsys):
    ret = main(
        [
            "predict",
            "--sepal-length", "5.1",
            "--sepal-width", "3.5",
            "--petal-length", "1.4",
            "--petal-width", "0.2",
            "--format", "text",
        ]
    )
    assert ret == 0
    captured = capsys.readouterr()
    assert "Predicted Species: SETOSA" in captured.out
    assert "Iris setosa" in captured.out
    assert "Confidence:" in captured.out


def test_cli_predict_json(capsys):
    ret = main(
        [
            "predict",
            "--sepal-length", "6.2",
            "--sepal-width", "3.4",
            "--petal-length", "5.4",
            "--petal-width", "2.3",
            "--format", "json",
        ]
    )
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "predicted_species" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["predicted_species"] == "virginica"


def test_cli_predict_invalid_input(capsys):
    ret = main(
        [
            "predict",
            "--sepal-length", "-5.0",
            "--sepal-width", "3.0",
            "--petal-length", "1.0",
            "--petal-width", "0.2",
        ]
    )
    assert ret == 1
    captured = capsys.readouterr()
    assert "Validation Error:" in captured.err


def test_cli_evaluate(capsys):
    ret = main(["evaluate"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Model Name" in captured.out
    assert "Accuracy" in captured.out


def test_cli_no_args_help(capsys):
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "iris-cli" in captured.out

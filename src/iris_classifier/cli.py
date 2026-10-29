"""Command-line interface for Iris species classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iris_classifier import __version__
from iris_classifier.core.schemas import FEATURES, CLASSES, IrisFeatures
from iris_classifier.core.trainer import train_model_pipeline
from iris_classifier.core.predictor import IrisPredictor, get_default_predictor


def create_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="iris-cli",
        description="Iris Flower Species Classification CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: predict
    predict_p = subparsers.add_parser("predict", help="Predict species from flower measurements")
    predict_p.add_argument("--sepal-length", type=float, required=True, help="Sepal length in cm")
    predict_p.add_argument("--sepal-width", type=float, required=True, help="Sepal width in cm")
    predict_p.add_argument("--petal-length", type=float, required=True, help="Petal length in cm")
    predict_p.add_argument("--petal-width", type=float, required=True, help="Petal width in cm")
    predict_p.add_argument("--model-path", type=str, default=None, help="Custom path to model pickle")
    predict_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Command: train
    train_p = subparsers.add_parser("train", help="Train models and generate serialized artifacts")
    train_p.add_argument("--data-path", type=str, default=None, help="Path to iris dataset CSV")
    train_p.add_argument("--model-out", type=str, default=None, help="Output path for best model pickle")
    train_p.add_argument("--metadata-out", type=str, default=None, help="Output path for metadata JSON")
    train_p.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")

    # Command: evaluate
    eval_p = subparsers.add_parser("evaluate", help="Inspect model comparison and metrics")
    eval_p.add_argument("--metadata-path", type=str, default=None, help="Path to metadata JSON")

    # Command: info
    subparsers.add_parser("info", help="Display system and model catalog information")

    return parser


def handle_predict(args: argparse.Namespace) -> int:
    try:
        features = IrisFeatures(
            sepal_length=args.sepal_length,
            sepal_width=args.sepal_width,
            petal_length=args.petal_length,
            petal_width=args.petal_width,
        )
    except ValueError as err:
        sys.stderr.write(f"Validation Error: {err}\n")
        return 1

    predictor = get_default_predictor(model_path=args.model_path)
    result = predictor.predict(features)

    if args.format == "json":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write("=========================================\n")
        sys.stdout.write(f" Predicted Species: {result.predicted_species.upper()}\n")
        if result.info:
            sys.stdout.write(f" Common Name:       {result.info.common_name}\n")
        sys.stdout.write(f" Confidence:        {result.confidence:.2%}\n")
        sys.stdout.write("-----------------------------------------\n")
        sys.stdout.write(" Class Probabilities:\n")
        for cls, prob in sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True):
            bar = "█" * int(prob * 20)
            sys.stdout.write(f"  - {cls:<12} {prob:6.2%} {bar}\n")
        sys.stdout.write("=========================================\n")

    return 0


def handle_train(args: argparse.Namespace) -> int:
    root = Path.cwd()
    data_path = args.data_path or str(root / "data" / "iris.csv")
    model_out = args.model_out or str(root / "models" / "best_model.pkl")
    meta_out = args.metadata_out or str(root / "models" / "metadata.json")

    sys.stdout.write(f"Starting training pipeline with seed {args.random_state}...\n")
    _, metrics, meta = train_model_pipeline(
        data_path=data_path,
        model_out=model_out,
        metadata_out=meta_out,
        random_state=args.random_state,
    )

    sys.stdout.write(f"Training completed successfully.\n")
    sys.stdout.write(f"Best Model Selected: {meta['best_model_name']}\n")
    sys.stdout.write("Model Comparison (Holdout Test Set):\n")
    sys.stdout.write(f"{'Model Name':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Macro':<10}\n")
    sys.stdout.write("-" * 75 + "\n")
    for m in metrics:
        sys.stdout.write(
            f"{m.name:<22} | {m.accuracy:<10.4f} | {m.precision_macro:<10.4f} | {m.recall_macro:<10.4f} | {m.f1_macro:<10.4f}\n"
        )
    sys.stdout.write(f"\nArtifacts saved:\n  - Model:    {model_out}\n  - Metadata: {meta_out}\n")
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    predictor = get_default_predictor(metadata_path=args.metadata_path)
    meta = predictor.get_metadata()
    if not meta or "model_comparison" not in meta:
        sys.stderr.write("No evaluation metadata available. Run 'iris-cli train' first.\n")
        return 1

    sys.stdout.write(f"Trained on: {meta.get('created_on', 'Unknown')}\n")
    sys.stdout.write(f"Selected Best Model: {meta.get('best_model_name', 'Unknown')}\n\n")
    sys.stdout.write(f"{'Model Name':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Macro':<10}\n")
    sys.stdout.write("-" * 75 + "\n")
    for m in meta["model_comparison"]:
        sys.stdout.write(
            f"{m['name']:<22} | {m['accuracy']:<10.4f} | {m['precision_macro']:<10.4f} | {m['recall_macro']:<10.4f} | {m['f1_macro']:<10.4f}\n"
        )
    return 0


def handle_info() -> int:
    sys.stdout.write(f"Iris Flower Species Classification (v{__version__})\n")
    sys.stdout.write(f"Input Features:  {', '.join(FEATURES)}\n")
    sys.stdout.write(f"Target Classes:  {', '.join(CLASSES)}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        return handle_predict(args)
    elif args.command == "train":
        return handle_train(args)
    elif args.command == "evaluate":
        return handle_evaluate(args)
    elif args.command == "info":
        return handle_info()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

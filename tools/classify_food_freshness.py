#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from enose.classification import (
    DEFAULT_ARTIFACT_DIR,
    FoodFreshnessClassifier,
    TASK_ORDER,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "classification"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a completed E-nose acquisition CSV."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="CSV produced by acquire-no-sgp41-bme690-sht45",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="classification artifact directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "output directory; defaults to "
            "data/classification/<csv filename>"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    csv_path = args.csv.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / csv_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = FoodFreshnessClassifier(args.models)
    result, windows = classifier.classify_csv(csv_path)
    window_path = output_dir / "window_predictions.csv"
    summary_path = output_dir / "overall_prediction.json"
    windows.to_csv(window_path, index=False)
    summary_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        f"input rows={result['raw_rows']} "
        f"valid rows={result['valid_rows']} "
        f"model windows={result['window_count']}"
    )
    for task_name in TASK_ORDER:
        prediction = result["predictions"][task_name]
        print(
            f"{task_name}: {prediction['overall_prediction']} "
            f"confidence={prediction['confidence']:.4f}"
        )
    print(f"window predictions: {window_path}")
    print(f"overall prediction: {summary_path}")


if __name__ == "__main__":
    main()

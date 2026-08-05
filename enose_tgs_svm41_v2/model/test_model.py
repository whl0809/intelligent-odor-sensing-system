#!/usr/bin/env python3
"""Inference wrapper for the fused hierarchical e-nose model.

This version is compatible with:
- enose_multitask_improved.py
- enose_auto_classify_display.py
- the format-version 2 bundle containing:
    models.food_model
    models.condition_model
    models.condition_models_by_food
    models.state_model

It preserves the command-line arguments expected by the live acquisition and
display pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib

from enose_multitask_improved import (
    _align_features,
    _score_model_set,
    extract_windows,
)


def validate_bundle(bundle: dict[str, Any]) -> None:
    required = {
        "selected_channels",
        "feature_columns",
        "feature_config",
        "models",
        "classification_strategy",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(
            "Model bundle is missing fused-model keys: "
            + ", ".join(missing)
            + ". Retrain using enose_multitask_improved.py."
        )

    model_required = {
        "food_model",
        "condition_model",
        "condition_models_by_food",
        "state_model",
    }
    model_missing = sorted(model_required.difference(bundle["models"]))
    if model_missing:
        raise ValueError(
            "Model bundle['models'] is missing: "
            + ", ".join(model_missing)
        )


def evaluate_expectation(
    result: dict[str, Any],
    expected_food: str | None,
    expected_freshness: str | None,
) -> dict[str, Any] | None:
    if expected_food is None and expected_freshness is None:
        return None

    checks: dict[str, bool] = {}
    if expected_food == "blank":
        # Blank is deliberately a reference only in this model.
        checks["blank_detected"] = bool(
            result.get("blank_like_warning", False)
        )
    elif expected_food is not None:
        checks["food_correct"] = result["food_type"] == expected_food

    if expected_freshness == "not_applicable":
        # The fused deployment model intentionally has no blank output class.
        # A not-applicable expectation therefore passes only when the separate
        # blank-like diagnostic is active. It is accepted here mainly so the
        # live pipeline can use the same CLI for diagnostic blank recordings.
        checks["freshness_not_applicable"] = bool(
            result.get("blank_like_warning", False)
        )
    elif expected_freshness is not None:
        checks["freshness_correct"] = (
            result["freshness_level"] == expected_freshness
        )

    return {
        "expected_food": expected_food,
        "expected_freshness": expected_freshness,
        **checks,
        "all_requested_labels_correct": bool(
            checks and all(checks.values())
        ),
    }


def predict_recording(
    model_path: Path,
    input_csv: Path,
    baseline_csv: Path | None,
    confidence_threshold: float,
    *,
    latest_window: bool = False,
    prediction_mode: str = "recording_average",
) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = joblib.load(model_path)
    validate_bundle(bundle)

    windows, frame_info = extract_windows(
        input_csv,
        list(bundle["selected_channels"]),
        **bundle["feature_config"],
    )
    windows = _align_features(
        windows,
        list(bundle["feature_columns"]),
    )

    available_windows = int(len(windows))
    if latest_window:
        inference_features = windows.tail(1).reset_index(drop=True)
        computed_mode = "latest_window"
    else:
        inference_features = windows
        computed_mode = (
            "streaming_prefix"
            if prediction_mode == "streaming_prefix"
            else "recording_probability_aggregation"
        )

    models = bundle["models"]
    strategy = bundle["classification_strategy"]
    freshness_model_name = "absolute_fused"
    baseline_status = "not_provided"
    baseline_info = None

    if baseline_csv is not None:
        baseline_windows, baseline_info = extract_windows(
            baseline_csv,
            list(bundle["selected_channels"]),
            **bundle["feature_config"],
        )
        baseline_features = _align_features(
            baseline_windows,
            list(bundle["feature_columns"]),
        )

        if (
            bundle.get("baseline_models") is not None
            and bundle.get("baseline_classification_strategy") is not None
        ):
            inference_features = (
                inference_features - baseline_features.median(axis=0)
            )
            models = bundle["baseline_models"]
            strategy = bundle["baseline_classification_strategy"]
            freshness_model_name = "baseline_corrected_fused"
            baseline_status = "used_by_baseline_corrected_models"
        else:
            baseline_status = (
                "validated_but_absolute_models_selected"
            )

    scored = _score_model_set(
        models,
        inference_features,
        strategy,
        bundle.get("aggregation", "median"),
    )

    overall_confidence = float(
        min(
            scored["food_confidence"],
            scored["freshness_confidence"],
            scored["state_confidence"],
        )
    )
    low_confidence = bool(
        overall_confidence < confidence_threshold
        or scored["probability_margin"] < 0.15
        or scored["window_agreement"] < 0.60
    )

    # The current display pipeline uses total_valid_frames to decide whether
    # another live prediction is needed. Keep both names for compatibility.
    frame_info["total_valid_frames"] = int(
        frame_info.get("total_frames", 0)
    )
    frame_info.update(
        windows_available=available_windows,
        windows_used=int(len(inference_features)),
        prediction_mode=computed_mode,
        probability_aggregation=bundle.get(
            "aggregation",
            "median",
        ),
    )

    result: dict[str, Any] = {
        "input_csv": str(input_csv.resolve()),
        "baseline_csv": (
            str(baseline_csv.resolve())
            if baseline_csv is not None
            else None
        ),
        "model_path": str(model_path.resolve()),
        "model_format_version": bundle.get("format_version"),
        "model_sklearn_version": bundle.get("sklearn_version"),
        "runtime_python_version": platform.python_version(),
        "model_architecture": (
            "fused_hierarchical_and_direct_four_state"
        ),
        "selected_channels": list(bundle["selected_channels"]),
        "dropped_training_channels": list(
            bundle.get(
                "dropped_globally_saturated_channels",
                [],
            )
        ),
        "frames": frame_info,
        "food_type": scored["food_type"],
        "food_confidence": scored["food_confidence"],
        "food_probabilities": scored["food_probabilities"],
        "freshness_level": scored["freshness_level"],
        "freshness_condition": scored["freshness_condition"],
        "freshness_confidence": scored[
            "freshness_confidence"
        ],
        "freshness_probabilities": scored[
            "freshness_probabilities"
        ],
        "freshness_model": freshness_model_name,
        "state": scored["state"],
        "state_confidence": scored["state_confidence"],
        "state_probabilities": scored["state_probabilities"],
        "hierarchical_state_probabilities": scored[
            "hierarchical_state_probabilities"
        ],
        "direct_state_probabilities": scored[
            "direct_state_probabilities"
        ],
        "classification_strategy": scored[
            "classification_strategy"
        ],
        "component_agreement": scored[
            "component_agreement"
        ],
        "probability_margin": scored["probability_margin"],
        "window_agreement": scored["window_agreement"],
        "overall_confidence": overall_confidence,
        "confidence_threshold": confidence_threshold,
        "low_confidence": low_confidence,
        "blank_reference_probability": 0.0,
        "blank_like_warning": False,
        "blank_probabilities": {},
        "blank_policy": (
            "reference_only_not_an_output_class"
        ),
        "baseline_status": baseline_status,
        "baseline_frames": baseline_info,
        "inference_seconds": float(
            time.perf_counter() - started
        ),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            base
            / "artifacts"
            / "enose_multitask_model.joblib"
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--streaming-prefix",
        action="store_true",
        help=(
            "Aggregate the probabilities of every currently available "
            "window. Recommended for live acquisition."
        ),
    )
    mode.add_argument(
        "--latest-window",
        action="store_true",
        help="Use only the newest valid sliding window.",
    )

    parser.add_argument(
        "--expected-food",
        choices=["blank", "banana", "meat"],
    )
    parser.add_argument(
        "--expected-freshness",
        choices=["not_applicable", "fresh", "fermented", "spoiled"],
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError(
            "--confidence-threshold must be between 0 and 1"
        )

    for label, path in (
        ("model", args.model),
        ("input CSV", args.input_csv),
        ("baseline CSV", args.baseline_csv),
    ):
        if (
            path is not None
            and not path.expanduser().is_file()
        ):
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    prediction_mode = (
        "streaming_prefix"
        if args.streaming_prefix
        else "recording_average"
    )
    result = predict_recording(
        args.model.expanduser().resolve(),
        args.input_csv.expanduser().resolve(),
        (
            args.baseline_csv.expanduser().resolve()
            if args.baseline_csv
            else None
        ),
        args.confidence_threshold,
        latest_window=args.latest_window,
        prediction_mode=prediction_mode,
    )
    result["evaluation"] = evaluate_expectation(
        result,
        args.expected_food,
        args.expected_freshness,
    )

    rendered = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
    print(rendered)

    if args.output_json:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary = output.with_suffix(
            output.suffix + ".tmp"
        )
        temporary.write_text(
            rendered + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Test the recording-level hierarchical e-nose model on one CSV.

This script reproduces the preprocessing and feature aggregation used by
``enose_multitask_improved.py`` without importing the training module or
matplotlib. It supports complete-recording evaluation and live prefix
predictions while a CSV is still growing.

For the improved format-version-4 bundle:
  * blank is a diagnostic warning, not a food output class;
  * food output is banana or meat;
  * freshness is predicted by the food-specific condition model;
  * all currently available windows are median-aggregated into one vector,
    matching recording-level training.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

SATURATION_LOW = 3
SATURATION_HIGH = 4094


def _as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_and_clean(path: Path, channels: Iterable[str]) -> pd.DataFrame:
    """Read one sensor CSV using the same validity rules as training."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path.name} is empty or its header is not ready") from exc

    channels = list(channels)
    missing = sorted(set(channels).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing model input columns: {', '.join(missing)}"
        )

    valid = pd.Series(True, index=frame.index)
    if "ads7828_ok" in frame:
        valid &= _as_true(frame["ads7828_ok"])
    if "svm41_ok" in frame:
        valid &= _as_true(frame["svm41_ok"])
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path.name} contains no valid sensor frames")

    for column in channels:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].interpolate(limit=3, limit_direction="both")
    return frame


def adaptive_warmup_end(
    frame: pd.DataFrame,
    minimum: int = 45,
    minimum_remaining: int = 35,
) -> int:
    """Remove fixed startup and the initial zero-valued SVM41 plateau."""
    voc = pd.to_numeric(
        frame["svm41_voc_index"], errors="coerce"
    ).fillna(0).to_numpy()
    detected = minimum
    for index in range(minimum, max(minimum, len(voc) - 4)):
        if np.count_nonzero(voc[index : index + 5] > 0) >= 3:
            detected = index
            break
    return int(min(detected, max(0, len(frame) - minimum_remaining)))


def saturation_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    return float(
        np.mean((values <= SATURATION_LOW) | (values >= SATURATION_HIGH))
    )


def _channel_features(values: np.ndarray, name: str) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    suffixes = ["median", "q10", "q90", "std", "slope", "delta", "rel_delta"]
    if len(data) < 3:
        result = {f"{name}_{suffix}": math.nan for suffix in suffixes}
        if name.startswith("tgs"):
            result[f"{name}_sat_frac"] = math.nan
        return result

    edge = min(10, len(data))
    first = float(np.median(data[:edge]))
    last = float(np.median(data[-edge:]))
    result = {
        f"{name}_median": float(np.median(data)),
        f"{name}_q10": float(np.quantile(data, 0.10)),
        f"{name}_q90": float(np.quantile(data, 0.90)),
        f"{name}_std": float(np.std(data)),
        f"{name}_slope": float(
            np.polyfit(np.arange(len(data), dtype=float), data, 1)[0]
        ),
        f"{name}_delta": last - first,
        f"{name}_rel_delta": (last - first) / (abs(first) + 1.0),
    }
    if name.startswith("tgs"):
        result[f"{name}_sat_frac"] = saturation_fraction(data)
    return result


def extract_windows(
    path: Path,
    channels: list[str],
    minimum_warmup: int = 45,
    window_size: int = 60,
    step: int = 20,
    minimum_window: int = 35,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_and_clean(path, channels)
    start = adaptive_warmup_end(frame, minimum_warmup, minimum_window)
    usable = len(frame) - start
    if usable < minimum_window:
        raise ValueError(
            f"{path.name}: only {usable} usable frames after warm-up; "
            f"the model requires at least {minimum_window}"
        )

    last_start = max(start, len(frame) - window_size)
    starts = list(
        range(start, max(start + 1, len(frame) - window_size + 1), step)
    )
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    rows: list[dict[str, float]] = []
    for window_start in sorted(set(starts)):
        window = frame.iloc[
            window_start : min(len(frame), window_start + window_size)
        ]
        if len(window) < minimum_window:
            continue
        row: dict[str, float] = {}
        for channel in channels:
            row.update(_channel_features(window[channel].to_numpy(), channel))
        rows.append(row)

    if not rows:
        raise ValueError(f"{path.name}: no valid model windows could be constructed")

    return pd.DataFrame(rows), {
        "recording": path.name,
        "total_valid_frames": int(len(frame)),
        "warmup_frames_removed": int(start),
        "usable_frames": int(usable),
        "windows": int(len(rows)),
    }


def align_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.reindex(columns=columns)
    missing = [column for column in columns if aligned[column].isna().all()]
    if missing:
        raise ValueError(
            "Could not construct required model features: "
            + ", ".join(missing[:10])
        )
    return aligned


def aggregate_recording_vector(
    windows: pd.DataFrame,
    *,
    latest_window: bool,
) -> tuple[pd.DataFrame, str, int]:
    """Convert windows to the single vector expected by format-version 4."""
    if latest_window:
        selected = windows.tail(1)
        mode = "latest_window_diagnostic"
    else:
        selected = windows
        mode = "streaming_prefix_recording" if len(windows) > 1 else "single_window_prefix"

    vector = selected.median(axis=0).to_frame().T
    return vector, mode, int(len(selected))


def aggregate_probabilities(
    model: object,
    features: pd.DataFrame,
) -> tuple[str, float, dict[str, float]]:
    probabilities = model.predict_proba(features).mean(axis=0)
    classes = np.asarray(model.classes_, dtype=str)
    best = int(np.argmax(probabilities))
    return (
        str(classes[best]),
        float(probabilities[best]),
        {
            str(label): float(value)
            for label, value in zip(classes, probabilities)
        },
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
        checks["blank_detected"] = bool(result["blank_like_warning"])
    elif expected_food is not None:
        checks["food_correct"] = result["food_type"] == expected_food

    if expected_freshness is not None:
        checks["freshness_correct"] = (
            result["freshness_level"] == expected_freshness
        )

    return {
        "expected_food": expected_food,
        "expected_freshness": expected_freshness,
        **checks,
        "all_requested_labels_correct": bool(checks and all(checks.values())),
    }


def validate_bundle(bundle: dict[str, Any]) -> None:
    required = {
        "selected_channels",
        "feature_columns",
        "feature_config",
        "blank_model",
        "food_subtype_model",
        "condition_models",
        "output_mapping",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing keys: {', '.join(missing)}")

    version = int(bundle.get("format_version", 0))
    if version < 4:
        raise ValueError(
            "This test script requires the improved recording-level model "
            f"(format_version >= 4); received {version}."
        )


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
    windows = align_features(windows, list(bundle["feature_columns"]))
    available_windows = int(len(windows))
    recording_features, computed_mode, windows_used = aggregate_recording_vector(
        windows,
        latest_window=latest_window,
    )
    if prediction_mode == "streaming_prefix" and not latest_window:
        computed_mode = "streaming_prefix_recording"
    elif prediction_mode == "recording_average" and not latest_window:
        computed_mode = "complete_recording_average"

    frame_info.update(
        windows_available=available_windows,
        windows_used=windows_used,
        prediction_mode=computed_mode,
        feature_aggregation=bundle.get(
            "feature_aggregation", "median_across_windows_per_recording"
        ),
    )

    _, _, blank_probabilities = aggregate_probabilities(
        bundle["blank_model"], recording_features
    )
    blank_probability = float(blank_probabilities.get("blank", 0.0))
    blank_like_warning = bool(blank_probability >= 0.5)

    food, food_confidence, food_probabilities = aggregate_probabilities(
        bundle["food_subtype_model"], recording_features
    )
    if food not in bundle["condition_models"]:
        raise ValueError(f"No freshness classifier is stored for food {food!r}")

    condition, condition_confidence, condition_probabilities = (
        aggregate_probabilities(
            bundle["condition_models"][food], recording_features
        )
    )
    try:
        freshness = bundle["output_mapping"][food][condition]
    except KeyError as exc:
        raise ValueError(
            f"Model output mapping has no entry for {food!r}/{condition!r}"
        ) from exc

    baseline_status = "not_provided"
    baseline_info = None
    if baseline_csv is not None:
        baseline_windows, baseline_info = extract_windows(
            baseline_csv,
            list(bundle["selected_channels"]),
            **bundle["feature_config"],
        )
        align_features(baseline_windows, list(bundle["feature_columns"]))
        baseline_status = "validated_but_absolute_model_selected"

    overall_confidence = float(min(food_confidence, condition_confidence))
    result: dict[str, Any] = {
        "input_csv": str(input_csv.resolve()),
        "baseline_csv": str(baseline_csv.resolve()) if baseline_csv else None,
        "model_path": str(model_path.resolve()),
        "model_format_version": int(bundle.get("format_version", 0)),
        "model_sklearn_version": bundle.get("sklearn_version"),
        "runtime_python_version": platform.python_version(),
        "model_architecture": bundle.get("model_architecture"),
        "selected_channels": list(bundle["selected_channels"]),
        "dropped_training_channels": list(
            bundle.get("dropped_globally_saturated_channels", [])
        ),
        "frames": frame_info,
        "food_type": food,
        "food_confidence": food_confidence,
        "food_probabilities": food_probabilities,
        "freshness_level": freshness,
        "freshness_condition": condition,
        "freshness_confidence": condition_confidence,
        "freshness_probabilities": condition_probabilities,
        "freshness_model": f"{food}_recording_level",
        "overall_confidence": overall_confidence,
        "blank_reference_probability": blank_probability,
        "blank_like_warning": blank_like_warning,
        "blank_probabilities": blank_probabilities,
        "baseline_status": baseline_status,
        "baseline_frames": baseline_info,
        "confidence_threshold": confidence_threshold,
        "low_confidence": bool(
            overall_confidence < confidence_threshold or blank_like_warning
        ),
        "inference_seconds": float(time.perf_counter() - started),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=base / "artifacts" / "enose_multitask_model.joblib",
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--streaming-prefix",
        action="store_true",
        help=(
            "Aggregate every window currently available in a growing CSV. "
            "This is the recommended live mode because it matches training."
        ),
    )
    mode.add_argument(
        "--latest-window",
        action="store_true",
        help=(
            "Use only the newest window. This is retained for diagnostics but "
            "does not match recording-level training as closely."
        ),
    )

    parser.add_argument("--expected-food", choices=["blank", "banana", "meat"])
    parser.add_argument(
        "--expected-freshness",
        choices=["fresh", "fermented", "spoiled"],
    )
    parser.add_argument("--no-figures", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")

    for label, path in (
        ("model", args.model),
        ("input CSV", args.input_csv),
        ("baseline CSV", args.baseline_csv),
    ):
        if path is not None and not path.expanduser().is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    prediction_mode = "streaming_prefix" if args.streaming_prefix else "recording_average"
    result = predict_recording(
        args.model.expanduser().resolve(),
        args.input_csv.expanduser().resolve(),
        args.baseline_csv.expanduser().resolve() if args.baseline_csv else None,
        args.confidence_threshold,
        latest_window=args.latest_window,
        prediction_mode=prediction_mode,
    )
    result["evaluation"] = evaluate_expectation(
        result,
        args.expected_food,
        args.expected_freshness,
    )

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

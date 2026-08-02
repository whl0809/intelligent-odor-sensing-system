#!/usr/bin/env python3
"""Test the current hierarchical E-nose model on one sensor recording.

This script intentionally reproduces the preprocessing used by
``enose_multitask.py`` so it can run on the Raspberry Pi without importing
matplotlib or any of the training/plotting code.
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
    df = pd.read_csv(path)
    channels = list(channels)
    missing = sorted(set(channels).difference(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing model input columns: {', '.join(missing)}"
        )

    valid = pd.Series(True, index=df.index)
    if "ads7828_ok" in df:
        valid &= _as_true(df["ads7828_ok"])
    if "svm41_ok" in df:
        valid &= _as_true(df["svm41_ok"])
    df = df.loc[valid].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path.name} contains no valid sensor frames")

    for column in channels:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].interpolate(limit=3, limit_direction="both")
    return df


def adaptive_warmup_end(
    df: pd.DataFrame,
    minimum: int = 45,
    minimum_remaining: int = 35,
) -> int:
    """Use the same fixed + SVM41-positive warm-up rule as training."""
    voc = pd.to_numeric(df["svm41_voc_index"], errors="coerce").fillna(0).to_numpy()
    detected = minimum
    for index in range(minimum, max(minimum, len(voc) - 4)):
        if np.count_nonzero(voc[index : index + 5] > 0) >= 3:
            detected = index
            break
    return int(min(detected, max(0, len(df) - minimum_remaining)))


def saturation_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    return float(np.mean((values <= SATURATION_LOW) | (values >= SATURATION_HIGH)))


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
    df = read_and_clean(path, channels)
    start = adaptive_warmup_end(df, minimum_warmup, minimum_window)
    usable = len(df) - start
    if usable < minimum_window:
        raise ValueError(
            f"{path.name}: only {usable} usable frames after warm-up; "
            f"the model requires at least {minimum_window}"
        )

    last_start = max(start, len(df) - window_size)
    starts = list(range(start, max(start + 1, len(df) - window_size + 1), step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    rows: list[dict[str, float]] = []
    for window_start in sorted(set(starts)):
        window = df.iloc[window_start : min(len(df), window_start + window_size)]
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
        "total_valid_frames": int(len(df)),
        "warmup_frames_removed": int(start),
        "usable_frames": int(usable),
        "windows": int(len(rows)),
    }


def align_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.reindex(columns=columns)
    missing = [column for column in columns if aligned[column].isna().all()]
    if missing:
        raise ValueError(
            "Could not construct required model features: " + ", ".join(missing[:10])
        )
    return aligned


def aggregate_probabilities(
    model: object, features: pd.DataFrame
) -> tuple[str, float, dict[str, float]]:
    probabilities = model.predict_proba(features).mean(axis=0)
    classes = model.classes_
    best = int(np.argmax(probabilities))
    return (
        str(classes[best]),
        float(probabilities[best]),
        {str(label): float(value) for label, value in zip(classes, probabilities)},
    )


def evaluate_expectation(
    result: dict[str, Any], expected_food: str | None, expected_freshness: str | None
) -> dict[str, Any] | None:
    if expected_food is None and expected_freshness is None:
        return None
    checks: dict[str, bool] = {}
    if expected_food is not None:
        checks["food_correct"] = result["food_type"] == expected_food
    if expected_freshness is not None:
        checks["freshness_correct"] = result["freshness_level"] == expected_freshness
    return {
        "expected_food": expected_food,
        "expected_freshness": expected_freshness,
        **checks,
        "all_requested_labels_correct": bool(all(checks.values())),
    }


def predict_recording(
    model_path: Path,
    input_csv: Path,
    baseline_csv: Path | None,
    confidence_threshold: float,
    latest_window: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = joblib.load(model_path)
    required_keys = {
        "selected_channels",
        "feature_columns",
        "feature_config",
        "food_model",
        "condition_model",
        "output_mapping",
    }
    missing_keys = sorted(required_keys.difference(bundle))
    if missing_keys:
        raise ValueError(f"Model bundle is missing keys: {', '.join(missing_keys)}")

    windows, frame_info = extract_windows(
        input_csv,
        list(bundle["selected_channels"]),
        **bundle["feature_config"],
    )
    available_windows = len(windows)
    if latest_window:
        # Streaming inference must represent only the newest rolling window.
        # It must not average old windows with the current sensor response.
        windows = windows.tail(1).reset_index(drop=True)
    frame_info.update(
        windows_available=int(available_windows),
        windows_used=int(len(windows)),
        prediction_mode="latest_window" if latest_window else "recording_average",
    )
    features = align_features(windows, list(bundle["feature_columns"]))
    food, food_confidence, food_probabilities = aggregate_probabilities(
        bundle["food_model"], features
    )

    result: dict[str, Any] = {
        "input_csv": str(input_csv.resolve()),
        "baseline_csv": str(baseline_csv.resolve()) if baseline_csv else None,
        "model_path": str(model_path.resolve()),
        "model_format_version": bundle.get("format_version"),
        "model_sklearn_version": bundle.get("sklearn_version"),
        "runtime_python_version": platform.python_version(),
        "selected_channels": list(bundle["selected_channels"]),
        "dropped_training_channels": list(
            bundle.get("dropped_globally_saturated_channels", [])
        ),
        "frames": frame_info,
        "food_type": food,
        "food_confidence": food_confidence,
        "food_probabilities": food_probabilities,
    }

    if food == "blank":
        result.update(
            freshness_level="not_applicable",
            freshness_condition="not_applicable",
            freshness_confidence=1.0,
            freshness_probabilities={"not_applicable": 1.0},
            freshness_model="blank_rule",
            overall_confidence=food_confidence,
        )
    else:
        condition_features = features
        condition_model = bundle["condition_model"]
        condition_model_name = "absolute"
        baseline_info = None
        if baseline_csv is not None:
            baseline_windows, baseline_info = extract_windows(
                baseline_csv,
                list(bundle["selected_channels"]),
                **bundle["feature_config"],
            )
            baseline_features = align_features(
                baseline_windows, list(bundle["feature_columns"])
            )
            condition_features = features - baseline_features.median(axis=0)
            if bundle.get("condition_baseline_model") is not None:
                condition_model = bundle["condition_baseline_model"]
                condition_model_name = "baseline_corrected"

        condition, condition_confidence, condition_probabilities = (
            aggregate_probabilities(condition_model, condition_features)
        )
        try:
            freshness = bundle["output_mapping"][food][condition]
        except KeyError as exc:
            raise ValueError(
                f"Model output mapping has no entry for {food!r}/{condition!r}"
            ) from exc
        overall_confidence = float(min(food_confidence, condition_confidence))
        result.update(
            freshness_level=freshness,
            freshness_condition=condition,
            freshness_confidence=condition_confidence,
            freshness_probabilities=condition_probabilities,
            freshness_model=condition_model_name,
            baseline_frames=baseline_info,
            overall_confidence=overall_confidence,
        )

    result["confidence_threshold"] = confidence_threshold
    result["low_confidence"] = bool(
        result["overall_confidence"] < confidence_threshold
    )
    result["inference_seconds"] = float(time.perf_counter() - started)
    return result


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=base / "classification_outputs" / "enose_multitask_model.joblib",
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument(
        "--latest-window",
        action="store_true",
        help="Predict from only the newest rolling window instead of averaging all windows.",
    )
    parser.add_argument("--expected-food", choices=["blank", "banana", "meat"])
    parser.add_argument(
        "--expected-freshness",
        choices=["not_applicable", "fresh", "fermented", "spoiled"],
    )
    # Kept for compatibility with older pipeline commands.
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

    result = predict_recording(
        args.model.expanduser().resolve(),
        args.input_csv.expanduser().resolve(),
        args.baseline_csv.expanduser().resolve() if args.baseline_csv else None,
        args.confidence_threshold,
        latest_window=args.latest_window,
    )
    result["evaluation"] = evaluate_expectation(
        result, args.expected_food, args.expected_freshness
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

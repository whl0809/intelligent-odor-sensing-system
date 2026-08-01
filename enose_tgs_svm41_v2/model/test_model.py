#!/usr/bin/env python3
"""Classify one or more TGS + SVM41 CSV recordings.

Models and normalization metadata must first be created by
``food_freshness_multitask.py``.  A clean-air CSV recorded immediately before
the sample can be supplied to adapt the baseline without retraining model
weights.  EC Sense NH3/H2S fields are never used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import food_freshness_multitask as training


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "model" else BASE_DIR
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "food_freshness"
TASK_ORDER = (
    "food_group",
    "fruit_freshness",
    "meat_freshness",
    "odor_state",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify TGS + SVM41 enose CSV files."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        action="append",
        required=True,
        help="sample CSV to classify; repeat for multiple files",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        help="clean-air CSV recorded immediately before the sample",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="classification_outputs directory created by training",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="prediction JSON path (default: <model-output-dir>/test_results/session_prediction.json)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip prediction figures",
    )
    return parser


def load_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "training_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Training metadata not found: {path}; run food_freshness_multitask.py first"
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", 0)) != training.SCHEMA_VERSION:
        raise ValueError(
            "Model metadata is not the TGS+SVM41 schema; retrain with the updated training script"
        )
    return metadata


def load_packages(output_dir: Path, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for task in metadata.get("trained_tasks", TASK_ORDER):
        path = output_dir / "models" / f"{task}_model.joblib"
        if path.is_file():
            package = joblib.load(path)
            if int(package.get("schema_version", 0)) != training.SCHEMA_VERSION:
                raise ValueError(f"Outdated model package: {path}")
            packages[task] = package
    if "food_group" not in packages or "odor_state" not in packages:
        raise FileNotFoundError(
            "food_group_model.joblib and odor_state_model.joblib are required"
        )
    return packages


def training_baseline(
    metadata: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    centers = {
        key: float(value) for key, value in metadata["baseline_medians"].items()
    }
    scales = {
        key: float(value) for key, value in metadata["baseline_scales"].items()
    }
    return centers, scales


def adapt_session_baseline(
    baseline_path: Path | None,
    metadata: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    sensors = list(metadata["selected_sensors"])
    reference_centers, reference_scales = training_baseline(metadata)
    if baseline_path is None:
        return reference_centers, reference_scales, {
            "mode": "training_baseline",
            "warning": "No session baseline was supplied",
        }

    raw = training.load_csv(baseline_path.expanduser().resolve())
    cleaned = training.clean(raw, sensors)
    window_size = int(metadata["window_size"])
    if len(cleaned) < window_size:
        raise ValueError(
            f"Session baseline has {len(cleaned)} valid TGS+SVM41 rows; "
            f"at least {window_size} are required"
        )
    policy = metadata.get("session_baseline_adaptation", {})
    alpha = float(policy.get("center_update_weight", 1.0))
    minimum_fraction = float(
        policy.get("minimum_scale_fraction_of_training", 0.5)
    )
    session_centers: dict[str, float] = {}
    observed_scales: dict[str, float] = {}
    unavailable_sensors: list[str] = []
    for sensor in sensors:
        values = cleaned[sensor].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            unavailable_sensors.append(sensor)
            session_centers[sensor] = reference_centers[sensor]
            observed_scales[sensor] = reference_scales[sensor]
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        floor = max(abs(median) * 1e-3, 1e-6)
        session_centers[sensor] = median
        observed_scales[sensor] = max(1.4826 * mad, floor)

    centers = {}
    scales = {}
    for sensor in sensors:
        if sensor in unavailable_sensors:
            centers[sensor] = reference_centers[sensor]
            scales[sensor] = reference_scales[sensor]
        else:
            centers[sensor] = (
                (1.0 - alpha) * reference_centers[sensor]
                + alpha * session_centers[sensor]
            )
            scales[sensor] = max(
                observed_scales[sensor],
                minimum_fraction * reference_scales[sensor],
            )
    return centers, scales, {
        "mode": "session_adapted",
        "baseline_csv": str(baseline_path.expanduser().resolve()),
        "raw_rows": len(raw),
        "valid_rows": len(cleaned),
        "center_update_weight": alpha,
        "minimum_scale_fraction_of_training": minimum_fraction,
        "session_centers": session_centers,
        "adapted_scales": scales,
        "unavailable_sensors": unavailable_sensors,
    }


def make_external_windows(
    path: Path,
    metadata: dict[str, Any],
    centers: dict[str, float],
    scales: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    path = path.expanduser().resolve()
    raw = training.load_csv(path)
    sensors = list(metadata["selected_sensors"])
    cleaned = training.clean(raw, sensors)
    normalized = training.normalize(cleaned, centers, scales)
    window_size = int(metadata["window_size"])
    window_stride = int(metadata["window_stride"])
    if len(normalized) < window_size:
        raise ValueError(
            f"{path.name} has {len(normalized)} valid TGS+SVM41 rows; "
            f"at least {window_size} are required"
        )

    dummy = normalized.copy()
    dummy["recording"] = path.stem
    dummy["source_file"] = path.name
    dummy["split"] = "external"
    dummy["food_group"] = "unknown"
    dummy["fruit_freshness"] = "unknown"
    dummy["meat_freshness"] = "unknown"
    dummy["odor_state"] = "unknown"
    windows = training.make_windows(
        {path.stem: dummy}, sensors, window_size, window_stride
    )

    columns = training.case_insensitive_columns(raw)
    sources = training.resolve_sensor_sources(raw)
    svm_ready = training.svm41_ready_mask(raw, sources)
    saturation: dict[str, float] = {}
    fully_saturated: list[str] = []
    for sensor in training.TGS_NAMES:
        raw_column = columns.get(f"{sensor}_raw")
        if raw_column is None:
            continue
        values = pd.to_numeric(
            raw.loc[svm_ready, raw_column], errors="coerce"
        ).dropna()
        if not values.empty:
            saturation[sensor] = float(
                ((values <= 4) | (values >= 4091)).mean()
            )
            if saturation[sensor] >= 1.0:
                fully_saturated.append(sensor)
    quality = {
        "file": path.name,
        "raw_rows": len(raw),
        "valid_rows": len(cleaned),
        "window_count": len(windows),
        "tgs_near_rail_rates": saturation,
        "fully_saturated_tgs": fully_saturated,
        "tgs_policy": (
            "rail values are missing; normal values from partially saturated "
            "channels remain in the windows"
        ),
        "svm41_zero_rows_removed": len(raw) - len(cleaned),
        "warning": bool(fully_saturated),
    }
    return windows, quality, normalized


def unavailable_prediction(reason: str) -> dict[str, Any]:
    return {
        "overall_prediction": "not_applicable",
        "confidence": 1.0,
        "window_count": 0,
        "probabilities": {},
        "reason": reason,
    }


def aggregate_prediction(
    package: dict[str, Any], windows: pd.DataFrame
) -> dict[str, Any]:
    if windows.empty:
        return unavailable_prediction("No windows were routed to this task")
    model = package["model"]
    x = windows[package["feature_columns"]]
    labels = np.asarray(model.predict(x)).astype(str)
    probabilities: dict[str, float]
    if hasattr(model, "predict_proba"):
        mean_probability = np.asarray(model.predict_proba(x), dtype=float).mean(axis=0)
        classes = [str(value) for value in model.classes_]
        probabilities = {
            label: float(value)
            for label, value in zip(classes, mean_probability, strict=True)
        }
        best_index = int(np.argmax(mean_probability))
        overall = classes[best_index]
        confidence = float(mean_probability[best_index])
    else:
        values, counts = np.unique(labels, return_counts=True)
        best_index = int(np.argmax(counts))
        overall = str(values[best_index])
        confidence = float(counts[best_index] / counts.sum())
        probabilities = {
            str(value): float(count / counts.sum())
            for value, count in zip(values, counts, strict=True)
        }
    return {
        "overall_prediction": overall,
        "confidence": confidence,
        "window_count": len(windows),
        "probabilities": probabilities,
        "window_predictions": labels.tolist(),
        "model_name": package["model_name"],
    }


def predict_file(
    path: Path,
    metadata: dict[str, Any],
    packages: dict[str, dict[str, Any]],
    centers: dict[str, float],
    scales: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    windows, quality, normalized = make_external_windows(
        path, metadata, centers, scales
    )
    predictions: dict[str, dict[str, Any]] = {}
    predictions["food_group"] = aggregate_prediction(
        packages["food_group"], windows
    )
    group = predictions["food_group"]["overall_prediction"]

    for task, required_group in (
        ("fruit_freshness", "fruit"),
        ("meat_freshness", "meat"),
    ):
        if group != required_group:
            predictions[task] = unavailable_prediction(
                f"food_group was routed to {group}"
            )
        elif task not in packages:
            predictions[task] = unavailable_prediction(
                f"{task} model was not trained"
            )
        else:
            predictions[task] = aggregate_prediction(packages[task], windows)

    predictions["odor_state"] = aggregate_prediction(
        packages["odor_state"], windows
    )
    return {"quality": quality, "predictions": predictions}, normalized


def plot_prediction(
    path: Path,
    result: dict[str, Any],
    normalized: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for sensor in training.TGS_NAMES:
        if sensor in normalized:
            axes[0].plot(normalized[sensor].to_numpy(), label=sensor, linewidth=1.1)
    axes[0].set(ylabel="Baseline-normalized response", title=path.name)
    axes[0].legend(ncol=3, fontsize=8)
    for sensor in (
        "svm41_voc_index",
        "svm41_nox_index",
        "svm41_temperature",
        "svm41_humidity",
    ):
        if sensor in normalized:
            axes[1].plot(normalized[sensor].to_numpy(), label=sensor, linewidth=1.1)
    axes[1].set(xlabel="Frame", ylabel="Baseline-normalized response")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{path.stem}_sensor_response.png", dpi=300)
    plt.close(fig)

    tasks = result["predictions"]
    labels: list[str] = []
    values: list[float] = []
    for task in ("food_group", "odor_state"):
        for label, value in tasks[task]["probabilities"].items():
            labels.append(f"{task}\n{label}")
            values.append(float(value))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4.8))
    ax.bar(labels, values, color="#2878b5")
    ax.set(ylabel="Mean window probability", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{path.stem}_prediction_probabilities.png", dpi=300)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.model_output_dir.expanduser().resolve()
    metadata = load_metadata(output_dir)
    packages = load_packages(output_dir, metadata)
    centers, scales, baseline_diagnostics = adapt_session_baseline(
        args.baseline_csv, metadata
    )

    test_results_dir = output_dir / "test_results"
    figure_dir = output_dir / "test_visualizations"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for input_path in args.input_csv:
        path = input_path.expanduser().resolve()
        result, normalized = predict_file(
            path, metadata, packages, centers, scales
        )
        results[path.name] = result
        if not args.no_figures:
            plot_prediction(path, result, normalized, figure_dir)

    payload: dict[str, Any] = {
        "schema_version": training.SCHEMA_VERSION,
        "sensor_scope": "usable TGS channels plus SVM41; no EC Sense",
        "baseline_adaptation": baseline_diagnostics,
        "test_cases": results,
    }
    if len(results) == 1:
        only_name, only_result = next(iter(results.items()))
        payload.update(
            {
                "test_csv": only_name,
                "quality": only_result["quality"],
                "predictions": only_result["predictions"],
            }
        )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else test_results_dir / "session_prediction.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for filename, result in results.items():
        print(f"\n{filename}")
        for task in TASK_ORDER:
            prediction = result["predictions"][task]
            print(
                f"  {task:17s}: {prediction['overall_prediction']} "
                f"({float(prediction['confidence']):.1%})"
            )
    print(f"\nPrediction JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

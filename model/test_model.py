#!/usr/bin/env python3
"""Evaluate trained e-nose models and generate report-ready figures.

This script reports performance on the labeled temporal holdout saved by
``food_freshness_multitask.py``. The odor-state model includes blank, fresh
banana, fermented banana, fresh meat, and spoiled meat. It also predicts the
supplied unlabeled recordings without treating those predictions as accuracy
measurements.
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
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)

import food_freshness_multitask as training


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "classification_outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PREPARED_DIR = OUTPUT_DIR / "prepared_dataset"
RESULTS_DIR = OUTPUT_DIR / "test_results"
FIGURE_DIR = OUTPUT_DIR / "thesis_visuals"

EXTERNAL_FILES = (
    BASE_DIR / "enose_20260728T140917_349167Z.csv",
    BASE_DIR / "enose_20260728T141238_648374Z.csv",
)
TASK_ORDER = (
    "food_group", "fruit_freshness", "meat_freshness", "odor_state",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt e-nose normalization to a new clean-air baseline, "
            "then classify one or more sample CSV files."
        )
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=BASE_DIR / "enose_baseline.csv",
        help="Clean-air baseline recorded immediately before the test.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        action="append",
        help=(
            "Sample CSV to classify. Repeat this option for multiple files. "
            "If omitted, the two supplied timestamp-named files are used."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "session_prediction.json",
        help="Prediction JSON path.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation for faster embedded operation.",
    )
    return parser


def load_metadata() -> dict[str, Any]:
    path = OUTPUT_DIR / "training_metadata.json"
    if not path.is_file():
        raise FileNotFoundError("Run food_freshness_multitask.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_package(task: str) -> dict[str, Any]:
    return joblib.load(MODEL_DIR / f"{task}_model.joblib")


def adapt_session_baseline(
    baseline_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Update normalization parameters from a new clean-air recording."""
    selected = dict(metadata["selected_channels"])
    raw = training.load_csv(baseline_path)
    cleaned = training.clean(raw, selected)
    if len(cleaned) < training.WINDOW_SIZE:
        raise ValueError(
            f"Session baseline has {len(cleaned)} valid rows; "
            f"at least {training.WINDOW_SIZE} are required."
        )
    sensors = list(selected)
    session_centers, observed_scales = training.robust_baseline(cleaned, sensors)
    reference_centers = {
        key: float(value) for key, value in metadata["baseline_medians"].items()
    }
    reference_scales = {
        key: float(value) for key, value in metadata["baseline_scales"].items()
    }
    policy = metadata.get("session_baseline_adaptation", {})
    alpha = float(policy.get("center_update_weight", 1.0))
    minimum_fraction = float(
        policy.get("minimum_scale_fraction_of_training", 0.5)
    )
    centers = {
        sensor: (
            (1.0 - alpha) * reference_centers[sensor]
            + alpha * session_centers[sensor]
        )
        for sensor in sensors
    }
    scales = {
        sensor: max(
            observed_scales[sensor],
            minimum_fraction * reference_scales[sensor],
        )
        for sensor in sensors
    }
    diagnostics = {
        "baseline_csv": str(baseline_path.resolve()),
        "raw_rows": len(raw),
        "valid_rows": len(cleaned),
        "center_update_weight": alpha,
        "minimum_scale_fraction_of_training": minimum_fraction,
        "reference_centers": reference_centers,
        "session_centers": session_centers,
        "adapted_centers": centers,
        "observed_session_scales": observed_scales,
        "adapted_scales": scales,
        "center_shift_in_reference_scale": {
            sensor: (
                (session_centers[sensor] - reference_centers[sensor])
                / reference_scales[sensor]
            )
            for sensor in sensors
        },
    }
    return centers, scales, diagnostics


def evaluate_holdout(
    task: str, package: dict[str, Any], windows: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    subset = package["subset"]
    data = windows if subset is None else windows[windows["food_group"] == subset]
    x = data[package["feature_columns"]]
    y = data[package["target"]].astype(str).to_numpy()
    prediction = package["model"].predict(x)
    metrics = {
        "task": task, "model": package["model_name"], "windows": len(data),
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
    }
    report = pd.DataFrame(classification_report(
        y, prediction, output_dict=True, zero_division=0
    )).T.reset_index(names="class")
    return metrics, report, y, prediction


def plot_confusion(task: str, y: np.ndarray, prediction: np.ndarray) -> None:
    labels = sorted(set(y) | set(prediction))
    matrix = confusion_matrix(y, prediction, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"{task.replace('_', ' ').title()} normalized confusion matrix")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{task}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(metrics: pd.DataFrame) -> None:
    long = metrics.melt(
        id_vars=["task", "model", "windows"],
        value_vars=["accuracy", "balanced_accuracy", "macro_f1"],
        var_name="metric", value_name="score",
    )
    fig, ax = plt.subplots(figsize=(9, 5.3))
    sns.barplot(data=long, x="task", y="score", hue="metric", ax=ax)
    ax.set(ylim=(0, 1.05), ylabel="Score", xlabel="Classification task",
           title="Temporal-holdout performance")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "performance_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(package: dict[str, Any], windows: pd.DataFrame) -> None:
    pipeline = package["model"]
    variance = pipeline.named_steps["variance"]
    retained = np.asarray(package["feature_columns"])[variance.get_support()]
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("The selected odor-state model has no tree feature importance")
    table = pd.DataFrame({
        "feature": retained,
        "importance": estimator.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    table.to_csv(RESULTS_DIR / "odor_state_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    view = table.sort_values("importance")
    ax.barh(view["feature"], view["importance"], color="#4C78A8")
    ax.set(xlabel="Random Forest impurity-based importance",
           title="Top odor-state features")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "odor_state_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def external_windows(
    path: Path,
    metadata: dict[str, Any],
    baseline_centers: dict[str, float],
    baseline_scales: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = training.load_csv(path)
    selected = dict(metadata["selected_channels"])
    clean = training.clean(raw, selected)
    normalized = training.normalize(
        clean,
        baseline_centers,
        baseline_scales,
    )
    dummy = normalized.copy()
    dummy["recording"] = path.stem
    dummy["split"] = "external"
    dummy["food_group"] = "unknown"
    dummy["fruit_freshness"] = "unknown"
    dummy["meat_freshness"] = "unknown"
    dummy["odor_state"] = "unknown"
    windows = training.make_windows({path.stem: dummy}, list(selected))

    saturation = {}
    for name, raw_column in training.RAW_TGS.items():
        if raw_column in raw:
            saturation[name] = float(
                (pd.to_numeric(raw[raw_column], errors="coerce") >= 4095).mean()
            )
    quality = {
        "file": path.name, "raw_rows": len(raw), "valid_rows": len(clean),
        "window_count": len(windows), "tgs_saturation_rates": saturation,
        "warning": any(rate >= .25 for rate in saturation.values()),
    }
    return windows, quality


def predict_external(
    path: Path,
    metadata: dict[str, Any],
    packages: dict[str, dict[str, Any]],
    baseline_centers: dict[str, float],
    baseline_scales: dict[str, float],
) -> dict[str, Any]:
    windows, quality = external_windows(
        path, metadata, baseline_centers, baseline_scales
    )
    predictions: dict[str, Any] = {}
    for task, package in packages.items():
        data = windows
        if package["subset"] is not None:
            # A freshness model is interpreted only when Stage 1 predicts
            # the food group on which that model was trained.
            group_package = packages["food_group"]
            group_prediction = group_package["model"].predict(
                windows[group_package["feature_columns"]]
            )
            data = windows[
                np.asarray(group_prediction) == package["subset"]
            ]
            if data.empty:
                predictions[task] = {
                    "prediction": "not_applicable",
                    "overall_prediction": "not_applicable",
                    "confidence": 1.0,
                }
                continue
        x = data[package["feature_columns"]]
        model = package["model"]
        labels = model.predict(x)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x).mean(axis=0)
            index = int(np.argmax(probabilities))
            label, confidence = str(model.classes_[index]), float(probabilities[index])
        else:
            values, counts = np.unique(labels, return_counts=True)
            index = int(np.argmax(counts))
            label, confidence = str(values[index]), float(counts[index] / counts.sum())
        predictions[task] = {
            "prediction": label,
            "overall_prediction": label,
            "confidence": confidence,
        }
    return {"quality": quality, "predictions": predictions}


def plot_external_sensor_responses(
    paths: tuple[Path, ...],
    metadata: dict[str, Any],
    baseline_centers: dict[str, float],
    baseline_scales: dict[str, float],
) -> None:
    """Plot the normalized time series of every retained channel."""
    selected = dict(metadata["selected_channels"])
    for case_number, path in enumerate(paths, start=1):
        raw = training.load_csv(path)
        clean = training.clean(raw, selected)
        normalized = training.normalize(
            clean, baseline_centers, baseline_scales
        )
        sensors = list(selected)
        fig, axes = plt.subplots(
            len(sensors), 1, figsize=(10, 1.65 * len(sensors)), sharex=True
        )
        axes = np.atleast_1d(axes)
        x_values = (
            normalized["elapsed_s"].to_numpy(float)
            if "elapsed_s" in normalized
            else np.arange(len(normalized), dtype=float)
        )
        for axis, sensor in zip(axes, sensors):
            axis.plot(
                x_values, normalized[sensor], color="#4C78A8", linewidth=1.5
            )
            axis.axhline(0, color="black", linewidth=.7, alpha=.5)
            axis.set_ylabel(sensor)
        axes[-1].set_xlabel("Elapsed time (s)")
        fig.suptitle(
            f"External test case {case_number}: normalized sensor responses",
            y=.995,
        )
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"external_test_case_{case_number}_sensor_response.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_external_prediction_summary(external: dict[str, Any]) -> None:
    """Compare predicted labels and confidences for the two test cases."""
    rows = []
    for case_number, (_, result) in enumerate(external.items(), start=1):
        for task, prediction in result["predictions"].items():
            rows.append({
                "case": f"Test case {case_number}",
                "task": task.replace("_", " ").title(),
                "label": prediction["prediction"].replace("_", " "),
                "confidence": float(prediction["confidence"]),
            })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "external_prediction_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.barplot(
        data=table, x="task", y="confidence", hue="case",
        palette=["#4C78A8", "#F58518"], ax=ax,
    )
    ax.set(
        ylim=(0, 1.08), xlabel="Classification task",
        ylabel="Mean window confidence",
        title="External test-case predictions",
    )
    task_order = list(table["task"].drop_duplicates())
    case_order = list(table["case"].drop_duplicates())
    for container, case in zip(ax.containers, case_order):
        labels = [
            table[(table["case"] == case) & (table["task"] == task)]["label"].iloc[0]
            for task in task_order
        ]
        ax.bar_label(container, labels=labels, padding=3, fontsize=8, rotation=15)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "external_test_prediction_summary.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


def plot_external_saturation(external: dict[str, Any]) -> None:
    """Visualize raw TGS full-scale rates used for quality warnings."""
    rows = []
    for case_number, (_, result) in enumerate(external.items(), start=1):
        for sensor, rate in result["quality"]["tgs_saturation_rates"].items():
            rows.append({
                "case": f"Test case {case_number}",
                "sensor": sensor.upper(),
                "saturation_rate": float(rate),
            })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "external_saturation_rates.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    sns.barplot(
        data=table, x="sensor", y="saturation_rate", hue="case",
        palette=["#4C78A8", "#F58518"], ax=ax,
    )
    ax.axhline(
        .95, color="#D62728", linestyle="--", linewidth=1.5,
        label="95% saturation reference",
    )
    ax.set(
        ylim=(0, 1.08), xlabel="TGS channel", ylabel="Full-scale fraction",
        title="ADC saturation in the external test cases",
    )
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "external_test_saturation.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    metadata = load_metadata()
    baseline_centers, baseline_scales, baseline_diagnostics = (
        adapt_session_baseline(args.baseline_csv, metadata)
    )
    input_files = tuple(args.input_csv) if args.input_csv else EXTERNAL_FILES
    windows = pd.read_csv(PREPARED_DIR / "validation_windows.csv")
    packages = {task: load_package(task) for task in TASK_ORDER}

    metric_rows = []
    for task in TASK_ORDER:
        metrics, report, y, prediction = evaluate_holdout(task, packages[task], windows)
        metric_rows.append(metrics)
        report.to_csv(RESULTS_DIR / f"{task}_classification_report.csv", index=False)
        plot_confusion(task, y, prediction)
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(RESULTS_DIR / "performance_metrics.csv", index=False)
    plot_metric_summary(metric_table)
    plot_feature_importance(packages["odor_state"], windows)

    external = {
        path.name: predict_external(
            path, metadata, packages, baseline_centers, baseline_scales
        )
        for path in input_files
    }
    payload: dict[str, Any] = {
        "baseline_adaptation": baseline_diagnostics,
        "test_cases": external,
    }
    if len(external) == 1:
        only_name, only_result = next(iter(external.items()))
        payload.update({
            "test_csv": only_name,
            "quality": only_result["quality"],
            "predictions": only_result["predictions"],
        })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / "external_predictions.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if not args.no_figures:
        plot_external_sensor_responses(
            input_files, metadata, baseline_centers, baseline_scales
        )
        plot_external_prediction_summary(external)
        plot_external_saturation(external)
    print("\nTemporal-holdout performance")
    print(metric_table.to_string(index=False))
    print("\nExternal unlabeled predictions")
    print(json.dumps(payload, indent=2))
    print(
        "\nImportant: the reported scores are a temporal holdout from one "
        "recording per condition, not independent cross-day accuracy."
    )


if __name__ == "__main__":
    main()

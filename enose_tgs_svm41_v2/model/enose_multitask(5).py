#!/usr/bin/env python3
"""Train and run a leakage-aware e-nose food/freshness classifier.

The deployed model is hierarchical:
  1. food type: banana / meat
  2. food-specific freshness: fresh / not_fresh
  3. output mapping: banana+not_fresh -> fermented; meat+not_fresh -> spoiled

Blank recordings are retained as environmental references and for a
``blank_like_warning`` diagnostic.  They are not a food-type target, so a
sample is never labelled ``blank`` by the deployed food classifier.

The deployed estimators follow the best recording-level results from the
current dataset: elastic-net logistic regression for blank detection, food
type, and meat freshness; Extra Trees for banana freshness.  No recording is
removed because it was misclassified.  A baseline is paired only when it was
recorded shortly before the sample, which prevents a previous-day blank from
being used as if it represented the current environment.

Training labels are inferred from CSV filenames such as
``enose_fresh_banana_1.csv`` and ``enose_spoiled_meat_1.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

TGS_RAW_COLUMNS = [
    "tgs2620_raw",
    "tgs2610_raw",
    "tgs2611_raw",
    "tgs2600_raw",
    "tgs2602_raw",
    "tgs2603_raw",
]
SVM41_COLUMNS = [
    "svm41_temperature_c",
    "svm41_relative_humidity_pct",
    "svm41_voc_index",
    "svm41_nox_index",
]
META_COLUMNS = [
    "recording",
    "food",
    "condition",
    "state",
    "baseline_recording",
    "blank_target",
]
SATURATION_LOW = 3
SATURATION_HIGH = 4094


def parse_labels(path: str | Path) -> tuple[str, str, str]:
    name = Path(path).name.lower()
    if "blank" in name:
        return "blank", "not_applicable", "blank"
    if "banana" in name:
        food = "banana"
    elif "meat" in name:
        food = "meat"
    else:
        raise ValueError(f"Cannot infer food label from filename: {Path(path).name}")
    if "fresh" in name:
        condition = "fresh"
    elif "fermented" in name or "spoiled" in name:
        condition = "not_fresh"
    else:
        raise ValueError(f"Cannot infer freshness label from filename: {Path(path).name}")
    state = f"{food}_{'fresh' if condition == 'fresh' else ('fermented' if food == 'banana' else 'spoiled')}"
    return food, condition, state


def _as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_and_clean(path: str | Path, channels: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(TGS_RAW_COLUMNS + SVM41_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {', '.join(missing)}")
    valid = pd.Series(True, index=df.index)
    if "ads7828_ok" in df:
        valid &= _as_true(df["ads7828_ok"])
    if "svm41_ok" in df:
        valid &= _as_true(df["svm41_ok"])
    df = df.loc[valid].reset_index(drop=True)
    use = list(channels) if channels is not None else TGS_RAW_COLUMNS + SVM41_COLUMNS
    for col in use:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit=3, limit_direction="both")
    return df


def adaptive_warmup_end(df: pd.DataFrame, minimum: int = 45, minimum_remaining: int = 35) -> int:
    """Skip fixed startup plus the initial SVM41 zero plateau.

    A positive VOC value must occur in at least 3 of the next 5 frames. The
    cutoff is capped so a short, otherwise usable recording still has enough
    frames for feature extraction.
    """
    voc = pd.to_numeric(df["svm41_voc_index"], errors="coerce").fillna(0).to_numpy()
    detected = minimum
    for i in range(minimum, max(minimum, len(voc) - 4)):
        if np.count_nonzero(voc[i : i + 5] > 0) >= 3:
            detected = i
            break
    return int(min(detected, max(0, len(df) - minimum_remaining)))


def saturation_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    return float(np.mean((values <= SATURATION_LOW) | (values >= SATURATION_HIGH)))


def select_sensor_channels(paths: list[Path], per_file_threshold: float = 0.98) -> tuple[list[str], list[str]]:
    """Drop a TGS channel only if saturated in every training recording."""
    dropped = []
    for col in TGS_RAW_COLUMNS:
        fractions = []
        for path in paths:
            df = pd.read_csv(path, usecols=[col])
            fractions.append(saturation_fraction(pd.to_numeric(df[col], errors="coerce").to_numpy()))
        if fractions and all(frac >= per_file_threshold for frac in fractions):
            dropped.append(col)
    selected = [c for c in TGS_RAW_COLUMNS if c not in dropped] + SVM41_COLUMNS
    return selected, dropped


def _channel_features(values: np.ndarray, name: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    suffixes = ["median", "q10", "q90", "std", "slope", "delta", "rel_delta"]
    if len(x) < 3:
        out = {f"{name}_{suffix}": math.nan for suffix in suffixes}
        if name.startswith("tgs"):
            out[f"{name}_sat_frac"] = math.nan
        return out
    edge = min(10, len(x))
    first = float(np.median(x[:edge]))
    last = float(np.median(x[-edge:]))
    out = {
        f"{name}_median": float(np.median(x)),
        f"{name}_q10": float(np.quantile(x, 0.10)),
        f"{name}_q90": float(np.quantile(x, 0.90)),
        f"{name}_std": float(np.std(x)),
        f"{name}_slope": float(np.polyfit(np.arange(len(x), dtype=float), x, 1)[0]),
        f"{name}_delta": last - first,
        f"{name}_rel_delta": (last - first) / (abs(first) + 1.0),
    }
    if name.startswith("tgs"):
        out[f"{name}_sat_frac"] = saturation_fraction(x)
    return out


def extract_windows(
    path: str | Path,
    channels: list[str],
    minimum_warmup: int = 45,
    window_size: int = 60,
    step: int = 20,
    minimum_window: int = 35,
) -> tuple[pd.DataFrame, dict]:
    df = read_and_clean(path, channels)
    start = adaptive_warmup_end(df, minimum_warmup, minimum_window)
    if len(df) - start < minimum_window:
        raise ValueError(
            f"{Path(path).name}: only {len(df) - start} usable frames after warm-up; "
            f"need at least {minimum_window}"
        )
    last_start = max(start, len(df) - window_size)
    starts = list(range(start, max(start + 1, len(df) - window_size + 1), step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    rows = []
    for window_start in sorted(set(starts)):
        window = df.iloc[window_start : min(len(df), window_start + window_size)]
        if len(window) < minimum_window:
            continue
        row = {}
        for channel in channels:
            row.update(_channel_features(window[channel].to_numpy(), channel))
        rows.append(row)
    info = {
        "recording": Path(path).name,
        "total_frames": int(len(df)),
        "warmup_frames_removed": int(start),
        "usable_frames": int(len(df) - start),
        "windows": int(len(rows)),
    }
    return pd.DataFrame(rows), info


def make_legacy_classifier() -> object:
    """The previous soft-voting model, retained only for bundle compatibility."""
    vote = VotingClassifier(
        estimators=[
            (
                "logistic",
                LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=7),
            ),
            (
                "shrinkage_lda",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ],
        voting="soft",
        weights=[1, 1],
    )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(threshold=1e-12),
        RobustScaler(),
        vote,
    )


def make_elastic_logistic() -> object:
    """Drift-tolerant linear classifier selected by grouped validation."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(threshold=1e-12),
        RobustScaler(),
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.3,
            max_iter=10000,
            class_weight="balanced",
            random_state=7,
        ),
    )


def make_banana_freshness_classifier() -> object:
    """Non-linear model that performed best for banana freshness."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(threshold=1e-12),
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=7,
            n_jobs=1,
        ),
    )


def first_timestamp(path: Path) -> pd.Timestamp:
    df = pd.read_csv(path, usecols=["timestamp_utc"], nrows=1)
    return pd.to_datetime(df.iloc[0, 0], utc=True)


def assign_baselines(
    paths: list[Path], max_age_hours: float = 6.0
) -> dict[str, str | None]:
    """Pair each sample with the most recent earlier, non-stale blank."""
    times = {path.name: first_timestamp(path) for path in paths}
    blanks = [path.name for path in paths if parse_labels(path)[0] == "blank"]
    mapping: dict[str, str | None] = {}
    for path in paths:
        food, _, _ = parse_labels(path)
        if food == "blank" or not blanks:
            mapping[path.name] = None
            continue
        prior = [
            name
            for name in blanks
            if times[name] <= times[path.name]
            and (times[path.name] - times[name]).total_seconds()
            <= max_age_hours * 3600.0
        ]
        if prior:
            mapping[path.name] = max(prior, key=lambda name: times[name])
        else:
            mapping[path.name] = None
    return mapping


def build_training_table(
    paths: list[Path],
    channels: list[str],
    config: dict,
    baseline_max_age_hours: float,
) -> tuple[pd.DataFrame, list[dict]]:
    baseline_map = assign_baselines(paths, baseline_max_age_hours)
    frames = []
    quality = []
    for path in paths:
        features, info = extract_windows(path, channels, **config)
        food, condition, state = parse_labels(path)
        features["recording"] = path.name
        features["food"] = food
        features["condition"] = condition
        features["state"] = state
        features["blank_target"] = "blank" if food == "blank" else "food"
        features["baseline_recording"] = baseline_map[path.name]
        frames.append(features)
        info.update(food=food, condition=condition, state=state, baseline_recording=baseline_map[path.name])
        quality.append(info)
    return pd.concat(frames, ignore_index=True), quality


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [col for col in table.columns if col not in META_COLUMNS]


def baseline_corrected_table(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    blank_vectors = {
        name: group[features].median()
        for name, group in table.loc[table.food == "blank"].groupby("recording")
    }
    rows = []
    for recording, group in table.loc[table.food != "blank"].groupby("recording", sort=False):
        baseline_name = group["baseline_recording"].iloc[0]
        if baseline_name not in blank_vectors:
            continue
        corrected = group.copy()
        corrected.loc[:, features] = group[features] - blank_vectors[baseline_name]
        rows.append(corrected)
    if not rows:
        return pd.DataFrame(columns=table.columns)
    return pd.concat(rows, ignore_index=True)


def aggregate_probabilities(model: object, X: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
    probability = model.predict_proba(X).mean(axis=0)
    classes = model.classes_
    index = int(np.argmax(probability))
    return (
        str(classes[index]),
        float(probability[index]),
        {str(label): float(value) for label, value in zip(classes, probability)},
    )


def grouped_cv(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    model_factory: Callable[[], object],
) -> tuple[dict, pd.DataFrame]:
    y = table[target].astype(str).to_numpy()
    groups = table["recording"].astype(str).to_numpy()
    X = table[features]
    rows = []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        train_classes = np.unique(y[train])
        if len(train_classes) < 2:
            continue
        model = model_factory()
        model.fit(X.iloc[train], y[train])
        pred, confidence, probabilities = aggregate_probabilities(model, X.iloc[test])
        rows.append(
            {
                "recording": groups[test][0],
                "true": y[test][0],
                "predicted": pred,
                "confidence": confidence,
                "probabilities": json.dumps(probabilities, sort_keys=True),
            }
        )
    result = pd.DataFrame(rows)
    labels = sorted(set(result["true"]).union(result["predicted"]))
    metrics = {
        "recordings": int(len(result)),
        "labels": labels,
        "accuracy": float(accuracy_score(result.true, result.predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(result.true, result.predicted)),
        "macro_f1": float(f1_score(result.true, result.predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(result.true, result.predicted, labels=labels).tolist(),
    }
    return metrics, result


def _recording_predictions(model: object, table: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    true_labels = []
    predicted_labels = []
    for _, group in table.groupby("recording", sort=False):
        predicted, _, _ = aggregate_probabilities(model, group[features])
        true_labels.append(str(group.iloc[0]["_curve_target"]))
        predicted_labels.append(predicted)
    return true_labels, predicted_labels


def recording_learning_curve(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    task: str,
    model_factory: Callable[[], object],
    repeats: int = 10,
    random_state: int = 7,
) -> pd.DataFrame:
    """Estimate train/validation accuracy versus independent recording count.

    Each repeat reserves one complete CSV per class for validation. Training
    uses an increasing number of the remaining complete CSV recordings, so no
    recording contributes windows to both sides of a split.
    """
    work = table.copy()
    work["_curve_target"] = work[target].astype(str)
    groups_by_class = {
        label: np.asarray(sorted(group["recording"].unique()))
        for label, group in work.groupby("_curve_target")
    }
    too_small = {label: len(groups) for label, groups in groups_by_class.items() if len(groups) < 2}
    if too_small:
        raise ValueError(f"Learning curve requires at least two recordings per class: {too_small}")
    max_train = {label: len(groups) - 1 for label, groups in groups_by_class.items()}
    maximum_stage = max(max_train.values())
    rng = np.random.default_rng(random_state)
    rows = []
    seen_training_sizes = set()
    for stage in range(1, maximum_stage + 1):
        per_class = {label: min(stage, count) for label, count in max_train.items()}
        training_size = int(sum(per_class.values()))
        if training_size in seen_training_sizes:
            continue
        seen_training_sizes.add(training_size)
        train_scores = []
        validation_scores = []
        for _ in range(repeats):
            validation_groups = []
            training_groups = []
            for label, groups in groups_by_class.items():
                validation_group = str(rng.choice(groups))
                validation_groups.append(validation_group)
                candidates = groups[groups != validation_group]
                chosen = rng.choice(candidates, size=per_class[label], replace=False)
                training_groups.extend(str(value) for value in np.atleast_1d(chosen))
            training = work.loc[work.recording.isin(training_groups)].copy()
            validation = work.loc[work.recording.isin(validation_groups)].copy()
            model = model_factory()
            model.fit(training[features], training["_curve_target"])
            train_true, train_pred = _recording_predictions(model, training, features)
            val_true, val_pred = _recording_predictions(model, validation, features)
            train_scores.append(accuracy_score(train_true, train_pred))
            validation_scores.append(accuracy_score(val_true, val_pred))
        rows.append(
            {
                "task": task,
                "training_recordings": training_size,
                "validation_recordings_per_repeat": len(groups_by_class),
                "repeats": repeats,
                "training_accuracy_mean": float(np.mean(train_scores)),
                "training_accuracy_std": float(np.std(train_scores)),
                "validation_accuracy_mean": float(np.mean(validation_scores)),
                "validation_accuracy_std": float(np.std(validation_scores)),
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curve(ax, curve: pd.DataFrame, title: str) -> None:
    x = curve["training_recordings"].to_numpy()
    train_mean = curve["training_accuracy_mean"].to_numpy()
    train_std = curve["training_accuracy_std"].to_numpy()
    validation_mean = curve["validation_accuracy_mean"].to_numpy()
    validation_std = curve["validation_accuracy_std"].to_numpy()
    ax.plot(x, train_mean, marker="o", linewidth=2.2, label="Training accuracy")
    ax.fill_between(
        x,
        np.clip(train_mean - train_std, 0, 1),
        np.clip(train_mean + train_std, 0, 1),
        alpha=0.16,
    )
    ax.plot(x, validation_mean, marker="o", linewidth=2.2, label="Validation accuracy")
    ax.fill_between(
        x,
        np.clip(validation_mean - validation_std, 0, 1),
        np.clip(validation_mean + validation_std, 0, 1),
        alpha=0.16,
    )
    ax.set_title(title)
    ax.set_xlabel("Independent training recordings")
    ax.set_ylabel("Recording-level accuracy")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")


def plot_confusion(ax, metrics: dict, title: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics["labels"]
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted recording label")
    ax.set_ylabel("True recording label")
    ax.set_title(title)


def data_quality_table(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        row = {"recording": path.name, "frames": len(df)}
        for col in TGS_RAW_COLUMNS:
            row[f"{col}_saturation_fraction"] = saturation_fraction(
                pd.to_numeric(df[col], errors="coerce").to_numpy()
            )
        voc = pd.to_numeric(df["svm41_voc_index"], errors="coerce").fillna(0).to_numpy()
        positives = np.flatnonzero(voc > 0)
        row["first_positive_voc_frame"] = int(positives[0]) if len(positives) else None
        row["nox_unique_values_after_45"] = int(pd.Series(df["svm41_nox_index"]).iloc[45:].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def train(
    data_dir: Path,
    output_dir: Path,
    config: dict,
    baseline_max_age_hours: float = 6.0,
) -> None:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    for path in paths:
        parse_labels(path)
    channels, dropped = select_sensor_channels(paths)
    table, quality_info = build_training_table(
        paths, channels, config, baseline_max_age_hours
    )
    features = feature_columns(table)
    nonblank = table.loc[table.food != "blank"].copy()
    banana = nonblank.loc[nonblank.food == "banana"].copy()
    meat = nonblank.loc[nonblank.food == "meat"].copy()

    tasks = [
        ("blank_vs_food", table, "blank_target", make_elastic_logistic),
        ("banana_vs_meat", nonblank, "food", make_elastic_logistic),
        ("banana_freshness", banana, "condition", make_banana_freshness_classifier),
        ("meat_freshness", meat, "condition", make_elastic_logistic),
    ]
    metrics_by_task: dict[str, dict] = {}
    prediction_tables = []
    learning_curves = []
    for task, task_table, target, factory in tasks:
        if task_table[target].nunique() < 2:
            raise ValueError(f"{task} requires two classes, found {task_table[target].unique()}")
        metrics, predictions = grouped_cv(task_table, target, features, factory)
        metrics_by_task[task] = metrics
        predictions.insert(0, "task", task)
        prediction_tables.append(predictions)
        learning_curves.append(
            recording_learning_curve(
                task_table,
                target,
                features,
                task,
                factory,
            )
        )
    learning_curve_table = pd.concat(learning_curves, ignore_index=True)

    blank_model = make_elastic_logistic().fit(table[features], table["blank_target"])
    food_subtype_model = make_elastic_logistic().fit(nonblank[features], nonblank["food"])
    condition_models = {
        "banana": make_banana_freshness_classifier().fit(
            banana[features], banana["condition"]
        ),
        "meat": make_elastic_logistic().fit(meat[features], meat["condition"]),
    }

    # Compatibility fields allow older test scripts to load this bundle.  The
    # food_model alias is binary so those scripts also exclude blank as a class.
    legacy_condition_model = make_elastic_logistic().fit(
        nonblank[features], nonblank["condition"]
    )
    baseline_vectors = {
        name: group[features].median().to_dict()
        for name, group in table.loc[table.food == "blank"].groupby("recording")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 3,
        "sklearn_version": sklearn.__version__,
        "selected_channels": channels,
        "dropped_globally_saturated_channels": dropped,
        "feature_columns": features,
        "feature_config": config,
        "model_architecture": "binary_food_type_with_food_specific_freshness",
        "blank_model": blank_model,
        "food_subtype_model": food_subtype_model,
        "condition_models": condition_models,
        # food_model is deliberately binary for older inference scripts that
        # still read this compatibility key directly.
        "food_model": food_subtype_model,
        "condition_model": legacy_condition_model,
        "condition_baseline_model": None,
        "training_baseline_vectors": baseline_vectors,
        "baseline_max_age_hours": baseline_max_age_hours,
        "baseline_policy": (
            "Baselines older than the configured limit are not paired. "
            "The selected deployed classifiers use absolute features because "
            "they performed best in recording-grouped validation."
        ),
        "output_mapping": {
            "blank": {"freshness": "not_applicable"},
            "banana": {"fresh": "fresh", "not_fresh": "fermented"},
            "meat": {"fresh": "fresh", "not_fresh": "spoiled"},
        },
    }
    joblib.dump(bundle, output_dir / "enose_multitask_model.joblib", compress=3)

    report = {
        "validation_protocol": "Leave-one-recording-out; window probabilities averaged per held-out CSV",
        "model_architecture": (
            "banana-vs-meat elastic logistic -> banana Extra Trees or meat "
            "elastic logistic freshness; blank model is diagnostic only"
        ),
        "recording_count": len(paths),
        "window_count": len(table),
        "selected_channels": channels,
        "dropped_globally_saturated_channels": dropped,
        "tasks": metrics_by_task,
        "baseline_max_age_hours": baseline_max_age_hours,
        "recordings_without_recent_baseline": sorted(
            info["recording"]
            for info in quality_info
            if info["food"] != "blank" and info["baseline_recording"] is None
        ),
        "learning_curves": {
            "definition": "Repeated recording-grouped learning curves; one complete CSV per class held out in each repeat",
            "repeats_per_point": 10,
            "points_file": "learning_curve_points.csv",
            "figure_file": "training_validation_learning_curves.png",
        },
        "limitations": [
            "The number of independent recordings is small; add samples from multiple days and physical specimens.",
            "Sensor saturation cannot be recovered by a classifier and should be fixed at acquisition time.",
            "Confidence values are model scores, not clinically or statistically calibrated probabilities.",
        ],
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output_dir / "recording_level_predictions.csv", index=False
    )

    quality = data_quality_table(paths)
    quality_meta = pd.DataFrame(quality_info)
    quality.merge(quality_meta, on="recording", how="left").to_csv(
        output_dir / "data_quality_report.csv", index=False
    )
    learning_curve_table.to_csv(output_dir / "learning_curve_points.csv", index=False)

    panels = len(tasks)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, (task, _, _, _) in zip(axes, tasks):
        metrics = metrics_by_task[task]
        plot_confusion(
            ax,
            metrics,
            f"{task.replace('_', ' ').title()}\nBalanced accuracy {metrics['balanced_accuracy']:.2f}",
        )
    fig.savefig(output_dir / "validation_confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, curve, (task, _, _, _) in zip(axes, learning_curves, tasks):
        plot_learning_curve(ax, curve, task.replace("_", " ").title())
    fig.suptitle("Training and validation accuracy vs. independent recordings", fontsize=15)
    fig.savefig(output_dir / "training_validation_learning_curves.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, indent=2))


def _align_features(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = features.reindex(columns=columns)
    missing = [col for col in columns if aligned[col].isna().all()]
    if missing:
        raise ValueError(f"Could not construct required model features: {', '.join(missing[:8])}")
    return aligned


def predict(model_path: Path, csv_path: Path, baseline_csv: Path | None) -> dict:
    bundle = joblib.load(model_path)
    features, info = extract_windows(
        csv_path,
        bundle["selected_channels"],
        **bundle["feature_config"],
    )
    X = _align_features(features, bundle["feature_columns"])
    format_version = int(bundle.get("format_version", 1))
    blank_reference_probability = None
    blank_like_warning = False
    if format_version >= 3 and "food_subtype_model" in bundle:
        food, food_confidence, food_probabilities = aggregate_probabilities(
            bundle["food_subtype_model"], X
        )
        if "blank_model" in bundle:
            _, _, blank_probabilities = aggregate_probabilities(
                bundle["blank_model"], X
            )
            blank_reference_probability = float(blank_probabilities.get("blank", 0.0))
            blank_like_warning = blank_reference_probability >= 0.5
    elif format_version >= 2 and "blank_model" in bundle:
        blank_target, blank_confidence, blank_probabilities = aggregate_probabilities(
            bundle["blank_model"], X
        )
        subtype, subtype_confidence, subtype_probabilities = aggregate_probabilities(
            bundle["food_subtype_model"], X
        )
        p_food = float(blank_probabilities.get("food", 0.0))
        food_probabilities = {
            "blank": float(blank_probabilities.get("blank", 1.0 - p_food)),
            "banana": p_food * float(subtype_probabilities.get("banana", 0.0)),
            "meat": p_food * float(subtype_probabilities.get("meat", 0.0)),
        }
        if blank_target == "blank":
            food = "blank"
            food_confidence = food_probabilities["blank"]
        else:
            food = subtype
            food_confidence = food_probabilities[food]
    else:
        food, food_confidence, food_probabilities = aggregate_probabilities(
            bundle["food_model"], X
        )
    result = {
        "csv": csv_path.name,
        "food_type": food,
        "food_confidence": food_confidence,
        "food_probabilities": food_probabilities,
        "blank_reference_probability": blank_reference_probability,
        "blank_like_warning": blank_like_warning,
        "frames_used": info,
        "dropped_training_channels": bundle["dropped_globally_saturated_channels"],
    }
    if food == "blank":
        result.update(
            freshness_level="not_applicable",
            freshness_confidence=food_confidence,
            freshness_probabilities={"not_applicable": 1.0},
            freshness_model="blank_rule",
            baseline_status="not_used_for_blank",
            overall_confidence=food_confidence,
            low_confidence=bool(food_confidence < 0.65),
        )
        return result

    condition_X = X
    if format_version >= 2 and "condition_models" in bundle:
        condition_model = bundle["condition_models"][food]
        model_name = f"{food}_absolute"
    else:
        condition_model = bundle["condition_model"]
        model_name = "absolute"
    baseline_status = "not_provided"
    if baseline_csv is not None:
        baseline_features, _ = extract_windows(
            baseline_csv,
            bundle["selected_channels"],
            **bundle["feature_config"],
        )
        baseline_X = _align_features(baseline_features, bundle["feature_columns"])
        if format_version < 2 and bundle.get("condition_baseline_model") is not None:
            baseline_vector = baseline_X.median()
            condition_X = X - baseline_vector
            condition_model = bundle["condition_baseline_model"]
            model_name = "baseline_corrected"
            baseline_status = "used_by_legacy_baseline_model"
        else:
            # Current grouped validation selected absolute features.  The
            # baseline is accepted and validated, but intentionally does not
            # alter the deployed feature vector.
            baseline_status = "validated_but_absolute_model_selected"
    condition, condition_confidence, condition_probabilities = aggregate_probabilities(
        condition_model, condition_X
    )
    freshness = bundle["output_mapping"][food][condition]
    result.update(
        freshness_level=freshness,
        freshness_condition=condition,
        freshness_confidence=condition_confidence,
        freshness_probabilities=condition_probabilities,
        freshness_model=model_name,
        baseline_status=baseline_status,
        overall_confidence=float(min(food_confidence, condition_confidence)),
        low_confidence=bool(
            min(food_confidence, condition_confidence) < 0.65
            or blank_like_warning
        ),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="train, validate, and save the model")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--minimum-warmup", type=int, default=45)
    train_parser.add_argument("--window-size", type=int, default=60)
    train_parser.add_argument("--step", type=int, default=20)
    train_parser.add_argument("--minimum-window", type=int, default=35)
    train_parser.add_argument(
        "--baseline-max-age-hours",
        type=float,
        default=6.0,
        help="maximum age of an earlier blank used for training-time pairing",
    )
    predict_parser = sub.add_parser("predict", help="classify one CSV recording")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--csv", type=Path, required=True)
    predict_parser.add_argument("--baseline-csv", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        config = {
            "minimum_warmup": args.minimum_warmup,
            "window_size": args.window_size,
            "step": args.step,
            "minimum_window": args.minimum_window,
        }
        train(
            args.data_dir,
            args.output_dir,
            config,
            baseline_max_age_hours=args.baseline_max_age_hours,
        )
    else:
        print(json.dumps(predict(args.model, args.csv, args.baseline_csv), indent=2))


if __name__ == "__main__":
    main()

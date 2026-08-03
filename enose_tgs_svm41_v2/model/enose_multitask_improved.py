#!/usr/bin/env python3
"""Leakage-aware recording-level e-nose food/freshness model.

Architecture
------------
1. Blank-vs-food diagnostic model. Blank is not a deployable food label.
2. Banana-vs-meat food classifier.
3. Food-specific freshness classifiers:
   * banana: fresh vs not_fresh (fermented)
   * meat: fresh vs not_fresh (spoiled)

The important change from the previous model is that overlapping windows from
one CSV are summarized into one recording-level feature vector before model
training and validation. Therefore, a long recording cannot receive more
training weight merely because it creates more overlapping windows.

Labels are inferred from filenames such as ``enose_fresh_banana_1.csv`` and
``enose_spoiled_meat_1.csv``. Unrecognized CSV files are skipped rather than
causing the entire training run to fail.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
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


def discover_training_paths(data_dir: Path) -> tuple[list[Path], list[str]]:
    accepted: list[Path] = []
    skipped: list[str] = []
    for path in sorted(data_dir.glob("*.csv")):
        try:
            parse_labels(path)
        except ValueError:
            skipped.append(path.name)
        else:
            accepted.append(path)
    if not accepted:
        raise FileNotFoundError(
            f"No labelled e-nose CSV files were found in {data_dir}. "
            "Expected filenames containing blank, banana, or meat."
        )
    return accepted, skipped


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
    """Remove fixed startup and the initial zero-valued SVM41 plateau."""
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


def select_sensor_channels(
    paths: list[Path], per_file_threshold: float = 0.98
) -> tuple[list[str], list[str]]:
    """Drop a TGS channel only when it is saturated in every recording."""
    dropped: list[str] = []
    for col in TGS_RAW_COLUMNS:
        fractions = []
        for path in paths:
            frame = pd.read_csv(path, usecols=[col])
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy()
            fractions.append(saturation_fraction(values))
        if fractions and all(value >= per_file_threshold for value in fractions):
            dropped.append(col)
    selected = [col for col in TGS_RAW_COLUMNS if col not in dropped] + SVM41_COLUMNS
    return selected, dropped


def _channel_features(values: np.ndarray, name: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    suffixes = ["median", "q10", "q90", "std", "slope", "delta", "rel_delta"]
    if len(x) < 3:
        result = {f"{name}_{suffix}": math.nan for suffix in suffixes}
        if name.startswith("tgs"):
            result[f"{name}_sat_frac"] = math.nan
        return result

    edge = min(10, len(x))
    first = float(np.median(x[:edge]))
    last = float(np.median(x[-edge:]))
    result = {
        f"{name}_median": float(np.median(x)),
        f"{name}_q10": float(np.quantile(x, 0.10)),
        f"{name}_q90": float(np.quantile(x, 0.90)),
        f"{name}_std": float(np.std(x)),
        f"{name}_slope": float(np.polyfit(np.arange(len(x), dtype=float), x, 1)[0]),
        f"{name}_delta": last - first,
        f"{name}_rel_delta": (last - first) / (abs(first) + 1.0),
    }
    if name.startswith("tgs"):
        result[f"{name}_sat_frac"] = saturation_fraction(x)
    return result


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

    rows: list[dict[str, float]] = []
    for window_start in sorted(set(starts)):
        window = df.iloc[window_start : min(len(df), window_start + window_size)]
        if len(window) < minimum_window:
            continue
        row: dict[str, float] = {}
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


def first_timestamp(path: Path) -> pd.Timestamp:
    frame = pd.read_csv(path, usecols=["timestamp_utc"], nrows=1)
    return pd.to_datetime(frame.iloc[0, 0], utc=True)


def assign_baselines(paths: list[Path], max_age_hours: float = 6.0) -> dict[str, str | None]:
    """Pair a food sample only with a recent earlier blank recording."""
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
            and (times[path.name] - times[name]).total_seconds() <= max_age_hours * 3600.0
        ]
        mapping[path.name] = max(prior, key=lambda name: times[name]) if prior else None
    return mapping


def build_window_table(
    paths: list[Path],
    channels: list[str],
    config: dict,
    baseline_max_age_hours: float,
) -> tuple[pd.DataFrame, list[dict]]:
    baseline_map = assign_baselines(paths, baseline_max_age_hours)
    frames: list[pd.DataFrame] = []
    quality: list[dict] = []
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
        info.update(
            food=food,
            condition=condition,
            state=state,
            baseline_recording=baseline_map[path.name],
        )
        quality.append(info)
    return pd.concat(frames, ignore_index=True), quality


# Backward-compatible name used by earlier visualization scripts.
build_training_table = build_window_table


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column not in META_COLUMNS]


def aggregate_recordings(window_table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Create exactly one robust feature vector per independent CSV recording."""
    rows: list[dict] = []
    for recording, group in window_table.groupby("recording", sort=False):
        row = {feature: float(group[feature].median()) for feature in features}
        for column in META_COLUMNS:
            if column in group:
                row[column] = group.iloc[0][column]
        row["window_count"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def task_feature_columns(features: list[str]) -> dict[str, list[str]]:
    level = [
        feature
        for feature in features
        if feature.endswith("_median") or feature.endswith("_q10") or feature.endswith("_q90")
    ]
    dynamic = [
        feature
        for feature in features
        if feature.endswith("_std")
        or feature.endswith("_slope")
        or feature.endswith("_delta")
        or feature.endswith("_rel_delta")
    ]
    return {
        "blank_vs_food": list(features),
        "banana_vs_meat": level,
        "banana_freshness": dynamic,
        "meat_freshness": level,
    }


def make_recording_classifier(selected_features: list[str], C: float) -> object:
    """Low-variance linear model suitable for the small recording count."""
    selector = ColumnTransformer(
        [("selected", "passthrough", selected_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return make_pipeline(
        selector,
        SimpleImputer(strategy="median"),
        VarianceThreshold(threshold=1e-12),
        RobustScaler(),
        LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=7,
        ),
    )


def model_factories(feature_sets: dict[str, list[str]]) -> dict[str, Callable[[], object]]:
    return {
        "blank_vs_food": lambda: make_recording_classifier(feature_sets["blank_vs_food"], C=0.3),
        "banana_vs_meat": lambda: make_recording_classifier(feature_sets["banana_vs_meat"], C=0.3),
        "banana_freshness": lambda: make_recording_classifier(feature_sets["banana_freshness"], C=0.3),
        "meat_freshness": lambda: make_recording_classifier(feature_sets["meat_freshness"], C=3.0),
    }


def aggregate_probabilities(model: object, X: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
    probabilities = model.predict_proba(X).mean(axis=0)
    classes = np.asarray(model.classes_, dtype=str)
    index = int(np.argmax(probabilities))
    return (
        str(classes[index]),
        float(probabilities[index]),
        {str(label): float(value) for label, value in zip(classes, probabilities)},
    )


def recording_level_cv(
    table: pd.DataFrame,
    target: str,
    all_features: list[str],
    model_factory: Callable[[], object],
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    for test_index in table.index:
        train_mask = table.index != test_index
        y_train = table.loc[train_mask, target].astype(str)
        if y_train.nunique() < 2:
            continue
        model = model_factory()
        model.fit(table.loc[train_mask, all_features], y_train)
        predicted, confidence, probabilities = aggregate_probabilities(
            model, table.loc[[test_index], all_features]
        )
        rows.append(
            {
                "recording": str(table.loc[test_index, "recording"]),
                "true": str(table.loc[test_index, target]),
                "predicted": predicted,
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


def hierarchical_recording_cv(
    recordings: pd.DataFrame,
    all_features: list[str],
    factories: dict[str, Callable[[], object]],
) -> tuple[dict, pd.DataFrame]:
    """Evaluate the complete deployed hierarchy with each food CSV held out once."""
    food_table = recordings.loc[recordings.food != "blank"].reset_index(drop=True)
    rows: list[dict] = []
    for test_index in food_table.index:
        train_mask = food_table.index != test_index
        training = food_table.loc[train_mask]
        test = food_table.loc[[test_index]]

        food_model = factories["banana_vs_meat"]()
        food_model.fit(training[all_features], training["food"])
        predicted_food, food_confidence, food_probabilities = aggregate_probabilities(
            food_model, test[all_features]
        )

        task = f"{predicted_food}_freshness"
        condition_training = training.loc[training.food == predicted_food]
        condition_model = factories[task]()
        condition_model.fit(condition_training[all_features], condition_training["condition"])
        predicted_condition, condition_confidence, condition_probabilities = aggregate_probabilities(
            condition_model, test[all_features]
        )

        predicted_state = (
            f"{predicted_food}_fresh"
            if predicted_condition == "fresh"
            else f"{predicted_food}_{'fermented' if predicted_food == 'banana' else 'spoiled'}"
        )
        rows.append(
            {
                "recording": str(test.iloc[0]["recording"]),
                "true_state": str(test.iloc[0]["state"]),
                "predicted_state": predicted_state,
                "true_food": str(test.iloc[0]["food"]),
                "predicted_food": predicted_food,
                "true_condition": str(test.iloc[0]["condition"]),
                "predicted_condition": predicted_condition,
                "food_confidence": food_confidence,
                "condition_confidence": condition_confidence,
                "overall_confidence": float(min(food_confidence, condition_confidence)),
                "food_probabilities": json.dumps(food_probabilities, sort_keys=True),
                "condition_probabilities": json.dumps(condition_probabilities, sort_keys=True),
            }
        )

    result = pd.DataFrame(rows)
    labels = ["banana_fresh", "banana_fermented", "meat_fresh", "meat_spoiled"]
    metrics = {
        "recordings": int(len(result)),
        "labels": labels,
        "accuracy": float(accuracy_score(result.true_state, result.predicted_state)),
        "balanced_accuracy": float(
            balanced_accuracy_score(result.true_state, result.predicted_state)
        ),
        "macro_f1": float(
            f1_score(result.true_state, result.predicted_state, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            result.true_state, result.predicted_state, labels=labels
        ).tolist(),
    }
    return metrics, result


def make_legacy_elastic() -> object:
    """Previous window-level linear model, used only for comparison visuals."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(threshold=1e-12),
        RobustScaler(),
        LogisticRegression(
            C=0.3,
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=7,
        ),
    )


def make_legacy_banana_extra_trees() -> object:
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


def grouped_window_cv(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    model_factory: Callable[[], object],
) -> tuple[dict, pd.DataFrame]:
    """Previous validation method retained for direct old-vs-new comparison."""
    y = table[target].astype(str).to_numpy()
    groups = table["recording"].astype(str).to_numpy()
    X = table[features]
    rows = []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[train])) < 2:
            continue
        model = model_factory()
        model.fit(X.iloc[train], y[train])
        predicted, confidence, probabilities = aggregate_probabilities(model, X.iloc[test])
        rows.append(
            {
                "recording": groups[test][0],
                "true": y[test][0],
                "predicted": predicted,
                "confidence": confidence,
                "probabilities": json.dumps(probabilities, sort_keys=True),
            }
        )
    result = pd.DataFrame(rows)
    labels = sorted(set(result.true).union(result.predicted))
    metrics = {
        "recordings": int(len(result)),
        "labels": labels,
        "accuracy": float(accuracy_score(result.true, result.predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(result.true, result.predicted)),
        "macro_f1": float(f1_score(result.true, result.predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(result.true, result.predicted, labels=labels).tolist(),
    }
    return metrics, result


def data_quality_table(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        row: dict[str, object] = {"recording": path.name, "frames": int(len(df))}
        for col in TGS_RAW_COLUMNS:
            row[f"{col}_saturation_fraction"] = saturation_fraction(
                pd.to_numeric(df[col], errors="coerce").to_numpy()
            )
        voc = pd.to_numeric(df["svm41_voc_index"], errors="coerce").fillna(0).to_numpy()
        positives = np.flatnonzero(voc > 0)
        row["first_positive_voc_frame"] = int(positives[0]) if len(positives) else None
        row["nox_unique_values_after_45"] = int(
            pd.to_numeric(df["svm41_nox_index"], errors="coerce").iloc[45:].nunique()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_deployed_models(
    recordings: pd.DataFrame,
    all_features: list[str],
    factories: dict[str, Callable[[], object]],
) -> dict[str, object]:
    nonblank = recordings.loc[recordings.food != "blank"]
    banana = nonblank.loc[nonblank.food == "banana"]
    meat = nonblank.loc[nonblank.food == "meat"]

    blank_model = factories["blank_vs_food"]()
    blank_model.fit(recordings[all_features], recordings["blank_target"])

    food_model = factories["banana_vs_meat"]()
    food_model.fit(nonblank[all_features], nonblank["food"])

    banana_model = factories["banana_freshness"]()
    banana_model.fit(banana[all_features], banana["condition"])

    meat_model = factories["meat_freshness"]()
    meat_model.fit(meat[all_features], meat["condition"])

    return {
        "blank_model": blank_model,
        "food_subtype_model": food_model,
        "condition_models": {"banana": banana_model, "meat": meat_model},
    }


def train(
    data_dir: Path,
    output_dir: Path,
    config: dict,
    baseline_max_age_hours: float = 6.0,
) -> None:
    paths, skipped = discover_training_paths(data_dir)
    channels, dropped = select_sensor_channels(paths)
    window_table, quality_info = build_window_table(
        paths, channels, config, baseline_max_age_hours
    )
    features = feature_columns(window_table)
    recordings = aggregate_recordings(window_table, features)
    feature_sets = task_feature_columns(features)
    factories = model_factories(feature_sets)

    task_specs = {
        "blank_vs_food": (recordings, "blank_target"),
        "banana_vs_meat": (recordings.loc[recordings.food != "blank"].reset_index(drop=True), "food"),
        "banana_freshness": (recordings.loc[recordings.food == "banana"].reset_index(drop=True), "condition"),
        "meat_freshness": (recordings.loc[recordings.food == "meat"].reset_index(drop=True), "condition"),
    }

    task_metrics: dict[str, dict] = {}
    prediction_tables: list[pd.DataFrame] = []
    for task, (task_table, target) in task_specs.items():
        metrics, predictions = recording_level_cv(
            task_table, target, features, factories[task]
        )
        task_metrics[task] = metrics
        predictions.insert(0, "task", task)
        prediction_tables.append(predictions)

    final_metrics, final_predictions = hierarchical_recording_cv(
        recordings, features, factories
    )
    deployed = fit_deployed_models(recordings, features, factories)

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 4,
        "sklearn_version": sklearn.__version__,
        "selected_channels": channels,
        "dropped_globally_saturated_channels": dropped,
        "feature_columns": features,
        "feature_config": config,
        "feature_aggregation": "median_across_windows_per_recording",
        "task_feature_columns": feature_sets,
        "model_architecture": "recording_level_binary_food_with_food_specific_freshness",
        "blank_model": deployed["blank_model"],
        "food_subtype_model": deployed["food_subtype_model"],
        "condition_models": deployed["condition_models"],
        # Compatibility aliases used by some earlier test scripts.
        "food_model": deployed["food_subtype_model"],
        "condition_model": deployed["condition_models"]["banana"],
        "condition_baseline_model": None,
        "baseline_max_age_hours": baseline_max_age_hours,
        "output_mapping": {
            "banana": {"fresh": "fresh", "not_fresh": "fermented"},
            "meat": {"fresh": "fresh", "not_fresh": "spoiled"},
        },
    }
    joblib.dump(bundle, output_dir / "enose_multitask_model.joblib", compress=3)

    report = {
        "validation_protocol": (
            "Leave-one-recording-out on one median feature vector per CSV; "
            "no overlapping window from a held-out CSV is used for training"
        ),
        "model_architecture": bundle["model_architecture"],
        "recording_count": int(len(recordings)),
        "food_recording_count": int((recordings.food != "blank").sum()),
        "window_count_before_recording_aggregation": int(len(window_table)),
        "feature_aggregation": bundle["feature_aggregation"],
        "selected_channels": channels,
        "dropped_globally_saturated_channels": dropped,
        "task_feature_counts": {key: len(value) for key, value in feature_sets.items()},
        "tasks": task_metrics,
        "final_hierarchical_state": final_metrics,
        "skipped_unrecognized_csv_files": skipped,
        "limitations": [
            "The current results use only a small number of independent recordings and are not an external test-set estimate.",
            "Only two blank and two spoiled-meat recordings are available, so perfect leave-one-recording-out scores for those classes are fragile.",
            "The task-specific feature subsets and regularization values were selected using the current dataset and should be rechecked after more data are collected.",
            "Confidence values are model scores and are not calibrated safety probabilities.",
        ],
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output_dir / "task_recording_predictions.csv", index=False
    )
    final_predictions.to_csv(
        output_dir / "final_hierarchical_predictions.csv", index=False
    )
    recordings.to_csv(output_dir / "recording_feature_table.csv", index=False)

    quality = data_quality_table(paths)
    quality_meta = pd.DataFrame(quality_info)
    quality.merge(quality_meta, on="recording", how="left").to_csv(
        output_dir / "data_quality_report.csv", index=False
    )
    print(json.dumps(report, indent=2))


def _align_features(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = features.reindex(columns=columns)
    missing = [column for column in columns if aligned[column].isna().all()]
    if missing:
        raise ValueError(f"Could not construct required model features: {', '.join(missing[:8])}")
    return aligned


def predict(model_path: Path, csv_path: Path, baseline_csv: Path | None = None) -> dict:
    bundle = joblib.load(model_path)
    windows, info = extract_windows(
        csv_path,
        bundle["selected_channels"],
        **bundle["feature_config"],
    )
    windows = _align_features(windows, bundle["feature_columns"])
    recording_X = windows.median(axis=0).to_frame().T

    _, _, blank_probabilities = aggregate_probabilities(bundle["blank_model"], recording_X)
    blank_probability = float(blank_probabilities.get("blank", 0.0))
    blank_like_warning = bool(blank_probability >= 0.5)

    food, food_confidence, food_probabilities = aggregate_probabilities(
        bundle["food_subtype_model"], recording_X
    )
    condition, condition_confidence, condition_probabilities = aggregate_probabilities(
        bundle["condition_models"][food], recording_X
    )
    freshness = bundle["output_mapping"][food][condition]

    baseline_status = "not_provided"
    if baseline_csv is not None:
        # The current recording-level models were selected using absolute
        # features. The baseline is checked for readability but does not alter
        # the deployed vector.
        baseline_windows, _ = extract_windows(
            baseline_csv,
            bundle["selected_channels"],
            **bundle["feature_config"],
        )
        _align_features(baseline_windows, bundle["feature_columns"])
        baseline_status = "validated_but_absolute_model_selected"

    overall_confidence = float(min(food_confidence, condition_confidence))
    return {
        "csv": csv_path.name,
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
        "low_confidence": bool(overall_confidence < 0.65 or blank_like_warning),
        "feature_aggregation": bundle.get("feature_aggregation"),
        "frames_used": info,
        "baseline_status": baseline_status,
        "dropped_training_channels": bundle["dropped_globally_saturated_channels"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="train and validate the recording-level model")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--minimum-warmup", type=int, default=45)
    train_parser.add_argument("--window-size", type=int, default=60)
    train_parser.add_argument("--step", type=int, default=20)
    train_parser.add_argument("--minimum-window", type=int, default=35)
    train_parser.add_argument("--baseline-max-age-hours", type=float, default=6.0)

    predict_parser = sub.add_parser("predict", help="classify one complete CSV recording")
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

#!/usr/bin/env python3
"""Train and run an improved e-nose food/freshness classifier.

Classification logic
--------------------
1. Blank recordings are used only as baseline references.
2. Food model: banana vs meat.
3. Global condition model: fresh vs not_fresh.
4. Optional food-specific condition models:
      banana -> fresh vs fermented
      meat   -> fresh vs spoiled
5. Direct state model:
      banana_fresh / banana_fermented / meat_fresh / meat_spoiled
6. The hierarchical and direct state probabilities are fused. The strategy
   and fusion weight are selected by leave-one-recording-out validation.
7. Window probabilities are aggregated with the median to reduce the effect
   of a noisy or transient window.

The command-line interface remains compatible with the previous model:

python enose_multitask_improved.py train \
    --data-dir ./training_csv \
    --output-dir ./artifacts

python enose_multitask_improved.py predict \
    --model ./artifacts/enose_multitask_model.joblib \
    --csv ./test.csv \
    [--baseline-csv ./blank.csv]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
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
]
STATE_ORDER = [
    "banana_fermented",
    "banana_fresh",
    "meat_fresh",
    "meat_spoiled",
]
SATURATION_LOW = 3
SATURATION_HIGH = 4094
EPSILON = 1e-9


def parse_labels(path: str | Path) -> tuple[str, str, str]:
    name = Path(path).name.lower()

    if "blank" in name:
        return "blank", "not_applicable", "blank"

    if "banana" in name:
        food = "banana"
    elif "meat" in name:
        food = "meat"
    else:
        raise ValueError(
            f"Cannot infer banana/meat label from filename: {Path(path).name}"
        )

    if "fresh" in name:
        condition = "fresh"
    elif "fermented" in name or "spoiled" in name:
        condition = "not_fresh"
    else:
        raise ValueError(
            f"Cannot infer freshness label from filename: {Path(path).name}"
        )

    if food == "banana":
        state = "banana_fresh" if condition == "fresh" else "banana_fermented"
    else:
        state = "meat_fresh" if condition == "fresh" else "meat_spoiled"
    return food, condition, state


def _as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_and_clean(
    path: str | Path,
    channels: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(TGS_RAW_COLUMNS + SVM41_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"{Path(path).name} is missing columns: {', '.join(missing)}"
        )

    valid = pd.Series(True, index=df.index)
    if "ads7828_ok" in df.columns:
        valid &= _as_true(df["ads7828_ok"])
    if "svm41_ok" in df.columns:
        valid &= _as_true(df["svm41_ok"])
    df = df.loc[valid].reset_index(drop=True)

    use = list(channels) if channels is not None else TGS_RAW_COLUMNS + SVM41_COLUMNS
    for col in use:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit=3, limit_direction="both")
    return df


def adaptive_warmup_end(
    df: pd.DataFrame,
    minimum: int = 45,
    minimum_remaining: int = 35,
) -> int:
    """Remove fixed startup and the initial SVM41 VOC zero plateau."""
    voc = (
        pd.to_numeric(df["svm41_voc_index"], errors="coerce")
        .fillna(0)
        .to_numpy()
    )
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
    return float(
        np.mean((values <= SATURATION_LOW) | (values >= SATURATION_HIGH))
    )


def select_sensor_channels(
    paths: list[Path],
    per_file_threshold: float = 0.98,
) -> tuple[list[str], list[str]]:
    """Drop a TGS channel only when it is saturated in every recording."""
    dropped: list[str] = []
    for column in TGS_RAW_COLUMNS:
        fractions: list[float] = []
        for path in paths:
            frame = pd.read_csv(path, usecols=[column])
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            fractions.append(saturation_fraction(values))
        if fractions and all(value >= per_file_threshold for value in fractions):
            dropped.append(column)

    selected = [
        column for column in TGS_RAW_COLUMNS if column not in dropped
    ] + SVM41_COLUMNS
    return selected, dropped


def _channel_features(values: np.ndarray, name: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    suffixes = [
        "median",
        "q10",
        "q90",
        "iqr",
        "std",
        "slope",
        "delta",
        "rel_delta",
    ]
    if len(x) < 3:
        output = {f"{name}_{suffix}": math.nan for suffix in suffixes}
        if name.startswith("tgs"):
            output[f"{name}_sat_frac"] = math.nan
        return output

    edge = min(10, len(x))
    first = float(np.median(x[:edge]))
    last = float(np.median(x[-edge:]))
    output = {
        f"{name}_median": float(np.median(x)),
        f"{name}_q10": float(np.quantile(x, 0.10)),
        f"{name}_q90": float(np.quantile(x, 0.90)),
        f"{name}_iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        f"{name}_std": float(np.std(x)),
        f"{name}_slope": float(
            np.polyfit(np.arange(len(x), dtype=float), x, 1)[0]
        ),
        f"{name}_delta": last - first,
        f"{name}_rel_delta": (last - first) / (abs(first) + 1.0),
    }
    if name.startswith("tgs"):
        output[f"{name}_sat_frac"] = saturation_fraction(x)
    return output


def _sensor_pattern_features(
    window: pd.DataFrame,
    channels: list[str],
) -> dict[str, float]:
    """Create drift-resistant cross-sensor fingerprint features."""
    tgs_channels = [
        channel for channel in channels if channel.startswith("tgs")
    ]
    medians: dict[str, float] = {}
    for channel in tgs_channels:
        value = pd.to_numeric(window[channel], errors="coerce").median()
        if pd.notna(value):
            medians[channel] = float(value)

    if len(medians) < 2:
        return {}

    log_values = {
        channel: float(np.log1p(max(value, 0.0)))
        for channel, value in medians.items()
    }
    center = float(np.median(list(log_values.values())))
    output: dict[str, float] = {}

    for channel, value in log_values.items():
        output[f"{channel}_log_pattern_centered"] = value - center

    keys = sorted(log_values)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            output[f"logratio_{left}__{right}"] = (
                log_values[left] - log_values[right]
            )

    if "svm41_voc_index" in window.columns:
        voc = pd.to_numeric(
            window["svm41_voc_index"], errors="coerce"
        ).median()
        if pd.notna(voc):
            output["voc_log_to_tgs_logmean"] = float(
                np.log1p(max(float(voc), 0.0))
                - np.mean(list(log_values.values()))
            )
    return output


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
    usable = len(df) - start
    if usable < minimum_window:
        raise ValueError(
            f"{Path(path).name}: only {usable} usable frames after warm-up; "
            f"need at least {minimum_window}"
        )

    last_start = max(start, len(df) - window_size)
    starts = list(
        range(
            start,
            max(start + 1, len(df) - window_size + 1),
            step,
        )
    )
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    rows: list[dict[str, float]] = []
    for window_start in sorted(set(starts)):
        window = df.iloc[
            window_start : min(len(df), window_start + window_size)
        ]
        if len(window) < minimum_window:
            continue

        row: dict[str, float] = {}
        for channel in channels:
            row.update(
                _channel_features(window[channel].to_numpy(), channel)
            )
        row.update(_sensor_pattern_features(window, channels))
        rows.append(row)

    info = {
        "recording": Path(path).name,
        "total_frames": int(len(df)),
        "warmup_frames_removed": int(start),
        "usable_frames": int(usable),
        "windows": int(len(rows)),
    }
    return pd.DataFrame(rows), info


def first_timestamp(path: Path) -> pd.Timestamp:
    frame = pd.read_csv(path, usecols=["timestamp_utc"], nrows=1)
    return pd.to_datetime(frame.iloc[0, 0], utc=True)


def assign_baselines(paths: list[Path]) -> dict[str, str | None]:
    times = {path.name: first_timestamp(path) for path in paths}
    blanks = [
        path.name for path in paths if parse_labels(path)[0] == "blank"
    ]
    mapping: dict[str, str | None] = {}

    for path in paths:
        food, _, _ = parse_labels(path)
        if food == "blank" or not blanks:
            mapping[path.name] = None
            continue

        prior = [
            name for name in blanks if times[name] <= times[path.name]
        ]
        if prior:
            mapping[path.name] = max(prior, key=lambda name: times[name])
        else:
            mapping[path.name] = min(
                blanks,
                key=lambda name: abs(times[name] - times[path.name]),
            )
    return mapping


def build_training_table(
    paths: list[Path],
    channels: list[str],
    config: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    baseline_map = assign_baselines(paths)
    frames: list[pd.DataFrame] = []
    quality: list[dict] = []

    for path in paths:
        features, info = extract_windows(path, channels, **config)
        food, condition, state = parse_labels(path)
        features["recording"] = path.name
        features["food"] = food
        features["condition"] = condition
        features["state"] = state
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


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [
        column for column in table.columns if column not in META_COLUMNS
    ]


def baseline_corrected_table(
    table: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    blank_vectors = {
        name: group[features].median()
        for name, group in table.loc[
            table.food == "blank"
        ].groupby("recording")
    }

    rows: list[pd.DataFrame] = []
    nonblank = table.loc[table.food != "blank"]
    for recording, group in nonblank.groupby("recording", sort=False):
        baseline_name = group["baseline_recording"].iloc[0]
        if baseline_name not in blank_vectors:
            continue
        corrected = group.copy()
        corrected.loc[:, features] = (
            group[features] - blank_vectors[baseline_name]
        )
        rows.append(corrected)

    if not rows:
        return pd.DataFrame(columns=table.columns)
    return pd.concat(rows, ignore_index=True)


def make_classifier(include_lda: bool = True) -> object:
    estimators = [
        (
            "logistic",
            LogisticRegression(
                C=0.3,
                max_iter=5000,
                class_weight="balanced",
                random_state=7,
            ),
        )
    ]
    if include_lda:
        estimators.append(
            (
                "shrinkage_lda",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                ),
            )
        )
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[1] * len(estimators),
    )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(threshold=1e-12),
        RobustScaler(),
        ensemble,
    )


def fit_classifier(X: pd.DataFrame, y: pd.Series) -> object:
    counts = pd.Series(y).astype(str).value_counts()
    include_lda = bool(not counts.empty and counts.min() >= 2)
    return make_classifier(include_lda=include_lda).fit(X, y)


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return np.full(len(values), 1.0 / len(values))
    return values / total


def aggregate_probabilities(
    model: object,
    X: pd.DataFrame,
    method: str = "median",
) -> tuple[str, float, dict[str, float], float, float]:
    window_probabilities = np.asarray(model.predict_proba(X), dtype=float)

    if method == "mean":
        probability = window_probabilities.mean(axis=0)
    elif method == "geometric":
        probability = np.exp(
            np.log(np.clip(window_probabilities, 1e-6, 1.0)).mean(axis=0)
        )
    else:
        probability = np.median(window_probabilities, axis=0)

    probability = _normalize(probability)
    classes = np.asarray(model.classes_, dtype=str)
    index = int(np.argmax(probability))
    label = str(classes[index])

    per_window = classes[np.argmax(window_probabilities, axis=1)]
    agreement = float(np.mean(per_window == label))
    sorted_probability = np.sort(probability)
    margin = float(
        sorted_probability[-1] - sorted_probability[-2]
        if len(sorted_probability) > 1
        else sorted_probability[-1]
    )
    return (
        label,
        float(probability[index]),
        {
            str(class_name): float(value)
            for class_name, value in zip(classes, probability)
        },
        agreement,
        margin,
    )


def _fit_model_set(
    table: pd.DataFrame,
    features: list[str],
) -> dict:
    nonblank = table.loc[table.food != "blank"].copy()
    if nonblank.empty:
        raise ValueError("No non-blank recordings are available for training")

    required_states = set(STATE_ORDER)
    present_states = set(nonblank["state"].astype(str))
    missing_states = sorted(required_states.difference(present_states))
    if missing_states:
        raise ValueError(
            "Training requires all four food states. Missing: "
            + ", ".join(missing_states)
        )

    models = {
        "food_model": fit_classifier(
            nonblank[features], nonblank["food"]
        ),
        "condition_model": fit_classifier(
            nonblank[features], nonblank["condition"]
        ),
        "state_model": fit_classifier(
            nonblank[features], nonblank["state"]
        ),
        "condition_models_by_food": {},
    }

    for food in ("banana", "meat"):
        subset = nonblank.loc[nonblank.food == food]
        if subset["condition"].nunique() == 2:
            models["condition_models_by_food"][food] = (
                fit_classifier(
                    subset[features],
                    subset["condition"],
                )
            )
    return models


def _dictionary_to_state_vector(
    probabilities: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [probabilities.get(state, 0.0) for state in STATE_ORDER],
        dtype=float,
    )


def _state_to_outputs(
    state_probabilities: np.ndarray,
) -> dict:
    state_probabilities = _normalize(state_probabilities)
    state_index = int(np.argmax(state_probabilities))
    state = STATE_ORDER[state_index]

    food_probabilities = {
        "banana": float(
            state_probabilities[0] + state_probabilities[1]
        ),
        "meat": float(
            state_probabilities[2] + state_probabilities[3]
        ),
    }
    condition_probabilities = {
        "fresh": float(
            state_probabilities[1] + state_probabilities[2]
        ),
        "not_fresh": float(
            state_probabilities[0] + state_probabilities[3]
        ),
    }

    food = max(food_probabilities, key=food_probabilities.get)
    condition = max(
        condition_probabilities,
        key=condition_probabilities.get,
    )
    freshness_level = (
        "fresh"
        if state in {"banana_fresh", "meat_fresh"}
        else "fermented"
        if state == "banana_fermented"
        else "spoiled"
    )

    sorted_probability = np.sort(state_probabilities)
    margin = float(sorted_probability[-1] - sorted_probability[-2])
    return {
        "state": state,
        "state_confidence": float(state_probabilities[state_index]),
        "state_probabilities": {
            state_name: float(value)
            for state_name, value in zip(
                STATE_ORDER,
                state_probabilities,
            )
        },
        "food_type": food,
        "food_confidence": float(food_probabilities[food]),
        "food_probabilities": food_probabilities,
        "freshness_condition": condition,
        "freshness_level": freshness_level,
        "freshness_confidence": float(
            condition_probabilities[condition]
        ),
        "freshness_probabilities": condition_probabilities,
        "probability_margin": margin,
    }


def _score_model_set(
    models: dict,
    X: pd.DataFrame,
    strategy: dict,
    aggregation: str,
) -> dict:
    _, _, food_probabilities, food_agreement, _ = (
        aggregate_probabilities(
            models["food_model"],
            X,
            aggregation,
        )
    )
    _, _, global_condition_probabilities, global_agreement, _ = (
        aggregate_probabilities(
            models["condition_model"],
            X,
            aggregation,
        )
    )
    _, _, direct_probabilities, direct_agreement, _ = (
        aggregate_probabilities(
            models["state_model"],
            X,
            aggregation,
        )
    )

    condition_source = strategy["condition_source"]
    conditional: dict[str, dict[str, float]] = {}

    for food in ("banana", "meat"):
        food_model = models["condition_models_by_food"].get(food)
        if condition_source == "food_specific" and food_model is not None:
            _, _, probabilities, _, _ = aggregate_probabilities(
                food_model,
                X,
                aggregation,
            )
            conditional[food] = probabilities
        else:
            conditional[food] = global_condition_probabilities

    hierarchical = np.asarray(
        [
            food_probabilities.get("banana", 0.0)
            * conditional["banana"].get("not_fresh", 0.0),
            food_probabilities.get("banana", 0.0)
            * conditional["banana"].get("fresh", 0.0),
            food_probabilities.get("meat", 0.0)
            * conditional["meat"].get("fresh", 0.0),
            food_probabilities.get("meat", 0.0)
            * conditional["meat"].get("not_fresh", 0.0),
        ],
        dtype=float,
    )
    hierarchical = _normalize(hierarchical)
    direct = _normalize(
        _dictionary_to_state_vector(direct_probabilities)
    )

    direct_weight = float(strategy["direct_weight"])
    fused = _normalize(
        (1.0 - direct_weight) * hierarchical
        + direct_weight * direct
    )
    output = _state_to_outputs(fused)
    output.update(
        classification_strategy={
            "condition_source": condition_source,
            "direct_weight": direct_weight,
            "aggregation": aggregation,
        },
        component_agreement={
            "food_windows": food_agreement,
            "condition_windows": global_agreement,
            "direct_state_windows": direct_agreement,
        },
        hierarchical_state_probabilities={
            state: float(value)
            for state, value in zip(STATE_ORDER, hierarchical)
        },
        direct_state_probabilities={
            state: float(value)
            for state, value in zip(STATE_ORDER, direct)
        },
    )
    output["window_agreement"] = float(
        min(food_agreement, global_agreement, direct_agreement)
    )
    return output


def _candidate_strategies() -> list[dict]:
    candidates: list[dict] = []
    for source in ("global", "food_specific"):
        for weight in (0.0, 0.25, 0.50, 0.75):
            candidates.append(
                {
                    "condition_source": source,
                    "direct_weight": weight,
                }
            )
    return candidates


def _strategy_id(strategy: dict) -> str:
    return (
        f"{strategy['condition_source']}"
        f"_direct_{strategy['direct_weight']:.2f}"
    )


def select_strategy_by_grouped_cv(
    table: pd.DataFrame,
    features: list[str],
    aggregation: str = "median",
) -> tuple[dict, dict, pd.DataFrame]:
    nonblank = table.loc[table.food != "blank"].copy()
    groups = nonblank["recording"].astype(str).to_numpy()
    y = nonblank["state"].astype(str).to_numpy()

    if len(np.unique(groups)) < 5:
        raise ValueError(
            "At least five independent recordings are required "
            "for strategy selection"
        )

    rows: list[dict] = []
    candidates = _candidate_strategies()

    for train_index, test_index in LeaveOneGroupOut().split(
        nonblank,
        y,
        groups,
    ):
        training = nonblank.iloc[train_index]
        test = nonblank.iloc[test_index]

        if set(training["state"]) != set(STATE_ORDER):
            continue

        model_set = _fit_model_set(training, features)
        for strategy in candidates:
            output = _score_model_set(
                model_set,
                test[features],
                strategy,
                aggregation,
            )
            rows.append(
                {
                    "strategy": _strategy_id(strategy),
                    "recording": str(test["recording"].iloc[0]),
                    "true": str(test["state"].iloc[0]),
                    "predicted": output["state"],
                    "confidence": output["state_confidence"],
                    "margin": output["probability_margin"],
                    "window_agreement": output["window_agreement"],
                }
            )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError(
            "Grouped validation could not create a valid training fold. "
            "Collect at least two recordings for every state."
        )

    summaries: list[dict] = []
    for strategy_name, group in predictions.groupby("strategy"):
        summaries.append(
            {
                "strategy": strategy_name,
                "recordings": int(len(group)),
                "accuracy": float(
                    accuracy_score(group.true, group.predicted)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        group.true,
                        group.predicted,
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        group.true,
                        group.predicted,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    summary = pd.DataFrame(summaries)

    # Prefer the simplest robust blend when validation scores tie.
    preference = {
        "global_direct_0.25": 8,
        "global_direct_0.50": 7,
        "global_direct_0.00": 6,
        "global_direct_0.75": 5,
        "food_specific_direct_0.25": 4,
        "food_specific_direct_0.50": 3,
        "food_specific_direct_0.00": 2,
        "food_specific_direct_0.75": 1,
    }
    summary["preference"] = summary["strategy"].map(
        preference
    ).fillna(0)
    summary = summary.sort_values(
        [
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "preference",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected_name = str(summary.iloc[0]["strategy"])
    selected_row = next(
        candidate
        for candidate in candidates
        if _strategy_id(candidate) == selected_name
    )
    selected_predictions = predictions.loc[
        predictions.strategy == selected_name
    ].copy()

    labels = STATE_ORDER
    metrics = {
        "recordings": int(len(selected_predictions)),
        "labels": labels,
        "accuracy": float(
            accuracy_score(
                selected_predictions.true,
                selected_predictions.predicted,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                selected_predictions.true,
                selected_predictions.predicted,
            )
        ),
        "macro_f1": float(
            f1_score(
                selected_predictions.true,
                selected_predictions.predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            selected_predictions.true,
            selected_predictions.predicted,
            labels=labels,
        ).tolist(),
        "strategy_comparison": summary.drop(
            columns=["preference"]
        ).to_dict(orient="records"),
    }
    return selected_row, metrics, selected_predictions


def grouped_cv(
    table: pd.DataFrame,
    target: str,
    features: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Compatibility helper for the previous visualization script."""
    work = table.copy()
    y = work[target].astype(str).to_numpy()
    groups = work["recording"].astype(str).to_numpy()
    rows: list[dict] = []

    for train_index, test_index in LeaveOneGroupOut().split(
        work,
        y,
        groups,
    ):
        if len(np.unique(y[train_index])) < 2:
            continue
        model = fit_classifier(
            work.iloc[train_index][features],
            pd.Series(y[train_index]),
        )
        predicted, confidence, probabilities, agreement, margin = (
            aggregate_probabilities(
                model,
                work.iloc[test_index][features],
                "median",
            )
        )
        rows.append(
            {
                "recording": groups[test_index][0],
                "true": y[test_index][0],
                "predicted": predicted,
                "confidence": confidence,
                "probabilities": json.dumps(
                    probabilities,
                    sort_keys=True,
                ),
                "window_agreement": agreement,
                "margin": margin,
            }
        )

    result = pd.DataFrame(rows)
    labels = sorted(
        set(result["true"]).union(result["predicted"])
    )
    metrics = {
        "recordings": int(len(result)),
        "labels": labels,
        "accuracy": float(
            accuracy_score(result.true, result.predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                result.true,
                result.predicted,
            )
        ),
        "macro_f1": float(
            f1_score(
                result.true,
                result.predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            result.true,
            result.predicted,
            labels=labels,
        ).tolist(),
    }
    return metrics, result


def data_quality_table(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        frame = pd.read_csv(path)
        row = {
            "recording": path.name,
            "frames": int(len(frame)),
        }
        for column in TGS_RAW_COLUMNS:
            row[f"{column}_saturation_fraction"] = (
                saturation_fraction(
                    pd.to_numeric(
                        frame[column],
                        errors="coerce",
                    ).to_numpy()
                )
            )

        voc = (
            pd.to_numeric(
                frame["svm41_voc_index"],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
        )
        positive = np.flatnonzero(voc > 0)
        row["first_positive_voc_frame"] = (
            int(positive[0]) if len(positive) else None
        )
        row["nox_unique_values_after_45"] = int(
            pd.Series(frame["svm41_nox_index"])
            .iloc[45:]
            .nunique()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _align_features(
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    aligned = features.reindex(columns=columns)
    missing = [
        column
        for column in columns
        if aligned[column].isna().all()
    ]
    if missing:
        raise ValueError(
            "Could not construct required model features: "
            + ", ".join(missing[:8])
        )
    return aligned


def train(
    data_dir: Path,
    output_dir: Path,
    config: dict,
) -> None:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}"
        )
    for path in paths:
        parse_labels(path)

    channels, dropped = select_sensor_channels(paths)
    table, quality_info = build_training_table(
        paths,
        channels,
        config,
    )
    features = feature_columns(table)
    nonblank = table.loc[table.food != "blank"].copy()

    strategy, metrics, predictions = (
        select_strategy_by_grouped_cv(
            nonblank,
            features,
            aggregation="median",
        )
    )
    model_set = _fit_model_set(nonblank, features)

    corrected = baseline_corrected_table(table, features)
    baseline_strategy = None
    baseline_metrics = None
    baseline_predictions = pd.DataFrame()
    baseline_model_set = None

    if (
        not corrected.empty
        and corrected["recording"].nunique() >= 5
        and set(corrected["state"]) == set(STATE_ORDER)
    ):
        (
            baseline_strategy,
            baseline_metrics,
            baseline_predictions,
        ) = select_strategy_by_grouped_cv(
            corrected,
            features,
            aggregation="median",
        )
        baseline_model_set = _fit_model_set(
            corrected,
            features,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 2,
        "sklearn_version": sklearn.__version__,
        "selected_channels": channels,
        "dropped_globally_saturated_channels": dropped,
        "feature_columns": features,
        "feature_config": config,
        "aggregation": "median",
        "classification_strategy": strategy,
        "models": model_set,
        "baseline_classification_strategy": baseline_strategy,
        "baseline_models": baseline_model_set,
        "blank_is_reference_only": True,
        "state_order": STATE_ORDER,
    }
    joblib.dump(
        bundle,
        output_dir / "enose_multitask_model.joblib",
        compress=3,
    )

    report = {
        "validation_protocol": (
            "Leave-one-recording-out validation; no windows from "
            "a held-out CSV are used for training"
        ),
        "classification_logic": {
            "blank_handling": (
                "Blank recordings are baseline references only"
            ),
            "food_stage": "banana vs meat",
            "condition_stage": (
                "global fresh/not_fresh or food-specific condition "
                "model, selected by grouped validation"
            ),
            "direct_stage": (
                "four-state classifier used as a cross-check"
            ),
            "fusion": (
                "hierarchical and direct state probabilities are "
                "combined with a validation-selected weight"
            ),
            "window_aggregation": "median probability",
        },
        "recording_count_total": int(len(paths)),
        "recording_count_nonblank": int(
            nonblank["recording"].nunique()
        ),
        "window_count_nonblank": int(len(nonblank)),
        "selected_channels": channels,
        "dropped_globally_saturated_channels": dropped,
        "selected_strategy": strategy,
        "four_state_validation": metrics,
        "baseline_selected_strategy": baseline_strategy,
        "baseline_four_state_validation": baseline_metrics,
        "limitations": [
            (
                "Validation performance is recording-level but the "
                "dataset is still small."
            ),
            (
                "A high confidence score does not replace additional "
                "independent recordings collected on different days."
            ),
            (
                "A blank-like test sample is still forced into banana "
                "or meat because blank is intentionally not an output class."
            ),
        ],
    }
    (
        output_dir / "evaluation_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    predictions = predictions.copy()
    predictions.insert(0, "task", "four_state_absolute")
    prediction_tables = [predictions]
    if not baseline_predictions.empty:
        baseline_predictions = baseline_predictions.copy()
        baseline_predictions.insert(
            0,
            "task",
            "four_state_baseline_corrected",
        )
        prediction_tables.append(baseline_predictions)
    pd.concat(
        prediction_tables,
        ignore_index=True,
    ).to_csv(
        output_dir / "recording_level_predictions.csv",
        index=False,
    )

    quality = data_quality_table(paths)
    quality_meta = pd.DataFrame(quality_info)
    quality.merge(
        quality_meta,
        on="recording",
        how="left",
    ).to_csv(
        output_dir / "data_quality_report.csv",
        index=False,
    )

    print(json.dumps(report, indent=2))


def predict(
    model_path: Path,
    csv_path: Path,
    baseline_csv: Path | None,
) -> dict:
    bundle = joblib.load(model_path)
    features, info = extract_windows(
        csv_path,
        bundle["selected_channels"],
        **bundle["feature_config"],
    )
    X = _align_features(
        features,
        bundle["feature_columns"],
    )

    model_set = bundle["models"]
    strategy = bundle["classification_strategy"]
    model_name = "absolute"

    if (
        baseline_csv is not None
        and bundle.get("baseline_models") is not None
        and bundle.get(
            "baseline_classification_strategy"
        ) is not None
    ):
        baseline_features, _ = extract_windows(
            baseline_csv,
            bundle["selected_channels"],
            **bundle["feature_config"],
        )
        baseline_X = _align_features(
            baseline_features,
            bundle["feature_columns"],
        )
        baseline_vector = baseline_X.median()
        X = X - baseline_vector
        model_set = bundle["baseline_models"]
        strategy = bundle[
            "baseline_classification_strategy"
        ]
        model_name = "baseline_corrected"

    output = _score_model_set(
        model_set,
        X,
        strategy,
        bundle.get("aggregation", "median"),
    )

    overall_confidence = float(
        min(
            output["food_confidence"],
            output["freshness_confidence"],
            output["state_confidence"],
        )
    )
    low_confidence = bool(
        output["state_confidence"] < 0.65
        or output["probability_margin"] < 0.15
        or output["window_agreement"] < 0.60
    )

    result = {
        "csv": csv_path.name,
        "food_type": output["food_type"],
        "food_confidence": output["food_confidence"],
        "food_probabilities": output["food_probabilities"],
        "freshness_level": output["freshness_level"],
        "freshness_condition": output[
            "freshness_condition"
        ],
        "freshness_confidence": output[
            "freshness_confidence"
        ],
        "freshness_probabilities": output[
            "freshness_probabilities"
        ],
        "freshness_model": model_name,
        "state": output["state"],
        "state_confidence": output["state_confidence"],
        "state_probabilities": output[
            "state_probabilities"
        ],
        "overall_confidence": overall_confidence,
        "low_confidence": low_confidence,
        "probability_margin": output[
            "probability_margin"
        ],
        "window_agreement": output["window_agreement"],
        "classification_strategy": output[
            "classification_strategy"
        ],
        "component_agreement": output[
            "component_agreement"
        ],
        "hierarchical_state_probabilities": output[
            "hierarchical_state_probabilities"
        ],
        "direct_state_probabilities": output[
            "direct_state_probabilities"
        ],
        "frames_used": info,
        "dropped_training_channels": bundle[
            "dropped_globally_saturated_channels"
        ],
        "blank_is_reference_only": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="train, validate, and save the model",
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    train_parser.add_argument(
        "--minimum-warmup",
        type=int,
        default=45,
    )
    train_parser.add_argument(
        "--window-size",
        type=int,
        default=60,
    )
    train_parser.add_argument(
        "--step",
        type=int,
        default=20,
    )
    train_parser.add_argument(
        "--minimum-window",
        type=int,
        default=35,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="classify one CSV recording",
    )
    predict_parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    predict_parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )
    predict_parser.add_argument(
        "--baseline-csv",
        type=Path,
    )

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
        )
    else:
        print(
            json.dumps(
                predict(
                    args.model,
                    args.csv,
                    args.baseline_csv,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

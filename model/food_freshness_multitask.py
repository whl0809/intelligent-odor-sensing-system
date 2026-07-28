#!/usr/bin/env python3
"""
Food type and freshness classification from four e-nose CSV files.

Run directly in PyCharm. No command-line arguments are required.

Input files expected in the same folder as this script:
    enose_fresh_banana.csv
    enose_fermented_banana (2).csv
    enose_fresh_meat.csv
    enose_baseline(2).csv

Three classification tasks are trained:
    1. Combined class:
       fresh_banana / fermented_banana / fresh_meat
    2. Food type:
       banana / meat
    3. Freshness:
       fresh / fermented

Training/validation split:
    - Each original food recording is split chronologically.
    - First 70%: training
    - Last 30%: validation
    - A 5-row purge region is removed around the boundary.
    - Sliding windows are created only after the split, so no raw row appears
      in both training and validation.

Important limitation:
    There is currently only one recording for each sample condition.
    Therefore, validation is a temporal holdout from the same recording,
    not a fully independent experimental-session validation.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

# Save figures reliably when running in PyCharm.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILES = {
    "fresh_banana": BASE_DIR / "enose_fresh_banana.csv",
    "fermented_banana": BASE_DIR / "enose_fermented_banana (2).csv",
    "fresh_meat": BASE_DIR / "enose_fresh_meat.csv",
}

BASELINE_FILE = BASE_DIR / "enose_baseline(2).csv"

OUTPUT_DIR = BASE_DIR / "classification_outputs"
PREPARED_DIR = OUTPUT_DIR / "prepared_dataset"
RAW_TRAIN_DIR = PREPARED_DIR / "raw_training"
RAW_VALIDATION_DIR = PREPARED_DIR / "raw_validation"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

RANDOM_STATE = 42

TRAIN_FRACTION = 0.70
PURGE_ROWS = 5

# At approximately 1 sample/second:
# each window covers 20 seconds and starts every 5 seconds.
WINDOW_SIZE = 20
WINDOW_STRIDE = 5

TGS_FULL_SCALE = 4095
SATURATION_DROP_RATE = 0.95
FEATURE_VARIANCE_THRESHOLD = 1e-12

SENSOR_COLUMNS = {
    "tgs2620": "tgs2620_voltage_v",
    "tgs2610": "tgs2610_voltage_v",
    "tgs2611": "tgs2611_voltage_v",
    "tgs2600": "tgs2600_voltage_v",
    "tgs2602": "tgs2602_voltage_v",
    "tgs2603": "tgs2603_voltage_v",
    "nh3": "nh3_diff_voltage_v",
    "h2s": "h2s_diff_voltage_v",
}

TGS_RAW_COLUMNS = {
    "tgs2620": "tgs2620_raw",
    "tgs2610": "tgs2610_raw",
    "tgs2611": "tgs2611_raw",
    "tgs2600": "tgs2600_raw",
    "tgs2602": "tgs2602_raw",
    "tgs2603": "tgs2603_raw",
}

STATUS_COLUMNS = [
    "ads7828_ok",
    "nh3_ok",
    "h2s_ok",
]

TASKS = {
    "combined_class": {
        "target_column": "combined_class",
        "display_name": "Combined food/freshness class",
    },
    "food_type": {
        "target_column": "food_type",
        "display_name": "Food type",
    },
    "freshness": {
        "target_column": "freshness",
        "display_name": "Freshness",
    },
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class SampleDefinition:
    name: str
    path: Path
    food_type: str
    freshness: str
    combined_class: str


SAMPLES = [
    SampleDefinition(
        name="fresh_banana",
        path=INPUT_FILES["fresh_banana"],
        food_type="banana",
        freshness="fresh",
        combined_class="fresh_banana",
    ),
    SampleDefinition(
        name="fermented_banana",
        path=INPUT_FILES["fermented_banana"],
        food_type="banana",
        freshness="fermented",
        combined_class="fermented_banana",
    ),
    SampleDefinition(
        name="fresh_meat",
        path=INPUT_FILES["fresh_meat"],
        food_type="meat",
        freshness="fresh",
        combined_class="fresh_meat",
    ),
]


# =============================================================================
# General utilities
# =============================================================================

def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes", "ok"})


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find input file:\n{path}\n\n"
            "Place all four CSV files in the same folder as this script."
        )

    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(f"CSV file is empty: {path.name}")

    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        frame = frame.sort_values("timestamp_utc")

    return frame.reset_index(drop=True)


def validate_required_columns(frame: pd.DataFrame, path: Path) -> None:
    required = list(SENSOR_COLUMNS.values())

    missing = [
        column for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing required sensor columns:\n"
            + "\n".join(missing)
        )


def filter_valid_rows(
    raw_frame: pd.DataFrame,
    selected_channels: dict[str, str],
) -> pd.DataFrame:
    valid = pd.Series(True, index=raw_frame.index)

    for status_column in STATUS_COLUMNS:
        if status_column in raw_frame.columns:
            valid &= parse_bool_series(raw_frame[status_column])

    cleaned = pd.DataFrame(index=raw_frame.index)

    for sensor_name, source_column in selected_channels.items():
        cleaned[sensor_name] = pd.to_numeric(
            raw_frame[source_column],
            errors="coerce",
        )

    valid &= cleaned.notna().all(axis=1)

    # Keep timing information when available.
    if "timestamp_utc" in raw_frame.columns:
        cleaned.insert(0, "timestamp_utc", raw_frame["timestamp_utc"])

    if "elapsed_s" in raw_frame.columns:
        cleaned.insert(
            1 if "timestamp_utc" in cleaned.columns else 0,
            "elapsed_s",
            pd.to_numeric(raw_frame["elapsed_s"], errors="coerce"),
        )

    return cleaned.loc[valid].reset_index(drop=True)


# =============================================================================
# Channel quality and baseline normalization
# =============================================================================

def determine_selected_channels(
    food_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, str], dict[str, Any]]:
    saturation_rates: dict[str, float] = {}
    dropped_channels: dict[str, str] = {}

    for sensor_name, raw_column in TGS_RAW_COLUMNS.items():
        values: list[np.ndarray] = []

        for frame in food_frames.values():
            if raw_column in frame.columns:
                numeric = pd.to_numeric(
                    frame[raw_column],
                    errors="coerce",
                ).dropna()

                if len(numeric):
                    values.append(numeric.to_numpy(dtype=float))

        if values:
            combined = np.concatenate(values)
            rate = float(
                np.mean(combined >= TGS_FULL_SCALE)
            )
        else:
            rate = 0.0

        saturation_rates[sensor_name] = rate

        if rate >= SATURATION_DROP_RATE:
            dropped_channels[sensor_name] = (
                f"ADC saturation rate {rate:.1%}"
            )

    selected = {
        sensor_name: source_column
        for sensor_name, source_column in SENSOR_COLUMNS.items()
        if sensor_name not in dropped_channels
    }

    # Drop any channel with effectively no variation across all food files.
    for sensor_name, source_column in list(selected.items()):
        values = []

        for frame in food_frames.values():
            numeric = pd.to_numeric(
                frame[source_column],
                errors="coerce",
            ).dropna()

            if len(numeric):
                values.append(numeric.to_numpy(dtype=float))

        if not values:
            dropped_channels[sensor_name] = "no valid numeric values"
            selected.pop(sensor_name)
            continue

        variance = float(np.var(np.concatenate(values)))

        if not math.isfinite(variance) or variance <= 1e-18:
            dropped_channels[sensor_name] = (
                f"near-zero variance {variance:.3g}"
            )
            selected.pop(sensor_name)

    if len(selected) < 2:
        raise ValueError(
            "Fewer than two usable sensor channels remain."
        )

    report = {
        "selected_channels": selected,
        "dropped_channels": dropped_channels,
        "tgs_saturation_rates": saturation_rates,
    }

    return selected, report


def calculate_baseline(
    baseline_frame: pd.DataFrame,
    selected_channels: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    clean_baseline = filter_valid_rows(
        baseline_frame,
        selected_channels,
    )

    sensor_names = list(selected_channels)

    if len(clean_baseline) < 2:
        raise ValueError(
            "Not enough valid baseline rows for normalization."
        )

    means: dict[str, float] = {}
    scales: dict[str, float] = {}

    for sensor_name in sensor_names:
        values = clean_baseline[
            sensor_name
        ].to_numpy(dtype=float)

        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=0))

        # Avoid unstable division for nearly constant baseline channels.
        scale_floor = max(
            abs(mean_value) * 1e-3,
            1e-6,
        )

        means[sensor_name] = mean_value
        scales[sensor_name] = max(std_value, scale_floor)

    return means, scales, clean_baseline


def normalize_sensor_frame(
    frame: pd.DataFrame,
    baseline_means: dict[str, float],
    baseline_scales: dict[str, float],
) -> pd.DataFrame:
    normalized = frame.copy()

    metadata_columns = {
        "timestamp_utc",
        "elapsed_s",
    }

    for column in normalized.columns:
        if column in metadata_columns:
            continue

        normalized[column] = (
            normalized[column].astype(float)
            - baseline_means[column]
        ) / baseline_scales[column]

    return normalized


# =============================================================================
# Chronological train/validation split
# =============================================================================

def split_one_session(
    frame: pd.DataFrame,
    sample: SampleDefinition,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n_rows = len(frame)
    split_index = int(n_rows * TRAIN_FRACTION)

    train_end = split_index - PURGE_ROWS
    validation_start = split_index + PURGE_ROWS

    training = frame.iloc[:train_end].copy()
    validation = frame.iloc[validation_start:].copy()

    if len(training) < WINDOW_SIZE:
        raise ValueError(
            f"{sample.name}: training section has only "
            f"{len(training)} rows."
        )

    if len(validation) < WINDOW_SIZE:
        raise ValueError(
            f"{sample.name}: validation section has only "
            f"{len(validation)} rows."
        )

    for output, split_name in [
        (training, "training"),
        (validation, "validation"),
    ]:
        output["source_session"] = sample.name
        output["split"] = split_name
        output["food_type"] = sample.food_type
        output["freshness"] = sample.freshness
        output["combined_class"] = sample.combined_class

    summary = {
        "source_session": sample.name,
        "original_rows": n_rows,
        "training_rows": len(training),
        "purged_rows": validation_start - train_end,
        "validation_rows": len(validation),
        "split_index": split_index,
    }

    return training, validation, summary


def save_prepared_raw_splits(
    train_frames: dict[str, pd.DataFrame],
    validation_frames: dict[str, pd.DataFrame],
) -> None:
    RAW_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    for name, frame in train_frames.items():
        frame.to_csv(
            RAW_TRAIN_DIR / f"{name}.csv",
            index=False,
        )

    for name, frame in validation_frames.items():
        frame.to_csv(
            RAW_VALIDATION_DIR / f"{name}.csv",
            index=False,
        )


# =============================================================================
# Feature extraction
# =============================================================================

def safe_auc(values: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values))

    return float(np.trapz(values))


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if len(first) < 2:
        return 0.0

    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0

    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else 0.0


def extract_features_from_window(
    window: pd.DataFrame,
    sensor_names: list[str],
) -> dict[str, float]:
    features: dict[str, float] = {}
    x_axis = np.arange(len(window), dtype=float)

    for sensor_name in sensor_names:
        values = window[sensor_name].to_numpy(dtype=float)
        prefix = safe_name(sensor_name)

        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_std"] = float(np.std(values, ddof=0))
        features[f"{prefix}_min"] = float(np.min(values))
        features[f"{prefix}_max"] = float(np.max(values))
        features[f"{prefix}_median"] = float(np.median(values))
        features[f"{prefix}_q25"] = float(
            np.quantile(values, 0.25)
        )
        features[f"{prefix}_q75"] = float(
            np.quantile(values, 0.75)
        )
        features[f"{prefix}_range"] = float(
            np.max(values) - np.min(values)
        )
        features[f"{prefix}_last_minus_first"] = float(
            values[-1] - values[0]
        )
        features[f"{prefix}_auc"] = safe_auc(values)
        features[f"{prefix}_slope"] = float(
            np.polyfit(x_axis, values, 1)[0]
        )

    # Pairwise sensor correlations.
    for first_index in range(len(sensor_names)):
        for second_index in range(
            first_index + 1,
            len(sensor_names),
        ):
            first_name = sensor_names[first_index]
            second_name = sensor_names[second_index]

            features[
                f"corr_{safe_name(first_name)}_"
                f"{safe_name(second_name)}"
            ] = safe_correlation(
                window[first_name].to_numpy(dtype=float),
                window[second_name].to_numpy(dtype=float),
            )

    tgs_names = [
        name for name in sensor_names
        if name.startswith("tgs")
    ]

    if tgs_names:
        tgs_window_means = np.array(
            [
                float(window[name].mean())
                for name in tgs_names
            ]
        )

        features["tgs_array_mean"] = float(
            np.mean(tgs_window_means)
        )
        features["tgs_array_std_between_sensors"] = float(
            np.std(tgs_window_means, ddof=0)
        )
        features["tgs_array_range_between_sensors"] = float(
            np.max(tgs_window_means)
            - np.min(tgs_window_means)
        )

    if "nh3" in sensor_names and "h2s" in sensor_names:
        features["nh3_minus_h2s_mean"] = float(
            window["nh3"].mean()
            - window["h2s"].mean()
        )

    return features


def frames_to_windows(
    split_frames: dict[str, pd.DataFrame],
    sensor_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for session_name, frame in split_frames.items():
        split_name = str(frame["split"].iloc[0])
        food_type = str(frame["food_type"].iloc[0])
        freshness = str(frame["freshness"].iloc[0])
        combined_class = str(
            frame["combined_class"].iloc[0]
        )

        for start in range(
            0,
            len(frame) - WINDOW_SIZE + 1,
            WINDOW_STRIDE,
        ):
            stop = start + WINDOW_SIZE
            window = frame.iloc[start:stop]

            feature_row: dict[str, Any] = (
                extract_features_from_window(
                    window,
                    sensor_names,
                )
            )

            feature_row.update(
                {
                    "source_session": session_name,
                    "split": split_name,
                    "window_start_row": start,
                    "window_end_row": stop - 1,
                    "food_type": food_type,
                    "freshness": freshness,
                    "combined_class": combined_class,
                }
            )

            rows.append(feature_row)

    if not rows:
        raise ValueError("No sliding windows were created.")

    return pd.DataFrame(rows)


# =============================================================================
# PLS-DA
# =============================================================================

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)

        target = np.zeros(
            (len(y), len(self.classes_)),
            dtype=float,
        )

        for row_index, label in enumerate(y):
            class_index = int(
                np.where(self.classes_ == label)[0][0]
            )
            target[row_index, class_index] = 1.0

        n_components = min(
            self.n_components,
            X.shape[1],
            max(1, len(self.classes_) - 1),
        )

        self.model_ = PLSRegression(
            n_components=max(1, n_components)
        )
        self.model_.fit(X, target)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.model_.predict(X)
        indices = np.argmax(scores, axis=1)

        return self.classes_[indices]


# =============================================================================
# Models
# =============================================================================

def scaled_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "variance",
                VarianceThreshold(
                    threshold=FEATURE_VARIANCE_THRESHOLD
                ),
            ),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def unscaled_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "variance",
                VarianceThreshold(
                    threshold=FEATURE_VARIANCE_THRESHOLD
                ),
            ),
            ("model", model),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": scaled_pipeline(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "SVM_Linear": scaled_pipeline(
            SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "SVM_RBF": scaled_pipeline(
            SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "KNN": scaled_pipeline(
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
            )
        ),
        "LDA": scaled_pipeline(
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            )
        ),
        "PLS_DA": scaled_pipeline(
            PLSDAClassifier(n_components=2)
        ),
        "GaussianNB": unscaled_pipeline(
            GaussianNB()
        ),
        "DecisionTree": unscaled_pipeline(
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "RandomForest": unscaled_pipeline(
            RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "ExtraTrees": unscaled_pipeline(
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "GradientBoosting": unscaled_pipeline(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        ),
        "MLP_ANN": scaled_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-3,
                max_iter=1500,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )
        ),
    }


# =============================================================================
# Training and evaluation
# =============================================================================

NON_FEATURE_COLUMNS = {
    "source_session",
    "split",
    "window_start_row",
    "window_end_row",
    "food_type",
    "freshness",
    "combined_class",
}


def evaluate_one_task(
    task_name: str,
    task_config: dict[str, str],
    train_windows: pd.DataFrame,
    validation_windows: pd.DataFrame,
) -> dict[str, Any]:
    target_column = task_config["target_column"]
    display_name = task_config["display_name"]

    feature_columns = [
        column
        for column in train_windows.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    x_train = train_windows[feature_columns]
    x_validation = validation_windows[feature_columns]

    y_train = train_windows[target_column].astype(str)
    y_validation = validation_windows[
        target_column
    ].astype(str)

    train_classes = sorted(y_train.unique())
    validation_classes = sorted(y_validation.unique())

    if train_classes != validation_classes:
        raise ValueError(
            f"{display_name}: training classes {train_classes} do not "
            f"match validation classes {validation_classes}."
        )

    results_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}

    print("\n" + "=" * 80)
    print(f"Task: {display_name}")
    print("=" * 80)
    print(f"Training windows:   {len(x_train)}")
    print(f"Validation windows: {len(x_validation)}")
    print(f"Classes:            {train_classes}")
    print(f"Feature count:      {len(feature_columns)}")

    for model_name, model_template in build_models().items():
        print(f"\nTraining {model_name}...")

        try:
            model = clone(model_template)
            model.fit(x_train, y_train)

            predicted = model.predict(x_validation)

            row = {
                "task": task_name,
                "model": model_name,
                "accuracy": accuracy_score(
                    y_validation,
                    predicted,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_validation,
                    predicted,
                ),
                "macro_f1": f1_score(
                    y_validation,
                    predicted,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    y_validation,
                    predicted,
                    average="weighted",
                    zero_division=0,
                ),
            }

            results_rows.append(row)
            fitted_models[model_name] = model
            predictions[model_name] = predicted

            print(
                f"  Accuracy: {row['accuracy']:.4f} | "
                f"Macro-F1: {row['macro_f1']:.4f}"
            )

        except Exception as error:
            print(f"  Failed: {error}")

    if not results_rows:
        raise RuntimeError(
            f"All models failed for task: {task_name}"
        )

    results = pd.DataFrame(results_rows).sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = str(results.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_prediction = predictions[best_model_name]

    task_model_path = MODEL_DIR / f"{task_name}_best_model.joblib"
    task_results_path = RESULTS_DIR / f"{task_name}_results.csv"
    task_report_path = RESULTS_DIR / (
        f"{task_name}_classification_report.txt"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "task_name": task_name,
            "display_name": display_name,
            "model_name": best_model_name,
            "model": best_model,
            "feature_columns": feature_columns,
            "classes": train_classes,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
        },
        task_model_path,
    )

    results.to_csv(task_results_path, index=False)

    report_text = classification_report(
        y_validation,
        best_prediction,
        labels=train_classes,
        zero_division=0,
    )
    task_report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print("\nModel ranking:")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(report_text)

    plot_model_results(
        task_name,
        display_name,
        results,
    )
    plot_task_confusion_matrix(
        task_name,
        display_name,
        y_validation.to_numpy(),
        best_prediction,
        train_classes,
        best_model_name,
    )

    return {
        "task": task_name,
        "display_name": display_name,
        "best_model": best_model_name,
        "best_metrics": results.iloc[0].to_dict(),
        "model_path": str(task_model_path.resolve()),
        "results_path": str(task_results_path.resolve()),
        "classes": train_classes,
    }


# =============================================================================
# Visualizations
# =============================================================================

def save_figure(filename: str) -> Path:
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VISUALIZATION_DIR / filename

    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {path.resolve()}")
    return path


def plot_model_results(
    task_name: str,
    display_name: str,
    results: pd.DataFrame,
) -> None:
    ordered = results.sort_values(
        "macro_f1",
        ascending=True,
    )

    positions = np.arange(len(ordered))
    width = 0.38

    plt.figure(figsize=(11, 7))
    plt.barh(
        positions - width / 2,
        ordered["accuracy"],
        height=width,
        label="Validation accuracy",
    )
    plt.barh(
        positions + width / 2,
        ordered["macro_f1"],
        height=width,
        label="Validation macro-F1",
    )

    plt.yticks(positions, ordered["model"])
    plt.xlim(0.0, 1.05)
    plt.xlabel("Score")
    plt.title(f"{display_name}: model comparison")
    plt.legend()
    plt.grid(axis="x", alpha=0.25)

    save_figure(f"{task_name}_01_model_comparison.png")


def plot_task_confusion_matrix(
    task_name: str,
    display_name: str,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    classes: list[str],
    model_name: str,
) -> None:
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=classes,
    )

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix)
    plt.colorbar(label="Validation window count")

    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(
        f"{display_name}: {model_name}\nvalidation confusion matrix"
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )

    save_figure(f"{task_name}_02_confusion_matrix.png")


def plot_selected_sensor_signals(
    normalized_frames: dict[str, pd.DataFrame],
    sensor_names: list[str],
) -> None:
    for session_name, frame in normalized_frames.items():
        plt.figure(figsize=(12, 7))

        for sensor_name in sensor_names:
            plt.plot(
                np.arange(len(frame)),
                frame[sensor_name],
                label=sensor_name,
            )

        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Sample")
        plt.ylabel("Air-baseline-normalized response")
        plt.title(
            f"{session_name.replace('_', ' ').title()}: "
            "selected sensor signals"
        )
        plt.legend(ncol=2)
        plt.grid(alpha=0.25)

        save_figure(
            f"signals_{session_name}.png"
        )


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    np.random.seed(RANDOM_STATE)

    print("Food type and freshness classification")
    print("=" * 80)
    print(f"Program folder: {BASE_DIR.resolve()}")
    print(f"Output folder:  {OUTPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    food_raw_frames: dict[str, pd.DataFrame] = {}

    for sample in SAMPLES:
        frame = read_csv(sample.path)
        validate_required_columns(frame, sample.path)
        food_raw_frames[sample.name] = frame

    baseline_raw = read_csv(BASELINE_FILE)
    validate_required_columns(
        baseline_raw,
        BASELINE_FILE,
    )

    selected_channels, channel_report = (
        determine_selected_channels(food_raw_frames)
    )

    print("\nSelected sensor channels")
    print("-" * 80)
    for sensor_name, source_column in (
        selected_channels.items()
    ):
        print(f"{sensor_name:>10} <- {source_column}")

    if channel_report["dropped_channels"]:
        print("\nAutomatically excluded channels")
        print("-" * 80)
        for sensor_name, reason in channel_report[
            "dropped_channels"
        ].items():
            print(f"{sensor_name:>10}: {reason}")

    (
        baseline_means,
        baseline_scales,
        clean_baseline,
    ) = calculate_baseline(
        baseline_raw,
        selected_channels,
    )

    sensor_names = list(selected_channels)

    normalized_full_frames: dict[str, pd.DataFrame] = {}
    train_frames: dict[str, pd.DataFrame] = {}
    validation_frames: dict[str, pd.DataFrame] = {}
    split_summaries: list[dict[str, Any]] = []

    for sample in SAMPLES:
        clean = filter_valid_rows(
            food_raw_frames[sample.name],
            selected_channels,
        )

        normalized = normalize_sensor_frame(
            clean,
            baseline_means,
            baseline_scales,
        )
        normalized_full_frames[
            sample.name
        ] = normalized.copy()

        training, validation, summary = (
            split_one_session(
                normalized,
                sample,
            )
        )

        train_frames[sample.name] = training
        validation_frames[sample.name] = validation
        split_summaries.append(summary)

    save_prepared_raw_splits(
        train_frames,
        validation_frames,
    )

    split_summary_frame = pd.DataFrame(
        split_summaries
    )
    split_summary_path = PREPARED_DIR / (
        "split_summary.csv"
    )
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    split_summary_frame.to_csv(
        split_summary_path,
        index=False,
    )

    train_windows = frames_to_windows(
        train_frames,
        sensor_names,
    )
    validation_windows = frames_to_windows(
        validation_frames,
        sensor_names,
    )

    train_feature_path = PREPARED_DIR / (
        "training_window_features.csv"
    )
    validation_feature_path = PREPARED_DIR / (
        "validation_window_features.csv"
    )

    train_windows.to_csv(
        train_feature_path,
        index=False,
    )
    validation_windows.to_csv(
        validation_feature_path,
        index=False,
    )

    metadata = {
        "input_files": {
            sample.name: str(sample.path.resolve())
            for sample in SAMPLES
        },
        "baseline_file": str(BASELINE_FILE.resolve()),
        "train_fraction": TRAIN_FRACTION,
        "purge_rows": PURGE_ROWS,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "selected_channels": selected_channels,
        "channel_quality": channel_report,
        "baseline_means": baseline_means,
        "baseline_scales": baseline_scales,
        "split_summary": split_summaries,
        "training_windows": len(train_windows),
        "validation_windows": len(
            validation_windows
        ),
    }

    metadata_path = OUTPUT_DIR / (
        "dataset_and_preprocessing_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nTraining/validation split")
    print("-" * 80)
    print(split_summary_frame.to_string(index=False))
    print(
        f"\nTraining feature windows: "
        f"{len(train_windows)}"
    )
    print(
        f"Validation feature windows: "
        f"{len(validation_windows)}"
    )

    task_summaries: list[dict[str, Any]] = []

    for task_name, task_config in TASKS.items():
        summary = evaluate_one_task(
            task_name,
            task_config,
            train_windows,
            validation_windows,
        )
        task_summaries.append(summary)

    overall_summary_path = OUTPUT_DIR / (
        "best_models_summary.json"
    )
    overall_summary_path.write_text(
        json.dumps(task_summaries, indent=2),
        encoding="utf-8",
    )

    plot_selected_sensor_signals(
        normalized_full_frames,
        sensor_names,
    )

    print("\n" + "=" * 80)
    print("Completed successfully")
    print("=" * 80)
    print(f"Prepared dataset: {PREPARED_DIR.resolve()}")
    print(f"Models:           {MODEL_DIR.resolve()}")
    print(f"Results:          {RESULTS_DIR.resolve()}")
    print(f"Figures:          {VISUALIZATION_DIR.resolve()}")
    print(f"Metadata:         {metadata_path.resolve()}")

    print(
        "\nValidation limitation: each condition currently comes from only "
        "one original recording. The code prevents row/window overlap, but "
        "the validation portion still belongs to the same measurement "
        "session as its training portion. Collect multiple independent "
        "recordings per condition for a stronger session-level validation."
    )

    print(
        "\nFreshness limitation: fermented samples currently include only "
        "banana, while meat is available only as fresh. The freshness model "
        "may therefore learn some banana-specific differences. Add "
        "fermented meat and more food types to reduce this confounding."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR")
        print("=" * 80)
        print(error)
        raise

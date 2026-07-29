#!/usr/bin/env python3
"""Train leakage-aware hierarchical e-nose classifiers.

The available dataset contains one continuous recording for each labeled
condition.  Each recording is therefore split chronologically: the first 70 %
is used for training and the final 30 % for validation, with a purge gap at the
boundary.  Sliding windows are created only after the split, so a raw sample
cannot occur in both subsets.  This is a temporal holdout, not an independent
cross-day validation.

Outputs are written below ``classification_outputs/`` and are consumed by
``test_model.py``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "classification_outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "visualizations"
PREPARED_DIR = OUTPUT_DIR / "prepared_dataset"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
WINDOW_SIZE = 20
WINDOW_STRIDE = 5
PURGE_ROWS = WINDOW_SIZE - 1
# TGS2600 is intentionally excluded because it is known to remain at the
# ADC full-scale value in the available hardware configuration.  Saturation
# in any other sensor is recorded for quality reporting but does not remove
# that channel or its clipped readings from model input.
EXCLUDED_CHANNELS = {
    "tgs2600": "manually excluded: known persistent ADC saturation",
}

CHANNELS = {
    "tgs2620": "tgs2620_voltage_v",
    "tgs2610": "tgs2610_voltage_v",
    "tgs2611": "tgs2611_voltage_v",
    "tgs2600": "tgs2600_voltage_v",
    "tgs2602": "tgs2602_voltage_v",
    "tgs2603": "tgs2603_voltage_v",
    "nh3": "nh3_diff_voltage_v",
    "h2s": "h2s_diff_voltage_v",
}
RAW_TGS = {name: f"{name}_raw" for name in CHANNELS if name.startswith("tgs")}
STATUS_COLUMNS = ("ads7828_ok", "nh3_ok", "h2s_ok")


@dataclass(frozen=True)
class Recording:
    name: str
    filename: str
    food_group: str
    fruit_freshness: str
    meat_freshness: str
    odor_state: str


RECORDINGS = (
    Recording("baseline", "enose_baseline.csv", "blank", "not_applicable", "not_applicable", "blank"),
    Recording("fresh_banana", "enose_fresh_banana.csv", "fruit", "fresh", "not_applicable", "fresh_banana"),
    Recording("fermented_banana", "enose_fermented_banana.csv", "fruit", "fermented", "not_applicable", "fermented_banana"),
    Recording("fresh_meat", "enose_fresh_meat.csv", "meat", "not_applicable", "fresh", "fresh_meat"),
    Recording("spoiled_meat", "enose_spoiled_meat.csv", "meat", "not_applicable", "spoiled", "spoiled_meat"),
)

TASKS = {
    "food_group": {"target": "food_group", "subset": None},
    "fruit_freshness": {"target": "fruit_freshness", "subset": "fruit"},
    "meat_freshness": {"target": "meat_freshness", "subset": "meat"},
    "odor_state": {"target": "odor_state", "subset": None},
}

META_COLUMNS = {
    "recording", "split", "window_start", "window_end",
    "food_group", "fruit_freshness", "meat_freshness", "odor_state",
}


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().str.strip().isin({"true", "1", "yes", "ok"})


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    if "timestamp_utc" in frame:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        frame = frame.sort_values("timestamp_utc")
    return frame.reset_index(drop=True)


def choose_channels(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, str], dict[str, Any]]:
    selected = dict(CHANNELS)
    dropped: dict[str, str] = {}
    saturation: dict[str, float] = {}

    for name, reason in EXCLUDED_CHANNELS.items():
        selected.pop(name, None)
        dropped[name] = reason

    for name, raw_col in RAW_TGS.items():
        values = pd.concat(
            [pd.to_numeric(frame[raw_col], errors="coerce") for frame in raw.values()],
            ignore_index=True,
        ).dropna()
        rate = float((values >= 4095).mean())
        saturation[name] = rate

        # Do not remove channels other than the explicitly excluded TGS2600.
        # A value of 4095 is retained as a clipped but still informative input.

    for name, source in list(selected.items()):
        if any(source not in frame for frame in raw.values()):
            selected.pop(name)
            dropped[name] = "missing source column"
            continue
        values = pd.concat(
            [pd.to_numeric(frame[source], errors="coerce") for frame in raw.values()],
            ignore_index=True,
        ).dropna()
        if values.empty or float(values.var(ddof=0)) <= 1e-18:
            selected.pop(name)
            dropped[name] = "no usable variation"
    if len(selected) < 2:
        raise ValueError("Fewer than two usable channels remain")
    return selected, {"dropped_channels": dropped, "saturation_rates": saturation}


def clean(frame: pd.DataFrame, selected: dict[str, str]) -> pd.DataFrame:
    valid = pd.Series(True, index=frame.index)
    for column in STATUS_COLUMNS:
        if column in frame:
            valid &= bool_series(frame[column])
    out = pd.DataFrame(index=frame.index)
    if "elapsed_s" in frame:
        out["elapsed_s"] = pd.to_numeric(frame["elapsed_s"], errors="coerce")
    for name, source in selected.items():
        out[name] = pd.to_numeric(frame[source], errors="coerce")
    valid &= out[list(selected)].notna().all(axis=1)
    return out.loc[valid].reset_index(drop=True)


def robust_baseline(frame: pd.DataFrame, sensors: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    medians, scales = {}, {}
    for sensor in sensors:
        values = frame[sensor].to_numpy(float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = 1.4826 * mad
        floor = max(abs(median) * 1e-3, 1e-6)
        medians[sensor] = median
        scales[sensor] = max(robust_scale, floor)
    return medians, scales


def normalize(frame: pd.DataFrame, medians: dict[str, float], scales: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    for sensor in medians:
        out[sensor] = (out[sensor].astype(float) - medians[sensor]) / scales[sensor]
    return out


def split_recording(frame: pd.DataFrame, recording: Recording) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    split = int(len(frame) * TRAIN_FRACTION)
    train_end = split - PURGE_ROWS
    valid_start = split + PURGE_ROWS
    train, valid = frame.iloc[:train_end].copy(), frame.iloc[valid_start:].copy()
    if min(len(train), len(valid)) < WINDOW_SIZE:
        raise ValueError(
            f"{recording.name}: insufficient rows after the leakage purge "
            f"(train={len(train)}, validation={len(valid)})"
        )
    for part, name in ((train, "train"), (valid, "validation")):
        part["recording"] = recording.name
        part["split"] = name
        part["food_group"] = recording.food_group
        part["fruit_freshness"] = recording.fruit_freshness
        part["meat_freshness"] = recording.meat_freshness
        part["odor_state"] = recording.odor_state
    summary = {
        "total": len(frame), "train": len(train),
        "purged": valid_start - train_end, "validation": len(valid),
    }
    return train, valid, summary


def safe_auc(values: np.ndarray) -> float:
    return float(np.trapezoid(values) if hasattr(np, "trapezoid") else np.trapz(values))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else 0.0


def window_features(window: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    x = np.arange(len(window), dtype=float)
    for sensor in sensors:
        values = window[sensor].to_numpy(float)
        result.update({
            f"{sensor}_mean": float(np.mean(values)),
            f"{sensor}_std": float(np.std(values)),
            f"{sensor}_min": float(np.min(values)),
            f"{sensor}_max": float(np.max(values)),
            f"{sensor}_median": float(np.median(values)),
            f"{sensor}_iqr": float(np.quantile(values, .75) - np.quantile(values, .25)),
            f"{sensor}_range": float(np.ptp(values)),
            f"{sensor}_delta": float(values[-1] - values[0]),
            f"{sensor}_slope": float(np.polyfit(x, values, 1)[0]),
            f"{sensor}_auc": safe_auc(values),
        })
    for i, first in enumerate(sensors):
        for second in sensors[i + 1:]:
            result[f"corr_{first}_{second}"] = safe_corr(
                window[first].to_numpy(float), window[second].to_numpy(float)
            )
    tgs = [sensor for sensor in sensors if sensor.startswith("tgs")]
    if tgs:
        means = np.array([window[sensor].mean() for sensor in tgs])
        result["tgs_array_mean"] = float(means.mean())
        result["tgs_array_spread"] = float(means.std())
    if {"nh3", "h2s"}.issubset(sensors):
        result["nh3_minus_h2s"] = float(window["nh3"].mean() - window["h2s"].mean())
    return result


def make_windows(parts: dict[str, pd.DataFrame], sensors: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for recording, frame in parts.items():
        for start in range(0, len(frame) - WINDOW_SIZE + 1, WINDOW_STRIDE):
            stop = start + WINDOW_SIZE
            row: dict[str, Any] = window_features(frame.iloc[start:stop], sensors)
            row.update({
                "recording": recording, "split": frame["split"].iloc[0],
                "window_start": start, "window_end": stop - 1,
                "food_group": frame["food_group"].iloc[0],
                "fruit_freshness": frame["fruit_freshness"].iloc[0],
                "meat_freshness": frame["meat_freshness"].iloc[0],
                "odor_state": frame["odor_state"].iloc[0],
            })
            rows.append(row)
    return pd.DataFrame(rows)


def scaled(model: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(1e-12)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def unscaled(model: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(1e-12)),
        ("model", model),
    ])


def candidate_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": scaled(LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        )),
        "SVM_RBF": scaled(SVC(
            C=5.0, kernel="rbf", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE
        )),
        "RandomForest": unscaled(RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        )),
        "ExtraTrees": unscaled(ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
    }


def train_task(
    task: str, config: dict[str, str | None],
    train_windows: pd.DataFrame, valid_windows: pd.DataFrame,
) -> pd.DataFrame:
    target, subset = str(config["target"]), config["subset"]
    train, valid = train_windows.copy(), valid_windows.copy()
    if subset:
        train = train[train["food_group"] == subset]
        valid = valid[valid["food_group"] == subset]
    features = [column for column in train if column not in META_COLUMNS]
    x_train, x_valid = train[features], valid[features]
    y_train, y_valid = train[target].astype(str), valid[target].astype(str)
    rows, fitted, predictions = [], {}, {}
    for name, template in candidate_models().items():
        model = clone(template)
        model.fit(x_train, y_train)
        prediction = model.predict(x_valid)
        rows.append({
            "task": task, "model": name,
            "accuracy": accuracy_score(y_valid, prediction),
            "balanced_accuracy": balanced_accuracy_score(y_valid, prediction),
            "macro_f1": f1_score(y_valid, prediction, average="macro", zero_division=0),
        })
        fitted[name], predictions[name] = model, prediction
    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"], ascending=False
    ).reset_index(drop=True)
    best_name = str(results.loc[0, "model"])
    package = {
        "task": task, "target": target, "subset": subset,
        "model_name": best_name, "model": fitted[best_name],
        "feature_columns": features, "classes": sorted(y_train.unique()),
        "window_size": WINDOW_SIZE, "window_stride": WINDOW_STRIDE,
    }
    joblib.dump(package, MODEL_DIR / f"{task}_model.joblib")
    results.to_csv(RESULTS_DIR / f"{task}_model_comparison.csv", index=False)
    pd.DataFrame({
        "recording": valid["recording"].to_numpy(),
        "window_start": valid["window_start"].to_numpy(),
        "true_label": y_valid.to_numpy(),
        "prediction": predictions[best_name],
    }).to_csv(RESULTS_DIR / f"{task}_validation_predictions.csv", index=False)
    return results


def plot_training_data(
    normalized: dict[str, pd.DataFrame], windows: pd.DataFrame,
    sensors: list[str], model_results: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 2.0 * len(sensors)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, sensor in zip(axes, sensors):
        for name, frame in normalized.items():
            axis.plot(np.arange(len(frame)), frame[sensor], linewidth=1.2, label=name)
        axis.set_ylabel(sensor)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Sample index (approximately 1 s per sample)")
    fig.suptitle("Baseline-normalized sensor responses", y=.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sensor_response_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    features = [column for column in windows if column not in META_COLUMNS]
    matrix = SimpleImputer(strategy="median").fit_transform(windows[features])
    matrix = StandardScaler().fit_transform(matrix)
    scores = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label in sorted(windows["odor_state"].unique()):
        mask = windows["odor_state"].eq(label).to_numpy()
        ax.scatter(scores[mask, 0], scores[mask, 1], s=35, alpha=.8, label=label)
    ax.set(xlabel="Principal component 1", ylabel="Principal component 2",
           title="PCA of temporal-window features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = model_results.pivot(index="model", columns="task", values="macro_f1")
    pivot.plot(kind="bar", ax=ax)
    ax.set(ylabel="Validation macro-F1", ylim=(0, 1.05),
           title="Model comparison on the temporal holdout")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in (MODEL_DIR, RESULTS_DIR, FIGURE_DIR, PREPARED_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw = {record.name: load_csv(BASE_DIR / record.filename) for record in RECORDINGS}
    selected, quality = choose_channels(raw)
    sensors = list(selected)
    cleaned = {name: clean(frame, selected) for name, frame in raw.items()}
    baseline_medians, baseline_scales = robust_baseline(cleaned["baseline"], sensors)
    normalized = {
        name: normalize(frame, baseline_medians, baseline_scales)
        for name, frame in cleaned.items()
    }

    train_parts, valid_parts, split_summary = {}, {}, {}
    by_name = {record.name: record for record in RECORDINGS}
    for name, frame in normalized.items():
        train, valid, summary = split_recording(frame, by_name[name])
        train_parts[name], valid_parts[name], split_summary[name] = train, valid, summary

    train_windows = make_windows(train_parts, sensors)
    valid_windows = make_windows(valid_parts, sensors)
    train_windows.to_csv(PREPARED_DIR / "training_windows.csv", index=False)
    valid_windows.to_csv(PREPARED_DIR / "validation_windows.csv", index=False)

    all_results = []
    for task, config in TASKS.items():
        result = train_task(task, config, train_windows, valid_windows)
        all_results.append(result)
        print(f"\n{task}\n{result.to_string(index=False)}")
    comparison = pd.concat(all_results, ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "all_model_comparison.csv", index=False)

    metadata = {
        "selected_channels": selected,
        "channel_quality": quality,
        "baseline_medians": baseline_medians,
        "baseline_scales": baseline_scales,
        "split_summary": split_summary,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "purge_rows": PURGE_ROWS,
        "session_baseline_adaptation": {
            "center_update_weight": 1.0,
            "minimum_scale_fraction_of_training": 0.5,
            "description": (
                "Before each test session, replace the normalization center "
                "with the new clean-air median. Use the larger of the new "
                "MAD scale and 50% of the training scale. Classifier weights "
                "remain frozen because the session baseline has no class label."
            ),
        },
        "limitation": (
            "One continuous recording per labeled condition; results are a "
            "temporal holdout and are not independent cross-day performance."
        ),
    }
    (OUTPUT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_training_data(
        normalized,
        pd.concat([train_windows, valid_windows], ignore_index=True),
        sensors, comparison,
    )
    print(f"\nSaved models, results, and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

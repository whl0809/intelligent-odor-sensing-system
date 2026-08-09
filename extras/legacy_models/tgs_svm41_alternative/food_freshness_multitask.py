#!/usr/bin/env python3
"""Train food/freshness models from TGS + SVM41 recordings.

The script automatically discovers ``enose_*.csv`` files created by
``collect_all_sensors.py``.  Labels are read from the CSV ``food_group``
column first and inferred from names such as ``enose_fresh_meat_2.csv`` only
as a fallback.  Timestamp-only filenames therefore work as long as the
collector wrote the label inside the CSV.

All model inputs come from usable ADS7828/TGS channels and SVM41.  SVM41 rows
with zero VOC/NOx indices are removed.  Individual TGS rail-saturated values
are treated as missing; a TGS channel with no usable training/baseline values
is excluded globally.  EC Sense NH3/H2S columns are neither read nor used.
"""

from __future__ import annotations

import argparse
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "model" else BASE_DIR
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "food_freshness"
RANDOM_STATE = 42
SCHEMA_VERSION = 3
TGS_NEAR_RAIL_LOW = 4
TGS_NEAR_RAIL_HIGH = 4091
TGS_REFERENCE_V = 2.5
TGS_FULL_SCALE_CODES = 4096

TGS_NAMES = (
    "tgs2620",
    "tgs2610",
    "tgs2611",
    "tgs2600",
    "tgs2602",
    "tgs2603",
)

# The first available alias is used.  New collector names are listed first;
# later aliases make previously recorded TGS/SVM41 CSVs easier to reuse.
SENSOR_ALIASES: dict[str, tuple[str, ...]] = {
    **{
        name: (
            f"{name}_voltage_v",
            f"{name}_v",
            name,
        )
        for name in TGS_NAMES
    },
    "svm41_temperature": (
        "svm41_temperature_c",
        "temperature_c",
    ),
    "svm41_humidity": (
        "svm41_relative_humidity_pct",
        "humidity_rh",
        "relative_humidity_pct",
    ),
    "svm41_voc_index": (
        "svm41_voc_index",
        "voc_index",
    ),
    "svm41_nox_index": (
        "svm41_nox_index",
        "nox_index",
    ),
    "svm41_raw_voc": (
        "svm41_raw_voc_ticks",
        "raw_voc_ticks",
        "sraw_voc",
    ),
    "svm41_raw_nox": (
        "svm41_raw_nox_ticks",
        "raw_nox_ticks",
        "sraw_nox",
    ),
}

REQUIRED_SVM41_SENSORS = {
    "svm41_temperature",
    "svm41_humidity",
    "svm41_voc_index",
    "svm41_nox_index",
}
CONDITION_ALIASES = {
    "blank": "blank",
    "baseline": "blank",
    "clean_air": "blank",
    "air": "blank",
    "fresh_banana": "fresh_banana",
    "banana_fresh": "fresh_banana",
    "fermented_banana": "fermented_banana",
    "banana_fermented": "fermented_banana",
    "fresh_meat": "fresh_meat",
    "meat_fresh": "fresh_meat",
    "spoiled_meat": "spoiled_meat",
    "meat_spoiled": "spoiled_meat",
}

TASKS: dict[str, dict[str, str | None]] = {
    "food_group": {"target": "food_group", "subset": None},
    "fruit_freshness": {"target": "fruit_freshness", "subset": "fruit"},
    "meat_freshness": {"target": "meat_freshness", "subset": "meat"},
    "odor_state": {"target": "odor_state", "subset": None},
}

META_COLUMNS = {
    "recording",
    "source_file",
    "split",
    "window_start",
    "window_end",
    "food_group",
    "fruit_freshness",
    "meat_freshness",
    "odor_state",
}


@dataclass(frozen=True)
class Recording:
    name: str
    path: Path
    condition: str
    food_group: str
    fruit_freshness: str
    meat_freshness: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TGS + SVM41 food/freshness classifiers."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing enose_*.csv (default: data/raw)",
    )
    parser.add_argument(
        "--pattern",
        default="enose_*.csv",
        help="CSV filename glob inside --data-dir (default: enose_*.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="model/result output directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="cleaned rows and window tables (default: data/processed)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="rows per temporal feature window (default: 20)",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=5,
        help="rows between window starts (default: 5)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="chronological training fraction per file (default: 0.70)",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help=(
            "optional manifest with filename and food_group/condition columns; "
            "useful for old timestamp-only CSVs without embedded labels"
        ),
    )
    parser.add_argument(
        "--require-all-labeled",
        action="store_true",
        help="fail instead of skipping CSV files whose class is unknown",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip PNG visualizations",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size < 5:
        raise ValueError("--window-size must be at least 5")
    if args.window_stride < 1:
        raise ValueError("--window-stride must be at least 1")
    if not 0.5 <= args.train_fraction <= 0.85:
        raise ValueError("--train-fraction must be between 0.5 and 0.85")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "ok"}
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    if "timestamp_utc" in frame:
        timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        frame = frame.assign(_timestamp_sort=timestamps).sort_values("_timestamp_sort")
        frame = frame.drop(columns="_timestamp_sort")
    elif "sequence" in frame:
        frame = frame.sort_values("sequence")
    return frame.reset_index(drop=True)


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def canonical_condition(value: Any) -> str | None:
    token = normalize_token(value)
    if token in CONDITION_ALIASES:
        return CONDITION_ALIASES[token]
    # Allow filenames containing a known class plus a recording suffix.
    for alias in sorted(CONDITION_ALIASES, key=len, reverse=True):
        if re.search(rf"(?:^|_){re.escape(alias)}(?:_|$)", token):
            return CONDITION_ALIASES[alias]
    return None


def condition_targets(condition: str) -> tuple[str, str, str]:
    if condition == "blank":
        return "blank", "not_applicable", "not_applicable"
    if condition == "fresh_banana":
        return "fruit", "fresh", "not_applicable"
    if condition == "fermented_banana":
        return "fruit", "fermented", "not_applicable"
    if condition == "fresh_meat":
        return "meat", "not_applicable", "fresh"
    if condition == "spoiled_meat":
        return "meat", "not_applicable", "spoiled"
    raise ValueError(f"unsupported condition: {condition}")


def read_label_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    table = pd.read_csv(path)
    if "filename" not in table:
        raise ValueError("--labels-csv must contain a filename column")
    label_column = "condition" if "condition" in table else "food_group"
    if label_column not in table:
        raise ValueError(
            "--labels-csv must contain a condition or food_group column"
        )
    result: dict[str, str] = {}
    for _, row in table.iterrows():
        condition = canonical_condition(row[label_column])
        if condition is None:
            raise ValueError(
                f"Unknown label {row[label_column]!r} for {row['filename']!r}"
            )
        result[Path(str(row["filename"])).name] = condition
    return result


def embedded_condition(frame: pd.DataFrame) -> str | None:
    for column in ("condition", "odor_state", "food_group"):
        if column not in frame:
            continue
        values = [
            canonical_condition(value)
            for value in frame[column].dropna().astype(str).unique()
        ]
        known = sorted({value for value in values if value is not None})
        if len(known) == 1:
            return known[0]
        if len(known) > 1:
            raise ValueError(f"column {column} contains multiple class labels: {known}")
    return None


def discover_recordings(
    data_dir: Path,
    pattern: str,
    manifest: dict[str, str],
    require_all_labeled: bool,
) -> tuple[list[Recording], list[str]]:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {data_dir / pattern}")
    recordings: list[Recording] = []
    skipped: list[str] = []
    for path in paths:
        frame = load_csv(path)
        condition = (
            manifest.get(path.name)
            or embedded_condition(frame)
            or canonical_condition(path.stem.removeprefix("enose_"))
        )
        if condition is None:
            message = (
                f"{path.name}: no known label in CSV or filename; "
                "add food_group to the CSV or use --labels-csv"
            )
            if require_all_labeled:
                raise ValueError(message)
            skipped.append(message)
            continue
        food, fruit, meat = condition_targets(condition)
        recordings.append(
            Recording(
                name=path.stem,
                path=path,
                condition=condition,
                food_group=food,
                fruit_freshness=fruit,
                meat_freshness=meat,
            )
        )
    if not recordings:
        raise ValueError("No labeled recordings were found")
    return recordings, skipped


def case_insensitive_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def resolve_sensor_sources(frame: pd.DataFrame) -> dict[str, str]:
    columns = case_insensitive_columns(frame)
    resolved: dict[str, str] = {}
    for sensor, aliases in SENSOR_ALIASES.items():
        for alias in aliases:
            if alias.lower() in columns:
                resolved[sensor] = columns[alias.lower()]
                break
    return resolved


def tgs_saturation_mask(
    frame: pd.DataFrame,
    sensor: str,
    source_column: str,
) -> pd.Series:
    """Locate rail values using ADC raw codes when available, else voltage."""
    columns = case_insensitive_columns(frame)
    raw_column = columns.get(f"{sensor}_raw")
    if raw_column is not None:
        values = pd.to_numeric(frame[raw_column], errors="coerce")
        return (values <= TGS_NEAR_RAIL_LOW) | (values >= TGS_NEAR_RAIL_HIGH)

    values = pd.to_numeric(frame[source_column], errors="coerce")
    finite = values.dropna()
    # Legacy files may put ADC codes in the canonical sensor column.  Values
    # above the ADC reference voltage make that representation unambiguous.
    if not finite.empty and float(finite.max()) > TGS_REFERENCE_V * 1.5:
        return (values <= TGS_NEAR_RAIL_LOW) | (values >= TGS_NEAR_RAIL_HIGH)
    low_v = TGS_NEAR_RAIL_LOW * TGS_REFERENCE_V / TGS_FULL_SCALE_CODES
    high_v = TGS_NEAR_RAIL_HIGH * TGS_REFERENCE_V / TGS_FULL_SCALE_CODES
    return (values <= low_v) | (values >= high_v)


def svm41_ready_mask(frame: pd.DataFrame, sources: dict[str, str]) -> pd.Series:
    voc = pd.to_numeric(frame[sources["svm41_voc_index"]], errors="coerce")
    nox = pd.to_numeric(frame[sources["svm41_nox_index"]], errors="coerce")
    ready = voc.gt(0.0) & nox.gt(0.0)
    if "svm41_ok" in frame:
        ready &= bool_series(frame["svm41_ok"])
    return ready


def choose_sensors(
    raw: dict[str, pd.DataFrame], blank_names: list[str]
) -> tuple[list[str], dict[str, Any]]:
    sources_by_recording = {
        name: resolve_sensor_sources(frame) for name, frame in raw.items()
    }
    missing: dict[str, list[str]] = {}
    selected: list[str] = []
    for sensor in SENSOR_ALIASES:
        absent = [
            name for name, sources in sources_by_recording.items() if sensor not in sources
        ]
        if absent:
            missing[sensor] = absent
    required_missing = sorted(
        sensor
        for sensor in REQUIRED_SVM41_SENSORS
        if sensor in missing
    )
    if required_missing:
        details = "; ".join(
            f"{sensor} missing from {', '.join(missing[sensor])}"
            for sensor in required_missing
        )
        raise ValueError(f"Required SVM41 columns are missing: {details}")

    tgs_quality: dict[str, dict[str, dict[str, int | float | bool]]] = {}
    excluded_tgs: dict[str, str] = {}
    for sensor in TGS_NAMES:
        if sensor in missing:
            excluded_tgs[sensor] = (
                "missing from: " + ", ".join(missing[sensor])
            )
            continue
        total_valid = 0
        blank_valid = 0
        sensor_stats: dict[str, dict[str, int | float | bool]] = {}
        for name, frame in raw.items():
            source = sources_by_recording[name][sensor]
            values = pd.to_numeric(frame[source], errors="coerce")
            saturated = tgs_saturation_mask(frame, sensor, source)
            ready = svm41_ready_mask(frame, sources_by_recording[name])
            available = values.notna() & ready
            normal = available & ~saturated.fillna(False)
            normal_count = int(normal.sum())
            saturated_count = int((available & saturated.fillna(False)).sum())
            total_valid += normal_count
            if name in blank_names:
                blank_valid += normal_count
            sensor_stats[name] = {
                "available_count": int(available.sum()),
                "normal_count": normal_count,
                "saturated_count": saturated_count,
                "saturation_rate": (
                    float(saturated_count / int(available.sum()))
                    if int(available.sum())
                    else 0.0
                ),
                "fully_saturated": bool(available.any() and normal_count == 0),
            }
        tgs_quality[sensor] = sensor_stats
        if total_valid == 0:
            excluded_tgs[sensor] = "all available readings are saturated"
        elif blank_valid == 0:
            excluded_tgs[sensor] = "no unsaturated blank/baseline readings"

    for sensor in SENSOR_ALIASES:
        if sensor in TGS_NAMES:
            if sensor not in excluded_tgs:
                selected.append(sensor)
        elif sensor not in missing:
            selected.append(sensor)

    selected_tgs = [sensor for sensor in selected if sensor in TGS_NAMES]
    if not selected_tgs:
        raise ValueError(
            "Every TGS channel is missing, fully saturated, or unusable in blank data"
        )

    quality = {
        "source_columns": sources_by_recording,
        "optional_missing": {
            sensor: names
            for sensor, names in missing.items()
            if sensor not in REQUIRED_SVM41_SENSORS
        },
        "tgs_quality": tgs_quality,
        "excluded_tgs": excluded_tgs,
        "selected_tgs": selected_tgs,
        "svm41_filter": {
            name: {
                "raw_rows": len(frame),
                "usable_rows": int(
                    svm41_ready_mask(frame, sources_by_recording[name]).sum()
                ),
                "removed_zero_or_invalid_rows": int(
                    (~svm41_ready_mask(frame, sources_by_recording[name])).sum()
                ),
            }
            for name, frame in raw.items()
        },
    }
    return selected, quality


def clean(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    sources = resolve_sensor_sources(frame)
    missing = [sensor for sensor in sensors if sensor not in sources]
    if missing:
        raise ValueError(f"Missing model sensor columns: {missing}")

    valid = svm41_ready_mask(frame, sources)

    out = pd.DataFrame(index=frame.index)
    if "elapsed_s" in frame:
        out["elapsed_s"] = pd.to_numeric(frame["elapsed_s"], errors="coerce")
    for sensor in sensors:
        out[sensor] = pd.to_numeric(frame[sources[sensor]], errors="coerce")
        if sensor in TGS_NAMES:
            saturated = tgs_saturation_mask(frame, sensor, sources[sensor])
            out.loc[saturated.fillna(False), sensor] = np.nan
            if "ads7828_ok" in frame:
                out.loc[~bool_series(frame["ads7828_ok"]), sensor] = np.nan

    required_present = [sensor for sensor in REQUIRED_SVM41_SENSORS if sensor in sensors]
    valid &= out[required_present].notna().all(axis=1)
    return out.loc[valid].reset_index(drop=True)


def robust_baseline(
    frame: pd.DataFrame, sensors: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for sensor in sensors:
        values = frame[sensor].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"No usable blank/baseline values remain for {sensor}"
            )
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        floor = max(abs(median) * 1e-3, 1e-6)
        medians[sensor] = median
        scales[sensor] = max(1.4826 * mad, floor)
    return medians, scales


def normalize(
    frame: pd.DataFrame,
    medians: dict[str, float],
    scales: dict[str, float],
) -> pd.DataFrame:
    out = frame.copy()
    for sensor in medians:
        out[sensor] = (out[sensor].astype(float) - medians[sensor]) / scales[sensor]
    return out


def label_frame(frame: pd.DataFrame, recording: Recording, split: str) -> pd.DataFrame:
    out = frame.copy()
    out["recording"] = recording.name
    out["source_file"] = recording.path.name
    out["split"] = split
    out["food_group"] = recording.food_group
    out["fruit_freshness"] = recording.fruit_freshness
    out["meat_freshness"] = recording.meat_freshness
    out["odor_state"] = recording.condition
    return out


def split_recording(
    frame: pd.DataFrame,
    recording: Recording,
    train_fraction: float,
    window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    # Windows are built separately after this non-overlapping row split, so no
    # raw sample can appear in both train and validation.  No extra purge is
    # needed; this lets a normal 90-frame recording remain usable.
    split = int(len(frame) * train_fraction)
    split = min(max(split, window_size), len(frame) - window_size)
    if split < window_size or len(frame) - split < window_size:
        raise ValueError(
            f"{recording.path.name}: {len(frame)} valid rows cannot provide "
            f"both train and validation windows of {window_size} rows"
        )
    train = label_frame(frame.iloc[:split], recording, "train")
    valid = label_frame(frame.iloc[split:], recording, "validation")
    return train, valid, {
        "total_valid": len(frame),
        "training_rows": len(train),
        "validation_rows": len(valid),
    }


def safe_auc(values: np.ndarray, x: np.ndarray | None = None) -> float:
    if values.size < 2:
        return float("nan")
    coordinates = np.arange(values.size, dtype=float) if x is None else x
    return float(
        np.trapezoid(values, coordinates)
        if hasattr(np, "trapezoid")
        else np.trapz(values, coordinates)
    )


def safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    paired = np.isfinite(first) & np.isfinite(second)
    first = first[paired]
    second = second[paired]
    if first.size < 2:
        return float("nan")
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else 0.0


def window_features(window: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    x = np.arange(len(window), dtype=float)
    for sensor in sensors:
        complete = window[sensor].to_numpy(float)
        finite = np.isfinite(complete)
        values = complete[finite]
        valid_x = x[finite]
        if values.size == 0:
            result.update(
                {
                    f"{sensor}_{suffix}": float("nan")
                    for suffix in (
                        "mean", "std", "min", "max", "median", "iqr",
                        "range", "delta", "slope", "auc", "valid_fraction",
                    )
                }
            )
            continue
        result.update(
            {
                f"{sensor}_mean": float(np.mean(values)),
                f"{sensor}_std": float(np.std(values)),
                f"{sensor}_min": float(np.min(values)),
                f"{sensor}_max": float(np.max(values)),
                f"{sensor}_median": float(np.median(values)),
                f"{sensor}_iqr": float(
                    np.quantile(values, 0.75) - np.quantile(values, 0.25)
                ),
                f"{sensor}_range": float(np.ptp(values)),
                f"{sensor}_delta": float(values[-1] - values[0]),
                f"{sensor}_slope": (
                    float(np.polyfit(valid_x, values, 1)[0])
                    if values.size >= 2
                    else float("nan")
                ),
                f"{sensor}_auc": safe_auc(values, valid_x),
                f"{sensor}_valid_fraction": float(values.size / len(window)),
            }
        )
    for index, first in enumerate(sensors):
        for second in sensors[index + 1 :]:
            result[f"corr_{first}_{second}"] = safe_corr(
                window[first].to_numpy(float), window[second].to_numpy(float)
            )
    active_tgs = [sensor for sensor in sensors if sensor in TGS_NAMES]
    tgs_means = np.array(
        [window[sensor].mean(skipna=True) for sensor in active_tgs], dtype=float
    )
    finite_tgs = tgs_means[np.isfinite(tgs_means)]
    result["tgs_array_mean"] = (
        float(finite_tgs.mean()) if finite_tgs.size else float("nan")
    )
    result["tgs_array_spread"] = (
        float(finite_tgs.std()) if finite_tgs.size else float("nan")
    )
    result["svm41_voc_minus_nox"] = float(
        window["svm41_voc_index"].mean() - window["svm41_nox_index"].mean()
    )
    return result


def make_windows(
    parts: dict[str, pd.DataFrame],
    sensors: list[str],
    window_size: int,
    window_stride: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for recording, frame in parts.items():
        for start in range(0, len(frame) - window_size + 1, window_stride):
            stop = start + window_size
            row: dict[str, Any] = window_features(frame.iloc[start:stop], sensors)
            row.update(
                {
                    "recording": recording,
                    "source_file": frame["source_file"].iloc[0],
                    "split": frame["split"].iloc[0],
                    "window_start": start,
                    "window_end": stop - 1,
                    "food_group": frame["food_group"].iloc[0],
                    "fruit_freshness": frame["fruit_freshness"].iloc[0],
                    "meat_freshness": frame["meat_freshness"].iloc[0],
                    "odor_state": frame["odor_state"].iloc[0],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def scaled(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(1e-12)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def unscaled(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(1e-12)),
            ("model", model),
        ]
    )


def candidate_models(min_class_windows: int) -> dict[str, Pipeline]:
    calibration_folds = max(2, min(3, min_class_windows))
    return {
        "LogisticRegression": scaled(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "SVM_RBF": scaled(
            CalibratedClassifierCV(
                SVC(C=5.0, kernel="rbf", class_weight="balanced"),
                method="sigmoid",
                cv=calibration_folds,
            )
        ),
        "RandomForest": unscaled(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "ExtraTrees": unscaled(
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def train_task(
    task: str,
    config: dict[str, str | None],
    train_windows: pd.DataFrame,
    valid_windows: pd.DataFrame,
    model_dir: Path,
    results_dir: Path,
    figure_dir: Path,
    window_size: int,
    window_stride: int,
    make_figures: bool,
) -> pd.DataFrame | None:
    target = str(config["target"])
    subset = config["subset"]
    train = train_windows
    valid = valid_windows
    if subset is not None:
        train = train[train["food_group"] == subset]
        valid = valid[valid["food_group"] == subset]
    if train.empty or valid.empty or train[target].nunique() < 2:
        print(f"Skipping {task}: fewer than two available classes")
        return None

    features = [column for column in train.columns if column not in META_COLUMNS]
    x_train = train[features]
    x_valid = valid[features]
    y_train = train[target].astype(str)
    y_valid = valid[target].astype(str)
    counts = y_train.value_counts()
    if int(counts.min()) < 2:
        raise ValueError(f"{task}: each training class needs at least 2 windows")

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, template in candidate_models(int(counts.min())).items():
        model = clone(template)
        model.fit(x_train, y_train)
        prediction = model.predict(x_valid)
        rows.append(
            {
                "task": task,
                "model": name,
                "accuracy": accuracy_score(y_valid, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_valid, prediction),
                "macro_f1": f1_score(
                    y_valid, prediction, average="macro", zero_division=0
                ),
            }
        )
        fitted[name] = model
        predictions[name] = prediction

    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"], ascending=False
    ).reset_index(drop=True)
    best_name = str(results.loc[0, "model"])
    package = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "target": target,
        "subset": subset,
        "model_name": best_name,
        "model": fitted[best_name],
        "feature_columns": features,
        "classes": sorted(y_train.unique()),
        "window_size": window_size,
        "window_stride": window_stride,
    }
    joblib.dump(package, model_dir / f"{task}_model.joblib")
    results.to_csv(results_dir / f"{task}_model_comparison.csv", index=False)
    best_prediction = predictions[best_name]
    pd.DataFrame(
        {
            "recording": valid["recording"].to_numpy(),
            "true_label": y_valid.to_numpy(),
            "prediction": best_prediction,
        }
    ).to_csv(results_dir / f"{task}_validation_predictions.csv", index=False)

    if make_figures:
        labels = sorted(set(y_valid) | set(best_prediction))
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ConfusionMatrixDisplay.from_predictions(
            y_valid,
            best_prediction,
            labels=labels,
            normalize="true",
            cmap="Blues",
            values_format=".2f",
            ax=ax,
        )
        ax.set_title(f"{task.replace('_', ' ').title()} — {best_name}")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task}_confusion_matrix.png", dpi=300)
        plt.close(fig)
    return results


def plot_overview(
    windows: pd.DataFrame,
    comparison: pd.DataFrame,
    figure_dir: Path,
) -> None:
    features = [column for column in windows if column not in META_COLUMNS]
    matrix = SimpleImputer(strategy="median").fit_transform(windows[features])
    matrix = StandardScaler().fit_transform(matrix)
    scores = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for label in sorted(windows["odor_state"].unique()):
        mask = windows["odor_state"].eq(label).to_numpy()
        ax.scatter(scores[mask, 0], scores[mask, 1], s=28, alpha=0.8, label=label)
    ax.set(
        xlabel="Principal component 1",
        ylabel="Principal component 2",
        title="TGS + SVM41 temporal-window features",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_pca.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = comparison.pivot(index="model", columns="task", values="macro_f1")
    pivot.plot(kind="bar", ax=ax)
    ax.set(
        ylabel="Validation macro-F1",
        ylim=(0, 1.05),
        title="Model comparison on chronological holdout",
    )
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=300)
    plt.close(fig)


def saturation_rates(raw: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for recording, frame in raw.items():
        columns = case_insensitive_columns(frame)
        sources = resolve_sensor_sources(frame)
        ready = svm41_ready_mask(frame, sources)
        rates: dict[str, float] = {}
        for sensor in TGS_NAMES:
            raw_column = columns.get(f"{sensor}_raw")
            if raw_column is not None:
                values = pd.to_numeric(frame.loc[ready, raw_column], errors="coerce").dropna()
                if not values.empty:
                    rates[sensor] = float(((values <= 4) | (values >= 4091)).mean())
        result[recording] = rates
    return result


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    model_dir = output_dir / "models"
    results_dir = output_dir / "results"
    prepared_dir = processed_dir
    figure_dir = output_dir / "visualizations"
    for directory in (model_dir, results_dir, prepared_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = read_label_manifest(args.labels_csv)
    recordings, skipped = discover_recordings(
        data_dir, args.pattern, manifest, args.require_all_labeled
    )
    for message in skipped:
        print(f"Skipping: {message}")
    print("Discovered labeled recordings:")
    for recording in recordings:
        print(f"  {recording.path.name}: {recording.condition}")

    raw = {recording.name: load_csv(recording.path) for recording in recordings}
    blank_names = [
        recording.name for recording in recordings if recording.condition == "blank"
    ]
    if not blank_names:
        raise ValueError(
            "At least one blank/baseline recording is required for normalization"
        )
    sensors, channel_quality = choose_sensors(raw, blank_names)
    cleaned = {name: clean(frame, sensors) for name, frame in raw.items()}
    for recording in recordings:
        cleaned_frame = cleaned[recording.name].copy()
        cleaned_frame.insert(0, "source_file", recording.path.name)
        cleaned_frame.insert(1, "condition", recording.condition)
        cleaned_frame.to_csv(
            processed_dir / f"cleaned_{recording.path.name}", index=False
        )
    blank_rows = pd.concat([cleaned[name] for name in blank_names], ignore_index=True)
    if len(blank_rows) < args.window_size:
        raise ValueError("Not enough valid blank rows for baseline normalization")
    baseline_medians, baseline_scales = robust_baseline(blank_rows, sensors)
    normalized = {
        name: normalize(frame, baseline_medians, baseline_scales)
        for name, frame in cleaned.items()
    }

    by_name = {recording.name: recording for recording in recordings}
    train_parts: dict[str, pd.DataFrame] = {}
    valid_parts: dict[str, pd.DataFrame] = {}
    split_summary: dict[str, dict[str, int]] = {}
    for name, frame in normalized.items():
        train, valid, summary = split_recording(
            frame,
            by_name[name],
            args.train_fraction,
            args.window_size,
        )
        train_parts[name] = train
        valid_parts[name] = valid
        split_summary[name] = summary

    train_windows = make_windows(
        train_parts, sensors, args.window_size, args.window_stride
    )
    valid_windows = make_windows(
        valid_parts, sensors, args.window_size, args.window_stride
    )
    train_windows.to_csv(prepared_dir / "training_windows.csv", index=False)
    valid_windows.to_csv(prepared_dir / "validation_windows.csv", index=False)

    all_results: list[pd.DataFrame] = []
    trained_tasks: list[str] = []
    for task, config in TASKS.items():
        result = train_task(
            task,
            config,
            train_windows,
            valid_windows,
            model_dir,
            results_dir,
            figure_dir,
            args.window_size,
            args.window_stride,
            not args.no_figures,
        )
        if result is not None:
            all_results.append(result)
            trained_tasks.append(task)
            print(f"\n{task}\n{result.to_string(index=False)}")
    if not all_results:
        raise ValueError("No classification task had at least two classes")

    comparison = pd.concat(all_results, ignore_index=True)
    comparison.to_csv(results_dir / "all_model_comparison.csv", index=False)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sensor_scope": "usable ADS7828/TGS channels plus external SVM41; no EC Sense",
        "data_dir": str(data_dir),
        "pattern": args.pattern,
        "recordings": [
            {
                "file": recording.path.name,
                "condition": recording.condition,
                "food_group": recording.food_group,
            }
            for recording in recordings
        ],
        "skipped_files": skipped,
        "selected_sensors": sensors,
        "sensor_aliases": {sensor: list(SENSOR_ALIASES[sensor]) for sensor in sensors},
        "channel_quality": channel_quality,
        "tgs_near_rail_rates": saturation_rates(raw),
        "cleaning_policy": {
            "svm41": (
                "keep only rows where VOC Index > 0 and NOx Index > 0; "
                "this dynamically removes the approximately 45-second warm-up"
            ),
            "tgs_saturation": (
                f"raw <= {TGS_NEAR_RAIL_LOW} or raw >= {TGS_NEAR_RAIL_HIGH} "
                "is replaced with NaN; normal values from the same channel remain"
            ),
            "global_tgs_exclusion": (
                "exclude a channel only when it has no unsaturated training "
                "values or no unsaturated blank/baseline values"
            ),
        },
        "baseline_medians": baseline_medians,
        "baseline_scales": baseline_scales,
        "split_summary": split_summary,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "train_fraction": args.train_fraction,
        "trained_tasks": trained_tasks,
        "session_baseline_adaptation": {
            "center_update_weight": 1.0,
            "minimum_scale_fraction_of_training": 0.5,
        },
        "limitation": (
            "The saved validation scores are chronological holdout results. "
            "They are not independent cross-day accuracy unless the input "
            "recordings themselves represent separate days."
        ),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not args.no_figures:
        plot_overview(
            pd.concat([train_windows, valid_windows], ignore_index=True),
            comparison,
            figure_dir,
        )
    print(f"\nCleaned data saved to {processed_dir}")
    print(f"Models, metrics, metadata, and figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

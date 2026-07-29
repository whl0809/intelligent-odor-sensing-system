from __future__ import annotations

import json
import math
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .csv_logger import frame_to_row
from .records import Frame


DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "food_freshness"
)
MODEL_FILENAMES = {
    "combined_class": "combined_class_best_model.joblib",
    "food_type": "food_type_best_model.joblib",
    "freshness": "freshness_best_model.joblib",
}
TASK_ORDER = ("food_type", "freshness", "combined_class")
STATUS_COLUMNS = ("ads7828_ok", "nh3_ok", "h2s_ok")


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "classification dependencies are unavailable; "
            "run: python -m pip install -e '.[classification]'"
        ) from exc
    return joblib, np, pd


def _parse_bool_series(series: Any) -> Any:
    _, _, pd = _dependencies()
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes", "ok"})


def _prepare_sensor_frame(
    raw_frame: Any,
    selected_channels: Mapping[str, str],
) -> Any:
    _, _, pd = _dependencies()
    missing_columns = [
        source
        for source in selected_channels.values()
        if source not in raw_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "input is missing sensor column(s):\n" + "\n".join(missing_columns)
        )

    valid_mask = pd.Series(True, index=raw_frame.index)
    for status_column in STATUS_COLUMNS:
        if status_column in raw_frame.columns:
            valid_mask &= _parse_bool_series(raw_frame[status_column])

    cleaned = pd.DataFrame(index=raw_frame.index)
    for sensor_name, source_column in selected_channels.items():
        cleaned[sensor_name] = pd.to_numeric(
            raw_frame[source_column],
            errors="coerce",
        )
    valid_mask &= cleaned.notna().all(axis=1)

    if "timestamp_utc" in raw_frame.columns:
        cleaned.insert(0, "timestamp_utc", raw_frame["timestamp_utc"])
    if "elapsed_s" in raw_frame.columns:
        position = 1 if "timestamp_utc" in cleaned.columns else 0
        cleaned.insert(
            position,
            "elapsed_s",
            pd.to_numeric(raw_frame["elapsed_s"], errors="coerce"),
        )

    cleaned = cleaned.loc[valid_mask].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(
            "no valid rows remain after checking status flags and sensor values"
        )
    return cleaned


def _normalize_sensor_frame(
    clean_frame: Any,
    baseline_means: Mapping[str, float],
    baseline_scales: Mapping[str, float],
) -> Any:
    normalized = clean_frame.copy()
    for column in normalized.columns:
        if column in {"timestamp_utc", "elapsed_s"}:
            continue
        if column not in baseline_means:
            raise ValueError(f"missing baseline mean for channel: {column}")
        if column not in baseline_scales:
            raise ValueError(f"missing baseline scale for channel: {column}")
        scale = float(baseline_scales[column])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid baseline scale for {column}: {scale}")
        normalized[column] = (
            normalized[column].astype(float) - float(baseline_means[column])
        ) / scale
    return normalized


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def _safe_auc(values: Any) -> float:
    _, np, _ = _dependencies()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values))
    return float(np.trapz(values))


def _safe_correlation(first: Any, second: Any) -> float:
    _, np, _ = _dependencies()
    if len(first) < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else 0.0


def _extract_features(window: Any, sensor_names: Sequence[str]) -> dict[str, float]:
    _, np, _ = _dependencies()
    features: dict[str, float] = {}
    x_axis = np.arange(len(window), dtype=float)

    for sensor_name in sensor_names:
        values = window[sensor_name].to_numpy(dtype=float)
        prefix = _safe_name(sensor_name)
        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_std"] = float(np.std(values, ddof=0))
        features[f"{prefix}_min"] = float(np.min(values))
        features[f"{prefix}_max"] = float(np.max(values))
        features[f"{prefix}_median"] = float(np.median(values))
        features[f"{prefix}_q25"] = float(np.quantile(values, 0.25))
        features[f"{prefix}_q75"] = float(np.quantile(values, 0.75))
        features[f"{prefix}_range"] = float(np.max(values) - np.min(values))
        features[f"{prefix}_last_minus_first"] = float(
            values[-1] - values[0]
        )
        features[f"{prefix}_auc"] = _safe_auc(values)
        features[f"{prefix}_slope"] = float(
            np.polyfit(x_axis, values, 1)[0]
        )

    for first_index, first_name in enumerate(sensor_names):
        for second_name in sensor_names[first_index + 1 :]:
            feature_name = (
                f"corr_{_safe_name(first_name)}_{_safe_name(second_name)}"
            )
            features[feature_name] = _safe_correlation(
                window[first_name].to_numpy(dtype=float),
                window[second_name].to_numpy(dtype=float),
            )

    tgs_names = [name for name in sensor_names if name.startswith("tgs")]
    if tgs_names:
        tgs_means = np.array(
            [float(window[name].mean()) for name in tgs_names],
            dtype=float,
        )
        features["tgs_array_mean"] = float(np.mean(tgs_means))
        features["tgs_array_std_between_sensors"] = float(
            np.std(tgs_means, ddof=0)
        )
        features["tgs_array_range_between_sensors"] = float(
            np.max(tgs_means) - np.min(tgs_means)
        )

    if "nh3" in sensor_names and "h2s" in sensor_names:
        features["nh3_minus_h2s_mean"] = float(
            window["nh3"].mean() - window["h2s"].mean()
        )
    return features


def _create_windows(
    normalized_frame: Any,
    sensor_names: Sequence[str],
    window_size: int,
    window_stride: int,
) -> tuple[Any, Any]:
    _, _, pd = _dependencies()
    if len(normalized_frame) < window_size:
        raise ValueError(
            f"only {len(normalized_frame)} valid rows are available; "
            f"the model requires at least {window_size}"
        )

    feature_rows: list[dict[str, float]] = []
    window_metadata: list[dict[str, Any]] = []
    for start in range(
        0,
        len(normalized_frame) - window_size + 1,
        window_stride,
    ):
        stop = start + window_size
        window = normalized_frame.iloc[start:stop]
        feature_rows.append(_extract_features(window, sensor_names))
        metadata: dict[str, Any] = {
            "window_index": len(window_metadata),
            "window_start_row": start,
            "window_end_row": stop - 1,
        }
        if "timestamp_utc" in normalized_frame.columns:
            metadata["window_start_timestamp"] = str(
                normalized_frame.iloc[start]["timestamp_utc"]
            )
            metadata["window_end_timestamp"] = str(
                normalized_frame.iloc[stop - 1]["timestamp_utc"]
            )
        if "elapsed_s" in normalized_frame.columns:
            metadata["window_start_elapsed_s"] = float(
                normalized_frame.iloc[start]["elapsed_s"]
            )
            metadata["window_end_elapsed_s"] = float(
                normalized_frame.iloc[stop - 1]["elapsed_s"]
            )
        window_metadata.append(metadata)
    return pd.DataFrame(feature_rows), pd.DataFrame(window_metadata)


def _predict_task(
    task_name: str,
    package: Mapping[str, Any],
    feature_frame: Any,
) -> tuple[dict[str, Any], Any]:
    _, np, pd = _dependencies()
    model = package["model"]
    feature_columns = list(package["feature_columns"])
    missing = [
        feature for feature in feature_columns if feature not in feature_frame
    ]
    if missing:
        raise ValueError(
            f"features for {task_name} are missing:\n" + "\n".join(missing)
        )
    model_input = feature_frame.reindex(columns=feature_columns)
    predicted_labels = model.predict(model_input)
    prediction_table = pd.DataFrame(
        {f"{task_name}_prediction": predicted_labels}
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)
        class_names = [str(value) for value in model.classes_]
        mean_probabilities = probabilities.mean(axis=0)
        best_index = int(np.argmax(mean_probabilities))
        overall_label = class_names[best_index]
        confidence = float(mean_probabilities[best_index])
        class_scores = {
            name: float(mean_probabilities[index])
            for index, name in enumerate(class_names)
        }
        for index, class_name in enumerate(class_names):
            prediction_table[
                f"{task_name}_probability_{_safe_name(class_name)}"
            ] = probabilities[:, index]
        aggregation_method = "mean_window_probability"
    else:
        counts = Counter(str(value) for value in predicted_labels)
        overall_label, winning_count = counts.most_common(1)[0]
        confidence = winning_count / len(predicted_labels)
        class_scores = {
            name: count / len(predicted_labels)
            for name, count in counts.items()
        }
        aggregation_method = "majority_window_vote"

    return (
        {
            "task": task_name,
            "model_name": str(package["model_name"]),
            "overall_prediction": overall_label,
            "confidence": confidence,
            "aggregation_method": aggregation_method,
            "window_count": len(predicted_labels),
            "class_scores": class_scores,
        },
        prediction_table,
    )


class FoodFreshnessClassifier:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
        joblib, _, _ = _dependencies()
        self.artifact_dir = Path(artifact_dir)
        metadata_path = (
            self.artifact_dir / "dataset_and_preprocessing_metadata.json"
        )
        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"cannot read classification metadata: {metadata_path}"
            ) from exc

        self.selected_channels = dict(self.metadata["selected_channels"])
        self.baseline_means = {
            name: float(value)
            for name, value in self.metadata["baseline_means"].items()
        }
        self.baseline_scales = {
            name: float(value)
            for name, value in self.metadata["baseline_scales"].items()
        }
        self.models: dict[str, Mapping[str, Any]] = {}
        for task_name, filename in MODEL_FILENAMES.items():
            model_path = self.artifact_dir / filename
            try:
                package = joblib.load(model_path)
            except Exception as exc:
                raise RuntimeError(
                    f"cannot load classification model: {model_path}"
                ) from exc
            required = {
                "task_name",
                "model_name",
                "model",
                "feature_columns",
                "classes",
                "window_size",
                "window_stride",
            }
            missing = required - set(package)
            if missing:
                raise ValueError(
                    f"{filename} is missing required keys: {sorted(missing)}"
                )
            if package["task_name"] != task_name:
                raise ValueError(
                    f"{filename} contains task {package['task_name']!r}, "
                    f"expected {task_name!r}"
                )
            self.models[task_name] = package

        window_sizes = {
            int(package["window_size"]) for package in self.models.values()
        }
        window_strides = {
            int(package["window_stride"]) for package in self.models.values()
        }
        if len(window_sizes) != 1 or len(window_strides) != 1:
            raise ValueError("classification models use inconsistent windows")
        self.model_window_size = window_sizes.pop()
        self.model_window_stride = window_strides.pop()
        if int(self.metadata["window_size"]) != self.model_window_size:
            raise ValueError("metadata and models use different window sizes")
        if int(self.metadata["window_stride"]) != self.model_window_stride:
            raise ValueError("metadata and models use different window strides")

    def classify_rows(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[dict[str, Any], Any]:
        _, _, pd = _dependencies()
        raw_frame = pd.DataFrame(rows)
        if raw_frame.empty:
            raise ValueError("classification input is empty")
        if "timestamp_utc" in raw_frame:
            raw_frame["timestamp_utc"] = pd.to_datetime(
                raw_frame["timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            raw_frame = raw_frame.sort_values("timestamp_utc").reset_index(
                drop=True
            )
        cleaned = _prepare_sensor_frame(raw_frame, self.selected_channels)
        normalized = _normalize_sensor_frame(
            cleaned,
            self.baseline_means,
            self.baseline_scales,
        )
        feature_frame, window_metadata = _create_windows(
            normalized,
            list(self.selected_channels),
            self.model_window_size,
            self.model_window_stride,
        )

        combined_output = window_metadata.copy()
        predictions: dict[str, dict[str, Any]] = {}
        for task_name in TASK_ORDER:
            summary, task_output = _predict_task(
                task_name,
                self.models[task_name],
                feature_frame,
            )
            predictions[task_name] = summary
            combined_output = pd.concat(
                [combined_output, task_output],
                axis=1,
            )
        return (
            {
                "raw_rows": len(raw_frame),
                "valid_rows": len(cleaned),
                "window_count": len(feature_frame),
                "selected_channels": self.selected_channels,
                "predictions": predictions,
            },
            combined_output,
        )

    def classify_csv(self, path: Path) -> tuple[dict[str, Any], Any]:
        _, _, pd = _dependencies()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"cannot find acquisition CSV: {path}")
        raw_frame = pd.read_csv(path)
        result, windows = self.classify_rows(
            raw_frame.to_dict(orient="records")
        )
        result["test_csv"] = str(path.resolve())
        return result, windows


class RowClassifier(Protocol):
    def classify_rows(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[dict[str, Any], Any]: ...


class SlidingWindowClassifier:
    def __init__(
        self,
        classifier: RowClassifier,
        window_rows: int = 60,
        update_rows: int = 10,
    ) -> None:
        if window_rows < 20:
            raise ValueError("window_rows must be at least 20")
        if update_rows < 1 or update_rows > window_rows:
            raise ValueError("update_rows must be between 1 and window_rows")
        self.classifier = classifier
        self.window_rows = window_rows
        self.update_rows = update_rows
        self._rows: deque[dict[str, object]] = deque(maxlen=window_rows)
        self._received = 0

    def add_frame(self, frame: Frame) -> dict[str, Any] | None:
        self._rows.append(frame_to_row(frame))
        self._received += 1
        if len(self._rows) < self.window_rows:
            return None
        if (self._received - self.window_rows) % self.update_rows:
            return None
        result, _ = self.classifier.classify_rows(tuple(self._rows))
        return result

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from enose.classification import (
    FoodFreshnessClassifier,
    SlidingWindowClassifier,
)
from enose.csv_logger import NO_SGP41_BME690_SHT45_CSV_COLUMNS
from enose.records import Frame


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "food_freshness"
EXPECTED_HASHES = {
    "freshness_best_model.joblib": (
        "eec2edf311a853994b8b47e56c4cfe47e4556b8d2d61fdc2156ad816fa1bb6cf"
    ),
    "food_type_best_model.joblib": (
        "5f9c42baf247d4bd4451ebfbc17805fb0b335b63bd88ae6dec7900eb00cf1d6b"
    ),
    "combined_class_best_model.joblib": (
        "c81b92d89224d1e3c77baa110efd5576953475cabc1646b03df2cf99a6586886"
    ),
}


def test_classification_metadata_matches_reduced_acquisition_csv() -> None:
    metadata = json.loads(
        (MODEL_DIR / "dataset_and_preprocessing_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["window_size"] == 20
    assert metadata["window_stride"] == 5
    assert set(metadata["selected_channels"].values()).issubset(
        NO_SGP41_BME690_SHT45_CSV_COLUMNS
    )
    assert set(metadata["baseline_means"]) == set(
        metadata["selected_channels"]
    )
    assert set(metadata["baseline_scales"]) == set(
        metadata["selected_channels"]
    )
    assert all(value > 0 for value in metadata["baseline_scales"].values())


def test_supplied_model_artifacts_are_preserved_exactly() -> None:
    for filename, expected_hash in EXPECTED_HASHES.items():
        digest = hashlib.sha256((MODEL_DIR / filename).read_bytes()).hexdigest()
        assert digest == expected_hash


def test_supplied_models_classify_a_60_row_input() -> None:
    try:
        __import__("joblib")
        __import__("pandas")
        __import__("sklearn")
    except ImportError as exc:
        pytest.skip(f"optional classification runtime unavailable: {exc}")
    metadata = json.loads(
        (MODEL_DIR / "dataset_and_preprocessing_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for sequence in range(60):
        row: dict[str, object] = {
            "elapsed_s": float(sequence),
            "ads7828_ok": True,
            "nh3_ok": True,
            "h2s_ok": True,
        }
        for sensor_name, csv_column in metadata["selected_channels"].items():
            mean = metadata["baseline_means"][sensor_name]
            scale = metadata["baseline_scales"][sensor_name]
            row[csv_column] = mean + scale * ((sequence % 7) - 3) / 10
        rows.append(row)

    result, window_predictions = FoodFreshnessClassifier(
        MODEL_DIR
    ).classify_rows(rows)

    assert result["raw_rows"] == 60
    assert result["valid_rows"] == 60
    assert result["window_count"] == 9
    assert set(result["predictions"]) == {
        "food_type",
        "freshness",
        "combined_class",
    }
    assert len(window_predictions) == 9
    for prediction in result["predictions"].values():
        assert 0.0 <= prediction["confidence"] <= 1.0


def _frame(sequence: int) -> Frame:
    return Frame(
        timestamp_utc=f"2026-01-01T00:00:{sequence:02d}.000Z",
        elapsed_s=float(sequence),
        sequence=sequence,
        frame_duration_ms=1.0,
        deadline_miss_ms=0.0,
        sht45=None,
        ads7828=None,
        nh3=None,
        h2s=None,
        sgp41=None,
        bme690=None,
    )


def test_live_classification_uses_latest_60_rows_every_10_frames() -> None:
    class FakeClassifier:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, object], ...]] = []

        def classify_rows(self, rows):
            captured = tuple(dict(row) for row in rows)
            self.calls.append(captured)
            return {"call": len(self.calls)}, None

    classifier = FakeClassifier()
    window = SlidingWindowClassifier(
        classifier,
        window_rows=60,
        update_rows=10,
    )
    results = [window.add_frame(_frame(sequence)) for sequence in range(80)]

    assert [index for index, result in enumerate(results) if result] == [
        59,
        69,
        79,
    ]
    assert len(classifier.calls) == 3
    assert [row["sequence"] for row in classifier.calls[0]] == list(range(60))
    assert [row["sequence"] for row in classifier.calls[1]] == list(
        range(10, 70)
    )
    assert classifier.calls[0][10:] == classifier.calls[1][:50]

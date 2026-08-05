"""Transparent rule-based classifier for the current ECE450 food demo.

This replaces model inference for the demonstrated four classes. It operates
on raw ADS7828 codes and is not a calibrated gas or food-safety classifier.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .records import Frame


MIN_ELAPSED_S = 60.0
DEFAULT_WINDOW_ROWS = 10
DEFAULT_CONFIRMATIONS = 5


@dataclass(frozen=True)
class RulePrediction:
    combined_class: str
    food_type: str
    freshness: str
    status: str
    reason: str


def classify_averages(tgs2603: float, tgs2620: float, tgs2602: float) -> RulePrediction:
    """Classify last-window raw averages using the eight-recording demo rules."""
    if tgs2603 < 2150:
        return RulePrediction("fresh_meat", "meat", "fresh", "DEMO_RULE", "TGS2603 < 2150")
    if tgs2603 < 2250:
        return RulePrediction("uncertain_fresh", "unknown", "uncertain", "DEMO_RULE_UNCERTAIN",
                              "TGS2603 is in the 2150--2250 fresh-class boundary")
    if tgs2603 < 3000:
        return RulePrediction("fresh_banana", "banana", "fresh", "DEMO_RULE",
                              "2250 <= TGS2603 < 3000")
    if tgs2620 >= 3800 and tgs2602 >= 3900:
        return RulePrediction("spoiled_meat", "meat", "spoiled", "DEMO_RULE",
                              "TGS2620 >= 3800 and TGS2602 >= 3900")
    if tgs2620 < 3800:
        return RulePrediction("fermented_banana", "banana", "fermented", "DEMO_RULE",
                              "TGS2603 >= 3000 and TGS2620 < 3800")
    return RulePrediction("uncertain", "unknown", "uncertain", "DEMO_RULE_UNCERTAIN",
                          "Readings are outside the four demonstrated regions")


def _final(prediction: RulePrediction) -> bool:
    return prediction.status == "DEMO_RULE"


class FoodFreshnessClassifier:
    """Compatibility wrapper retained for the acquire-classify command."""

    def predict(self, tgs2603: float, tgs2620: float, tgs2602: float) -> RulePrediction:
        return classify_averages(tgs2603, tgs2620, tgs2602)


class SlidingWindowClassifier:
    """Wait 60 s, average 10 rows, then require five matching rule outputs."""

    def __init__(self, classifier: FoodFreshnessClassifier, *, window_rows: int = DEFAULT_WINDOW_ROWS,
                 update_rows: int = 1, min_elapsed_s: float = MIN_ELAPSED_S,
                 confirmations: int = DEFAULT_CONFIRMATIONS) -> None:
        if window_rows < 1 or update_rows < 1 or confirmations < 1 or min_elapsed_s < 0:
            raise ValueError("window_rows, update_rows, and confirmations must be positive")
        self._classifier = classifier
        self._window: deque[tuple[float, float, float]] = deque(maxlen=window_rows)
        self._history: deque[RulePrediction] = deque(maxlen=confirmations)
        self._window_rows = window_rows
        self._update_rows = update_rows
        self._min_elapsed_s = min_elapsed_s
        self._confirmations = confirmations

    def add_frame(self, frame: Frame) -> dict[str, Any] | None:
        if frame.ads7828 is None:
            return None
        readings = frame.ads7828.by_sensor()
        required = ("tgs2603", "tgs2620", "tgs2602")
        if any(name not in readings for name in required):
            return None
        self._window.append(tuple(float(readings[name].raw) for name in required))
        if frame.elapsed_s < self._min_elapsed_s or len(self._window) < self._window_rows:
            return None
        if (frame.sequence + 1) % self._update_rows != 0:
            return None

        averages = tuple(sum(row[index] for row in self._window) / len(self._window) for index in range(3))
        prediction = self._classifier.predict(*averages)
        self._history.append(prediction)
        confirmed = (_final(prediction) and len(self._history) == self._confirmations
                     and all(item.combined_class == prediction.combined_class for item in self._history))
        status = prediction.status if confirmed or not _final(prediction) else "DEMO_RULE_STABILIZING"
        return {
            "raw_rows": self._window_rows,
            "valid_rows": self._window_rows,
            "window_count": 1,
            "confirmed": confirmed,
            "rule_status": status,
            "rule_reason": prediction.reason,
            "sensor_averages": {"tgs2603_raw": averages[0], "tgs2620_raw": averages[1], "tgs2602_raw": averages[2]},
            "predictions": {
                "food_type": {"overall_prediction": prediction.food_type, "confidence": 0.0},
                "freshness": {"overall_prediction": prediction.freshness, "confidence": 0.0},
                "combined_class": {"overall_prediction": prediction.combined_class, "confidence": 0.0},
            },
        }

from __future__ import annotations

from enose.classification import FoodFreshnessClassifier, SlidingWindowClassifier, classify_averages
from enose.records import ADS7828Reading, ADS7828Sample, Frame


def test_eight_stable_period_averages_match_the_rule() -> None:
    cases = (
        (1761.7, 2934.1, 2943.9, "fresh_meat"),
        (2110.0, 3189.5, 3246.1, "fresh_meat"),
        (2391.4, 3015.7, 3045.7, "fresh_banana"),
        (2337.7, 3153.2, 3170.1, "fresh_banana"),
        (3968.5, 3631.8, 3833.8, "fermented_banana"),
        (3297.6, 3421.1, 3543.7, "fermented_banana"),
        (4043.2, 4068.4, 3976.3, "spoiled_meat"),
        (4095.0, 4053.6, 3986.8, "spoiled_meat"),
    )
    for tgs2603, tgs2620, tgs2602, expected in cases:
        assert classify_averages(tgs2603, tgs2620, tgs2602).combined_class == expected


def test_fresh_boundary_is_uncertain() -> None:
    assert classify_averages(2200, 3100, 3100).combined_class == "uncertain_fresh"


def _frame(sequence: int, elapsed_s: float) -> Frame:
    readings = tuple(
        ADS7828Reading(sensor=name, channel=index, raw=raw, voltage_v=0.0, saturated=False)
        for index, (name, raw) in enumerate((("tgs2603", 2390), ("tgs2620", 3020), ("tgs2602", 3050)))
    )
    return Frame("2026-08-05T00:00:00Z", elapsed_s, sequence, 1.0, 0.0, None,
                 ADS7828Sample(readings), None, None, None, None)


def test_live_rule_waits_then_requires_five_matching_outputs() -> None:
    classifier = SlidingWindowClassifier(FoodFreshnessClassifier(), window_rows=10, update_rows=1,
                                         min_elapsed_s=60.0, confirmations=5)
    for sequence in range(59):
        assert classifier.add_frame(_frame(sequence, float(sequence + 1))) is None
    results = [classifier.add_frame(_frame(sequence, float(sequence + 1))) for sequence in range(59, 64)]
    assert all(result is not None for result in results)
    assert results[0]["rule_status"] == "DEMO_RULE_STABILIZING"
    assert results[-1]["confirmed"] is True
    assert results[-1]["predictions"]["combined_class"]["overall_prediction"] == "fresh_banana"

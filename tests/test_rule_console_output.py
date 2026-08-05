from __future__ import annotations

from enose.cli import _print_classification
from enose.records import Frame


def test_rule_console_output_does_not_claim_model_confidence(capsys) -> None:
    frame = Frame("2026-08-05T00:00:00Z", 64.0, 63, 1.0, 0.0, None, None, None, None, None, None)
    result = {
        "rule_status": "DEMO_RULE",
        "confirmed": True,
        "sensor_averages": {"tgs2603_raw": 1918.0, "tgs2620_raw": 3362.0, "tgs2602_raw": 3390.0},
        "predictions": {
            "food_type": {"overall_prediction": "meat", "confidence": 0.0},
            "freshness": {"overall_prediction": "fresh", "confidence": 0.0},
            "combined_class": {"overall_prediction": "fresh_meat", "confidence": 0.0},
        },
    }
    _print_classification(frame, result)
    output = capsys.readouterr().out
    assert "mode=hard_code" in output
    assert "confirmed=True" in output
    assert "confidence" not in output

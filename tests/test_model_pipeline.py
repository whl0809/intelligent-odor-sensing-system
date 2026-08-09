from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "model" / "enose_multitask.py"
TEST_SCRIPT = ROOT / "model" / "test_model.py"
TRAINING_DATA = ROOT / "data" / "training"
TEST_DATA = ROOT / "data" / "test"


def test_expected_training_and_test_recordings_are_present() -> None:
    training_files = sorted(TRAINING_DATA.glob("*.csv"))
    test_files = sorted(TEST_DATA.glob("*.csv"))

    assert len(training_files) == 15
    assert len(test_files) == 3
    assert {path.stem.rsplit("_", 1)[0] for path in training_files} == {
        "enose_blank",
        "enose_fermented_banana",
        "enose_fresh_banana",
        "enose_fresh_meat",
        "enose_spoiled_meat",
    }


def test_train_then_classify_included_recording(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "train",
            "--data-dir",
            str(TRAINING_DATA),
            "--output-dir",
            str(artifact_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    model_path = artifact_dir / "enose_multitask_model.joblib"
    output_path = tmp_path / "prediction.json"
    subprocess.run(
        [
            sys.executable,
            str(TEST_SCRIPT),
            "--model",
            str(model_path),
            "--input-csv",
            str(TEST_DATA / "test1.csv"),
            "--output-json",
            str(output_path),
            "--no-figures",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["model_format_version"] == 2
    assert result["frames"]["windows"] > 0
    assert result["food_type"] in {"banana", "meat"}
    assert result["freshness_level"] in {"fresh", "fermented", "spoiled"}
    assert result["state"] in {
        "banana_fermented",
        "banana_fresh",
        "meat_fresh",
        "meat_spoiled",
    }


def test_live_pipeline_uses_external_unified_acquisition_command() -> None:
    sys.path.insert(0, str(ROOT / "pipeline"))
    from run_model_pipeline import acquisition_command

    command = acquisition_command(
        {"config": Path("/acquisition/config/rpi5.toml")},
        frames=120,
        uart_device="/dev/ttyUSB0",
    )

    assert command[3] == "acquire"
    assert command[command.index("--sensors") + 1] == "tgs,svm41"
    assert "acquire-tgs-svm41" not in command

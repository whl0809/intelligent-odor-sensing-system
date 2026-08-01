#!/usr/bin/env python3
"""Automatic TGS + SVM41 acquisition, classification, and CO5300 display.

Default workflow:

1. Wait for non-zero SVM41 gas indices, then collect 60 clean-air frames.
2. Wait for the operator to insert a food sample.
3. Collect 90 test frames from the six TGS sensors and SVM41.
4. Run ``model/test_model.py`` with session-baseline adaptation.
5. Write ``runtime/display_state.json`` and optionally show it on CO5300.

No EC Sense NH3/H2S device is initialized or used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_FRAMES = 90
DEFAULT_BASELINE_FRAMES = 60
MINIMUM_ROWS = 20
SENSOR_AVERAGE_ROWS = 10


def default_project_root() -> Path:
    script = Path(__file__).resolve()
    return script.parent.parent if script.parent.name == "tools" else script.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect six TGS + SVM41 data, classify food/freshness, and "
            "optionally display the result on CO5300."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
        help="enose_tgs_svm41_v2 pipeline root",
    )
    parser.add_argument(
        "--hardware-root",
        type=Path,
        help=(
            "ECE450_software root containing the CO5300 tools/config; "
            "default is the parent of --project-root"
        ),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        help=(
            "trained model directory; default: "
            "<project-root>/models/food_freshness"
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help="test frames to acquire (default: 90)",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=DEFAULT_BASELINE_FRAMES,
        help="clean-air frames to acquire (default: 60)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds between acquisition frames (default: 1.0)",
    )
    parser.add_argument(
        "--svm41-port",
        default="auto",
        help="SVM41 serial port or auto (default: auto)",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="classify an existing sample CSV instead of acquiring one",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        help="existing clean-air CSV; required with --input-csv",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip clean-air and sample-ready prompts",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not access the CO5300 display",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="draw the display once and exit",
    )
    parser.add_argument(
        "--skip-te-check",
        action="store_true",
        help="skip the CO5300 TE edge check",
    )
    return parser


def resolve_paths(
    project_root: Path,
    hardware_root: Path | None = None,
    model_output_dir: Path | None = None,
) -> dict[str, Path]:
    root = project_root.expanduser().resolve()
    hardware = (
        hardware_root.expanduser().resolve()
        if hardware_root is not None
        else root.parent
    )
    model_dir = root / "model"
    classification_dir = (
        model_output_dir.expanduser().resolve()
        if model_output_dir is not None
        else root / "models" / "food_freshness"
    )
    runtime_dir = root / "runtime"
    return {
        "root": root,
        "collector": root / "tools" / "collect_all_sensors.py",
        "test_script": model_dir / "test_model.py",
        "classification": classification_dir,
        "metadata": classification_dir / "training_metadata.json",
        "food_model": classification_dir / "models" / "food_group_model.joblib",
        "odor_model": classification_dir / "models" / "odor_state_model.joblib",
        "baseline_csv": root / "data" / "session" / "enose_session_baseline.csv",
        "test_csv": root / "data" / "session" / "enose_test.csv",
        "prediction_json": classification_dir / "test_results" / "session_prediction.json",
        "display_state": runtime_dir / "display_state.json",
        "dashboard": hardware / "tools" / "co5300_dashboard.py",
        "display_init": hardware / "config" / "co5300_init.json",
        "mpl_cache": runtime_dir / "matplotlib",
    }


def require_files(paths: dict[str, Path], acquisition: bool, display: bool) -> None:
    required = {
        "model test script": paths["test_script"],
        "training metadata": paths["metadata"],
        "food-group model": paths["food_model"],
        "odor-state model": paths["odor_model"],
    }
    if acquisition:
        required["TGS+SVM41 collector"] = paths["collector"]
    if display:
        required["CO5300 dashboard"] = paths["dashboard"]
        required["CO5300 init file"] = paths["display_init"]
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) are missing:\n" + "\n".join(missing))


def process_environment(paths: dict[str, Path]) -> dict[str, str]:
    environment = os.environ.copy()
    source_dir = paths["root"] / "src"
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(source_dir) if not existing else str(source_dir) + os.pathsep + existing
    )
    paths["mpl_cache"].mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(paths["mpl_cache"])
    return environment


def run_command(
    command: list[str], paths: dict[str, Path], label: str
) -> None:
    print(f"\n--- {label} ---")
    print(" ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=paths["root"],
        env=process_environment(paths),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def acquire(
    paths: dict[str, Path],
    destination: Path,
    frames: int,
    interval: float,
    svm41_port: str,
    food_group: str,
) -> Path:
    if frames < MINIMUM_ROWS:
        raise ValueError(
            f"At least {MINIMUM_ROWS} frames are required by the model window"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(paths["collector"]),
        "--samples",
        str(frames),
        "--interval",
        str(interval),
        "--svm41-port",
        svm41_port,
        "--csv",
        str(destination),
        "--food-group",
        food_group,
        "--overwrite",
    ]
    run_command(command, paths, f"Acquire {food_group} recording")
    if not destination.is_file():
        raise FileNotFoundError(f"Collector did not create {destination}")
    return destination


def run_model(
    paths: dict[str, Path], baseline_csv: Path, input_csv: Path
) -> dict[str, Any]:
    paths["prediction_json"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(paths["test_script"]),
        "--baseline-csv",
        str(baseline_csv),
        "--input-csv",
        str(input_csv),
        "--model-output-dir",
        str(paths["classification"]),
        "--output-json",
        str(paths["prediction_json"]),
        "--no-figures",
    ]
    run_command(command, paths, "Run TGS+SVM41 classifier")
    if not paths["prediction_json"].is_file():
        raise FileNotFoundError(
            f"Prediction JSON was not created: {paths['prediction_json']}"
        )
    payload = json.loads(paths["prediction_json"].read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "predictions" not in payload:
        raise ValueError("Prediction JSON does not contain one test result")
    return payload


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "ok"}


def mean_column(rows: list[dict[str, str]], name: str) -> float | None:
    values = [numeric(row.get(name)) for row in rows]
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def sensor_summary(csv_path: Path) -> dict[str, float | None]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No sensor rows in {csv_path}")
    valid = []
    for row in rows:
        voc = numeric(row.get("svm41_voc_index"))
        nox = numeric(row.get("svm41_nox_index"))
        if (
            ("svm41_ok" not in row or truthy(row["svm41_ok"]))
            and voc is not None
            and nox is not None
            and voc > 0.0
            and nox > 0.0
        ):
            valid.append(row)
    selected = (valid or rows)[-SENSOR_AVERAGE_ROWS:]
    return {
        "temperature_c": mean_column(selected, "svm41_temperature_c"),
        "humidity_rh": mean_column(selected, "svm41_relative_humidity_pct"),
        "voc_index": mean_column(selected, "svm41_voc_index"),
        "nox_index": mean_column(selected, "svm41_nox_index"),
    }


def task_prediction(payload: dict[str, Any], task: str) -> dict[str, Any]:
    prediction = payload.get("predictions", {}).get(task)
    if not isinstance(prediction, dict):
        raise ValueError(f"Prediction JSON is missing task {task}")
    return prediction


def build_display_state(
    payload: dict[str, Any], summary: dict[str, float | None]
) -> dict[str, Any]:
    food = task_prediction(payload, "food_group")
    odor = task_prediction(payload, "odor_state")
    food_label = str(food["overall_prediction"])
    if food_label == "fruit":
        freshness = task_prediction(payload, "fruit_freshness")
        implied = (
            "fresh_banana"
            if freshness["overall_prediction"] == "fresh"
            else "fermented_banana"
        )
    elif food_label == "meat":
        freshness = task_prediction(payload, "meat_freshness")
        implied = (
            "fresh_meat"
            if freshness["overall_prediction"] == "fresh"
            else "spoiled_meat"
        )
    else:
        freshness = {
            "overall_prediction": "not_applicable",
            "confidence": 1.0,
        }
        implied = "blank"

    confidence = min(
        float(food.get("confidence", 0.0)),
        float(freshness.get("confidence", 0.0)),
        float(odor.get("confidence", 0.0)),
    )
    consistent = str(odor["overall_prediction"]) == implied
    return {
        "food_type": food_label.replace("_", " ").title(),
        "freshness_level": str(freshness["overall_prediction"])
        .replace("_", " ")
        .title(),
        "confidence": confidence,
        "temperature_c": summary["temperature_c"],
        "humidity_rh": summary["humidity_rh"],
        "voc_raw": summary["voc_index"],
        "nox_raw": summary["nox_index"],
        "system_status": "OK" if consistent else "WARNING",
        "combined_class": odor["overall_prediction"],
        "food_confidence": food.get("confidence", 0.0),
        "freshness_confidence": freshness.get("confidence", 0.0),
        "combined_confidence": odor.get("confidence", 0.0),
        "sensor_scope": "6 TGS + SVM41",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run_dashboard(
    paths: dict[str, Path], hold: bool, skip_te_check: bool
) -> None:
    command = [
        sys.executable,
        str(paths["dashboard"]),
        "--state-file",
        str(paths["display_state"]),
        "--init",
        str(paths["display_init"]),
        "--gpiochip",
        "auto",
        "--clk",
        "21",
        "--sio0",
        "20",
        "--sio1",
        "19",
        "--sio2",
        "16",
        "--sio3",
        "26",
        "--cs",
        "18",
        "--rst",
        "25",
        "--te",
        "24",
        "--half-period-us",
        "5",
        "--chunk-bytes",
        "1024",
        "--once",
    ]
    if hold:
        command.append("--hold")
    if skip_te_check:
        command.append("--skip-te-check")
    run_command(command, paths, "Update CO5300 display")


def print_result(payload: dict[str, Any], input_csv: Path) -> None:
    food = task_prediction(payload, "food_group")
    odor = task_prediction(payload, "odor_state")
    group = str(food["overall_prediction"])
    freshness_task = "meat_freshness" if group == "meat" else "fruit_freshness"
    freshness = (
        task_prediction(payload, freshness_task)
        if group in {"fruit", "meat"}
        else {"overall_prediction": "not_applicable", "confidence": 1.0}
    )
    print("\n" + "=" * 72)
    print("FINAL E-NOSE RESULT — 6 TGS + SVM41")
    print("=" * 72)
    print(f"CSV file       : {input_csv}")
    print(
        f"Food type      : {food['overall_prediction']} "
        f"({float(food.get('confidence', 0.0)):.1%})"
    )
    print(
        f"Freshness      : {freshness['overall_prediction']} "
        f"({float(freshness.get('confidence', 0.0)):.1%})"
    )
    print(
        f"Combined class : {odor['overall_prediction']} "
        f"({float(odor.get('confidence', 0.0)):.1%})"
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.frames < MINIMUM_ROWS or args.baseline_frames < MINIMUM_ROWS:
        raise ValueError(f"--frames and --baseline-frames must be at least {MINIMUM_ROWS}")
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    if args.input_csv is not None and args.baseline_csv is None:
        raise ValueError("--baseline-csv is required with --input-csv")

    paths = resolve_paths(
        args.project_root, args.hardware_root, args.model_output_dir
    )
    acquisition = args.input_csv is None
    require_files(paths, acquisition=acquisition, display=not args.no_display)

    try:
        if acquisition:
            if not args.yes:
                input(
                    "\nPrepare clean air in the chamber, then press Enter "
                    "to collect the session baseline..."
                )
            baseline_csv = acquire(
                paths,
                paths["baseline_csv"],
                args.baseline_frames,
                args.interval,
                args.svm41_port,
                "blank",
            )
            if not args.yes:
                input(
                    "\nInsert the food sample, then press Enter to collect "
                    "the test recording..."
                )
            input_csv = acquire(
                paths,
                paths["test_csv"],
                args.frames,
                args.interval,
                args.svm41_port,
                "unknown",
            )
        else:
            baseline_csv = args.baseline_csv.expanduser().resolve()
            input_csv = args.input_csv.expanduser().resolve()
            if not baseline_csv.is_file() or not input_csv.is_file():
                raise FileNotFoundError("The supplied baseline or input CSV does not exist")

        payload = run_model(paths, baseline_csv, input_csv)
        summary = sensor_summary(input_csv)
        state = build_display_state(payload, summary)
        write_json_atomic(paths["display_state"], state)
        print_result(payload, input_csv)
        print(f"Display state   : {paths['display_state']}")

        if not args.no_display:
            run_dashboard(
                paths,
                hold=not args.no_hold,
                skip_te_check=args.skip_te_check,
            )
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except Exception as exc:
        error_state = {
            "food_type": "Unknown",
            "freshness_level": "Unknown",
            "confidence": 0.0,
            "temperature_c": None,
            "humidity_rh": None,
            "voc_raw": None,
            "nox_raw": None,
            "system_status": "ERROR",
            "sensor_scope": "6 TGS + SVM41",
            "error_message": str(exc),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            write_json_atomic(paths["display_state"], error_state)
        except Exception:
            pass
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

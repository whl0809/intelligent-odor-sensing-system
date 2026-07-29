#!/usr/bin/env python3
"""
Automatic baseline-adaptive E-nose pipeline for Raspberry Pi 5:

1. Collect a clean-air baseline immediately before the test.
2. Ask the operator to insert the sample.
3. Collect the sample recording.
4. Update the normalization center/scale from the new baseline.
5. Run the frozen hierarchical classifiers.
6. Write runtime/display_state.json and update the CO5300 screen.

Recommended location:
    ECE450_software/tools/enose_auto_classify_display.py

Expected repository structure:
    ECE450_software/
    ├── config/
    │   ├── rpi5.toml
    │   └── co5300_init.json
    ├── src/enose/
    ├── tools/
    │   ├── co5300_dashboard.py
    │   ├── co5300_qspi_test.py
    │   └── enose_auto_classify_display.py
    └── model/
        ├── test_model.py
        └── classification_outputs/
            ├── training_metadata.json
            └── models/
                ├── food_group_model.joblib
                ├── fruit_freshness_model.joblib
                ├── meat_freshness_model.joblib
                └── odor_state_model.joblib

Default run:
    cd /home/pi/Documents/ECE450_software
    sudo python3 tools/enose_auto_classify_display.py

Useful alternatives:
    # Collect 120 samples instead of the default 90
    sudo python3 tools/enose_auto_classify_display.py --frames 120

    # Test with existing baseline and sample CSVs; do not touch GPIO
    python3 tools/enose_auto_classify_display.py \
        --baseline-csv data/raw/session_baseline.csv \
        --input-csv data/raw/example.csv \
        --no-display

The default acquisition mode intentionally disables SGP41, BME690 and SHT45,
because the current trained models use the TGS array plus NH3/H2S only.
Use --full-acquisition only when all configured sensors are connected and work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError as exc:
    raise SystemExit("Python 3.11 or newer is required.") from exc


DEFAULT_FRAMES = 90
MINIMUM_TEST_ROWS = 20
SENSOR_AVERAGE_ROWS = 10


def default_project_root() -> Path:
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "tools":
        return script_path.parent.parent
    return script_path.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect E-nose data, classify food/freshness, "
            "and display the result on CO5300."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
        help="ECE450_software repository root.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help="Number of 1 Hz sensor frames to collect.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help=(
            "Skip hardware acquisition and classify this existing CSV. "
            "Useful for software testing."
        ),
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        help=(
            "Clean-air baseline recorded immediately before --input-csv. "
            "Required when --input-csv is used."
        ),
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=60,
        help="Number of clean-air frames acquired before the sample test.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive clean-air and sample-ready prompts.",
    )
    parser.add_argument(
        "--full-acquisition",
        action="store_true",
        help=(
            "Use the full acquire command, including SHT45, SGP41 and BME690. "
            "By default, only TGS + NH3 + H2S are acquired."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not access the CO5300 GPIO/QSPI display.",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="Draw the result once and exit instead of holding it on-screen.",
    )
    parser.add_argument(
        "--skip-te-check",
        action="store_true",
        help="Skip the CO5300 TE edge check.",
    )
    return parser


def resolve_paths(project_root: Path) -> dict[str, Path]:
    root = project_root.expanduser().resolve()
    model_dir = root / "model"
    classification_dir = model_dir / "classification_outputs"

    return {
        "root": root,
        "config": root / "config" / "rpi5.toml",
        "display_init": root / "config" / "co5300_init.json",
        "test_script": model_dir / "test_model.py",
        "test_csv": model_dir / "enose_test.csv",
        "baseline_csv": model_dir / "enose_session_baseline.csv",
        "metadata": (
            classification_dir
            / "training_metadata.json"
        ),
        "food_model": (
            classification_dir
            / "models"
            / "food_group_model.joblib"
        ),
        "freshness_model": (
            classification_dir
            / "models"
            / "fruit_freshness_model.joblib"
        ),
        "meat_freshness_model": (
            classification_dir
            / "models"
            / "meat_freshness_model.joblib"
        ),
        "combined_model": (
            classification_dir
            / "models"
            / "odor_state_model.joblib"
        ),
        "prediction_json": (
            classification_dir
            / "test_results"
            / "session_prediction.json"
        ),
        "display_state": root / "runtime" / "display_state.json",
        "dashboard": root / "tools" / "co5300_dashboard.py",
    }


def require_files(paths: dict[str, Path], need_acquisition: bool) -> None:
    required = {
        "test script": paths["test_script"],
        "training metadata": paths["metadata"],
        "food-group model": paths["food_model"],
        "fruit-freshness model": paths["freshness_model"],
        "meat-freshness model": paths["meat_freshness_model"],
        "odor-state model": paths["combined_model"],
    }

    if need_acquisition:
        required["acquisition config"] = paths["config"]

    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required file(s) are missing:\n"
            + "\n".join(missing)
        )


def python_environment(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    source_dir = project_root / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(source_dir)
        if not existing
        else str(source_dir) + os.pathsep + existing
    )
    return env


def read_output_directory(
    config_path: Path,
    project_root: Path,
) -> Path:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    try:
        configured = Path(
            str(config["acquisition"]["output_dir"])
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Cannot find acquisition.output_dir in {config_path}"
        ) from exc

    if configured.is_absolute():
        return configured

    return (project_root / configured).resolve()


def stream_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> tuple[int, list[str]]:
    print("\nRunning command:")
    print(" ".join(command))
    print("-" * 80)

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line.rstrip("\n"))

    return_code = process.wait()
    return return_code, output_lines


def locate_created_csv(
    output_lines: list[str],
    output_dir: Path,
    before_files: set[Path],
    project_root: Path,
) -> Path:
    writing_pattern = re.compile(r"^writing\s+(.+\.csv)\s*$")

    for line in output_lines:
        match = writing_pattern.match(line.strip())
        if not match:
            continue

        candidate = Path(match.group(1))
        if not candidate.is_absolute():
            candidate = project_root / candidate

        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate

    after_files = set(output_dir.glob("enose_*.csv"))
    new_files = sorted(
        after_files - before_files,
        key=lambda path: path.stat().st_mtime,
    )

    if new_files:
        return new_files[-1].resolve()

    all_files = sorted(
        output_dir.glob("enose_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )

    if all_files:
        return all_files[-1].resolve()

    raise FileNotFoundError(
        f"Acquisition finished but no CSV was found in {output_dir}"
    )


def run_acquisition(
    paths: dict[str, Path],
    frames: int,
    full_acquisition: bool,
) -> Path:
    if frames < MINIMUM_TEST_ROWS:
        raise ValueError(
            f"--frames must be at least {MINIMUM_TEST_ROWS} "
            "because the trained model uses 20-row windows."
        )

    output_dir = read_output_directory(
        paths["config"],
        paths["root"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    before_files = set(output_dir.glob("enose_*.csv"))

    command_name = (
        "acquire"
        if full_acquisition
        else "acquire-no-sgp41-bme690-sht45"
    )

    command = [
        sys.executable,
        "-m",
        "enose",
        command_name,
        "--config",
        str(paths["config"]),
        "--frames",
        str(frames),
    ]

    return_code, output_lines = stream_process(
        command,
        cwd=paths["root"],
        env=python_environment(paths["root"]),
    )

    if return_code != 0:
        raise RuntimeError(
            f"Sensor acquisition failed with exit code {return_code}."
        )

    return locate_created_csv(
        output_lines,
        output_dir,
        before_files,
        paths["root"],
    )


def prepare_test_csv(
    source_csv: Path,
    destination_csv: Path,
) -> None:
    source = source_csv.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Input CSV does not exist: {source}"
        )

    destination_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if source != destination_csv.resolve():
        shutil.copy2(source, destination_csv)

    print(f"\nPrepared test CSV:")
    print(f"  source      : {source}")
    print(f"  model input : {destination_csv.resolve()}")


def run_test_script(paths: dict[str, Path]) -> dict[str, Any]:
    prediction_path = paths["prediction_json"]

    if prediction_path.exists():
        prediction_path.unlink()

    command = [
        sys.executable,
        str(paths["test_script"]),
        "--baseline-csv",
        str(paths["baseline_csv"]),
        "--input-csv",
        str(paths["test_csv"]),
        "--output-json",
        str(prediction_path),
        "--no-figures",
    ]
    return_code, _ = stream_process(
        command,
        cwd=paths["test_script"].parent,
        env=python_environment(paths["root"]),
    )

    if return_code != 0:
        raise RuntimeError(
            f"Model test failed with exit code {return_code}."
        )

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Prediction JSON was not generated: {prediction_path}"
        )

    with prediction_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        result = json.load(handle)

    if not isinstance(result, dict):
        raise ValueError(
            "Prediction JSON must contain one object."
        )

    return result


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if result != result:
        return None

    return result


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "ok",
    }


def mean_available(
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
    multiplier: float = 1.0,
) -> float | None:
    values: list[float] = []

    for row in rows:
        for column in columns:
            value = float_or_none(row.get(column))
            if value is not None:
                values.append(value * multiplier)
                break

    if not values:
        return None

    return sum(values) / len(values)


def load_recent_sensor_summary(
    csv_path: Path,
    count: int = SENSOR_AVERAGE_ROWS,
) -> dict[str, float | None]:
    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(
            f"Acquired CSV contains no data rows: {csv_path}"
        )

    valid_rows = [
        row
        for row in rows
        if (
            ("ads7828_ok" not in row or truthy(row["ads7828_ok"]))
            and ("nh3_ok" not in row or truthy(row["nh3_ok"]))
            and ("h2s_ok" not in row or truthy(row["h2s_ok"]))
        )
    ]

    selected_rows = (
        valid_rows[-count:]
        if valid_rows
        else rows[-count:]
    )

    return {
        "temperature_c": mean_available(
            selected_rows,
            (
                "sht45_temperature_c",
                "bme690_temperature_c",
            ),
        ),
        "humidity_rh": mean_available(
            selected_rows,
            (
                "sht45_relative_humidity_pct",
                "bme690_relative_humidity_pct",
            ),
        ),
        "voc_raw": mean_available(
            selected_rows,
            (
                "sgp41_sraw_voc",
                "sgp41_voc_index",
            ),
        ),
        "nox_raw": mean_available(
            selected_rows,
            (
                "sgp41_sraw_nox",
                "sgp41_nox_index",
            ),
        ),
        "nh3_mv": mean_available(
            selected_rows,
            ("nh3_diff_voltage_v",),
            multiplier=1000.0,
        ),
        "h2s_mv": mean_available(
            selected_rows,
            ("h2s_diff_voltage_v",),
            multiplier=1000.0,
        ),
    }


def get_task_prediction(
    result: dict[str, Any],
    task_name: str,
) -> dict[str, Any]:
    try:
        task = result["predictions"][task_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Prediction JSON is missing task '{task_name}'."
        ) from exc

    if not isinstance(task, dict):
        raise ValueError(
            f"Prediction task '{task_name}' must be an object."
        )

    return task


def build_display_state(
    prediction_result: dict[str, Any],
    sensor_summary: dict[str, float | None],
) -> dict[str, Any]:
    food = get_task_prediction(
        prediction_result,
        "food_group",
    )
    freshness = get_task_prediction(
        prediction_result,
        "fruit_freshness",
    )
    meat_freshness = get_task_prediction(
        prediction_result,
        "meat_freshness",
    )
    combined = get_task_prediction(
        prediction_result,
        "odor_state",
    )

    food_label = str(
        food["overall_prediction"]
    )
    freshness_label = str(
        freshness["overall_prediction"]
    )
    meat_freshness_label = str(
        meat_freshness["overall_prediction"]
    )
    combined_label = str(
        combined["overall_prediction"]
    )

    food_confidence = float(
        food.get("confidence", 0.0)
    )
    freshness_confidence = float(
        freshness.get("confidence", 0.0)
    )
    meat_freshness_confidence = float(
        meat_freshness.get("confidence", 0.0)
    )
    combined_confidence = float(
        combined.get("confidence", 0.0)
    )

    if food_label == "fruit":
        display_freshness = freshness_label
        implied_combined = (
            "fresh_banana"
            if freshness_label == "fresh"
            else "fermented_banana"
        )
        # Fruit decisions use all three hierarchical classifiers.
        confidence = min(
            food_confidence,
            freshness_confidence,
            combined_confidence,
        )
    elif food_label == "meat":
        display_freshness = meat_freshness_label
        implied_combined = {
            "fresh": "fresh_meat",
            "spoiled": "spoiled_meat",
        }.get(meat_freshness_label, "unknown")
        confidence = min(
            food_confidence,
            meat_freshness_confidence,
            combined_confidence,
        )
    else:
        display_freshness = "not_applicable"
        implied_combined = "blank"
        confidence = min(
            food_confidence,
            combined_confidence,
        )
    consistent = combined_label == implied_combined

    return {
        "food_type": food_label.replace("_", " ").title(),
        "freshness_level": display_freshness.replace("_", " ").title(),
        "confidence": confidence,
        "temperature_c": sensor_summary["temperature_c"],
        "humidity_rh": sensor_summary["humidity_rh"],
        "voc_raw": sensor_summary["voc_raw"],
        "nox_raw": sensor_summary["nox_raw"],
        "nh3_value": sensor_summary["nh3_mv"],
        "nh3_unit": "mV",
        "h2s_value": sensor_summary["h2s_mv"],
        "h2s_unit": "mV",
        "system_status": "OK" if consistent else "WARNING",
        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        # Extra fields are ignored by the current dashboard, but remain useful
        # when inspecting the JSON or extending the UI later.
        "combined_class": combined_label,
        "food_confidence": food_confidence,
        "freshness_confidence": (
            meat_freshness_confidence
            if food_label == "meat"
            else freshness_confidence
        ),
        "fruit_freshness_confidence": freshness_confidence,
        "meat_freshness_confidence": meat_freshness_confidence,
        "combined_confidence": combined_confidence,
    }


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )
        handle.write("\n")
        temporary_path = Path(handle.name)

    temporary_path.replace(path)


def write_error_display_state(
    paths: dict[str, Path],
    message: str,
) -> None:
    state = {
        "food_type": "Unknown",
        "freshness_level": "Unknown",
        "confidence": 0.0,
        "temperature_c": None,
        "humidity_rh": None,
        "voc_raw": None,
        "nox_raw": None,
        "nh3_value": None,
        "nh3_unit": "mV",
        "h2s_value": None,
        "h2s_unit": "mV",
        "system_status": "ERROR",
        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "error_message": message,
    }

    write_json_atomic(
        paths["display_state"],
        state,
    )


def run_dashboard(
    paths: dict[str, Path],
    *,
    hold: bool,
    skip_te_check: bool,
) -> int:
    if not paths["dashboard"].is_file():
        raise FileNotFoundError(
            f"CO5300 dashboard script is missing: "
            f"{paths['dashboard']}"
        )

    if not paths["display_init"].is_file():
        raise FileNotFoundError(
            f"CO5300 init file is missing: "
            f"{paths['display_init']}"
        )

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

    return_code, _ = stream_process(
        command,
        cwd=paths["root"],
        env=python_environment(paths["root"]),
    )

    return return_code


def print_final_result(
    prediction_result: dict[str, Any],
    acquired_csv: Path,
) -> None:
    food = get_task_prediction(
        prediction_result,
        "food_group",
    )
    freshness = get_task_prediction(
        prediction_result,
        "fruit_freshness",
    )
    meat_freshness = get_task_prediction(
        prediction_result,
        "meat_freshness",
    )
    combined = get_task_prediction(
        prediction_result,
        "odor_state",
    )

    print("\n" + "=" * 80)
    print("FINAL E-NOSE RESULT")
    print("=" * 80)
    print(f"CSV file       : {acquired_csv}")
    print(
        f"Food type      : "
        f"{food['overall_prediction']} "
        f"({float(food.get('confidence', 0.0)):.1%})"
    )
    selected_freshness = (
        meat_freshness
        if food["overall_prediction"] == "meat"
        else freshness
    )
    print(
        f"Freshness      : "
        f"{selected_freshness['overall_prediction']} "
        f"({float(selected_freshness.get('confidence', 0.0)):.1%})"
    )
    print(
        f"Combined class : "
        f"{combined['overall_prediction']} "
        f"({float(combined.get('confidence', 0.0)):.1%})"
    )


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_paths(args.project_root)

    need_acquisition = args.input_csv is None
    if args.input_csv is not None and args.baseline_csv is None:
        raise ValueError(
            "--baseline-csv is required when --input-csv is used."
        )
    require_files(
        paths,
        need_acquisition=need_acquisition,
    )

    try:
        if args.input_csv is None:
            if not args.yes:
                input(
                    "\nPrepare clean air in the chamber, then press Enter "
                    "to acquire the session baseline..."
                )
            baseline_source = run_acquisition(
                paths,
                frames=args.baseline_frames,
                full_acquisition=args.full_acquisition,
            )
            prepare_test_csv(
                baseline_source,
                paths["baseline_csv"],
            )
            if not args.yes:
                input(
                    "\nInsert the food sample, then press Enter to begin "
                    "the sample acquisition..."
                )
            source_csv = run_acquisition(
                paths,
                frames=args.frames,
                full_acquisition=args.full_acquisition,
            )
        else:
            prepare_test_csv(
                args.baseline_csv.expanduser().resolve(),
                paths["baseline_csv"],
            )
            source_csv = args.input_csv.expanduser().resolve()

        prepare_test_csv(
            source_csv,
            paths["test_csv"],
        )

        prediction_result = run_test_script(
            paths
        )

        sensor_summary = (
            load_recent_sensor_summary(
                paths["test_csv"]
            )
        )

        display_state = build_display_state(
            prediction_result,
            sensor_summary,
        )

        write_json_atomic(
            paths["display_state"],
            display_state,
        )

        print_final_result(
            prediction_result,
            paths["test_csv"],
        )

        print("\nDisplay state written to:")
        print(f"  {paths['display_state']}")

        if args.no_display:
            return 0

        dashboard_code = run_dashboard(
            paths,
            hold=not args.no_hold,
            skip_te_check=args.skip_te_check,
        )

        if dashboard_code != 0:
            raise RuntimeError(
                "CO5300 dashboard exited with "
                f"code {dashboard_code}."
            )

        return 0

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130

    except Exception as error:
        print("\nERROR")
        print("=" * 80)
        print(error)

        try:
            write_error_display_state(
                paths,
                str(error),
            )
        except Exception as state_error:
            print(
                f"Could not write display error state: "
                f"{state_error}"
            )

        if not args.no_display:
            try:
                run_dashboard(
                    paths,
                    hold=not args.no_hold,
                    skip_te_check=args.skip_te_check,
                )
            except Exception as display_error:
                print(
                    f"Could not display error on CO5300: "
                    f"{display_error}"
                )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())

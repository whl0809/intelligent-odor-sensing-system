#!/usr/bin/env python3
"""Acquire sensor data and continuously update CO5300 from the latest window.

Expected repository layout:

    ECE450_software/
    ├── src/enose/
    └── enose_tgs_svm41_v2/
        ├── config/rpi5.toml
        ├── model/test_model.py
        ├── model/artifacts/enose_multitask_model.joblib
        └── tools/

Run from ``enose_tgs_svm41_v2``:

    /usr/bin/python3 tools/enose_auto_classify_display.py

The physical sample change still requires an operator prompt. Pass ``--yes``
only when chamber/pump control already performs that transition automatically.
During sample acquisition, inference begins after warm-up plus the minimum
window length, then repeats whenever the recording prefix advances. Each live
prediction median-aggregates all currently available windows, matching the
recording-level training distribution. Sensor data
is collected by ``python -m enose acquire-tgs-svm41``: ADS7828/TGS over I2C
and SVM41 over UART. The standalone collect_all_sensors.py is not used. The
dashboard runs continuously in the background so every sliding-window result
is shown immediately instead of only showing the final prediction.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

try:
    import tomllib
except ImportError as exc:
    raise SystemExit("Python 3.11 or newer is required.") from exc


DEFAULT_BASELINE_FRAMES = 120
DEFAULT_SAMPLE_FRAMES = 120
SUMMARY_ROWS = 10
DEFAULT_UART = "auto"


def default_project_root() -> Path:
    script = Path(__file__).resolve()
    return script.parent.parent if script.parent.name == "tools" else script.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    parser.add_argument("--baseline-frames", type=int, default=DEFAULT_BASELINE_FRAMES)
    parser.add_argument(
        "--uart",
        default=DEFAULT_UART,
        help=(
            "SVM41 UART device; default 'auto' searches /dev/serial/by-id, "
            "/dev/ttyUSB*, and /dev/ttyACM*"
        ),
    )
    parser.add_argument(
        "--prediction-step",
        type=int,
        help="New valid frames between live predictions; default: model training step.",
    )
    parser.add_argument("--input-csv", type=Path, help="Use an existing sample CSV")
    parser.add_argument("--baseline-csv", type=Path, help="Use an existing baseline CSV")
    parser.add_argument("--yes", action="store_true", help="Skip operator prompts")
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--collect-baseline",
        action="store_true",
        help="Collect an optional diagnostic baseline before the sample.",
    )
    baseline_group.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline collection (recommended for the improved absolute model).",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument("--skip-te-check", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--expected-food", choices=["blank", "banana", "meat"])
    parser.add_argument(
        "--expected-freshness",
        choices=["not_applicable", "fresh", "fermented", "spoiled"],
    )
    return parser


def resolve_paths(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    output = root / "model" / "classification_outputs"
    model_dir = root / "model" / "artifacts"
    return {
        "root": root,
        "config": root / "config" / "rpi5.toml",
        "display_init": root / "config" / "co5300_init.json",
        "dashboard": root / "tools" / "co5300_dashboard.py",
        "test_script": root / "model" / "test_model.py",
        "model": model_dir / "enose_multitask_model.joblib",
        "result": output / "test_results" / "session_prediction.json",
        "display_state": root / "runtime" / "display_state.json",
    }


def require_file(label: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def project_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source_candidates = (root / "src", root.parent / "src")
    source = next(
        (
            candidate
            for candidate in source_candidates
            if (candidate / "enose" / "__init__.py").is_file()
        ),
        source_candidates[0],
    )
    old = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source) if not old else str(source) + os.pathsep + old
    )
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def resolve_uart_device(requested: str) -> str:
    """Resolve the SVM41 serial port while keeping the normal command short."""
    if requested != "auto":
        path = Path(requested).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SVM41 UART device not found: {path}")
        return str(path)

    candidates: list[Path] = []
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ):
        candidates.extend(Path(path) for path in sorted(glob.glob(pattern)))

    unique: list[Path] = []
    resolved_paths: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in resolved_paths:
            unique.append(candidate)
            resolved_paths.add(resolved)

    if not unique:
        raise FileNotFoundError(
            "No SVM41 UART device found. Connect the Sensirion USB-UART "
            "adapter or run with --uart /dev/ttyUSB0."
        )

    for candidate in unique:
        if "sensirion" in str(candidate).lower():
            return str(candidate)
    return str(unique[0])


def stream(command: list[str], root: Path, environment: dict[str, str]) -> list[str]:
    print("\n$ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line.rstrip("\n"))
    code = process.wait()
    if code:
        raise RuntimeError(f"Command exited with status {code}: {' '.join(command)}")
    return lines


def acquisition_output_dir(config_path: Path, root: Path) -> Path:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    try:
        configured = Path(str(config["acquisition"]["output_dir"]))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing acquisition.output_dir in {config_path}") from exc
    return configured if configured.is_absolute() else (root / configured).resolve()


def find_acquired_csv(
    lines: list[str], output_dir: Path, before: set[Path], root: Path
) -> Path:
    pattern = re.compile(r"^writing\s+(.+\.csv)\s*$", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            candidate = Path(match.group(1))
            candidate = candidate if candidate.is_absolute() else root / candidate
            if candidate.is_file():
                return candidate.resolve()
    new_files = set(output_dir.glob("*.csv")) - before
    if not new_files:
        raise FileNotFoundError(f"Acquisition created no CSV in {output_dir}")
    return max(new_files, key=lambda path: path.stat().st_mtime).resolve()


def find_live_csv(
    lines: list[str], output_dir: Path, before: set[Path], root: Path
) -> Path | None:
    """Return the currently written acquisition CSV, if it is visible yet."""
    pattern = re.compile(r"^writing\s+(.+\.csv)\s*$", re.IGNORECASE)
    for line in reversed(lines.copy()):
        match = pattern.match(line.strip())
        if match:
            candidate = Path(match.group(1))
            candidate = candidate if candidate.is_absolute() else root / candidate
            if candidate.is_file():
                return candidate.resolve()
    new_files = set(output_dir.glob("*.csv")) - before
    if new_files:
        return max(new_files, key=lambda path: path.stat().st_mtime).resolve()
    return None


def acquisition_command(
    paths: dict[str, Path], frames: int, uart_device: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "enose",
        "acquire-tgs-svm41",
        "--config",
        str(paths["config"]),
        "--uart",
        uart_device,
        "--frames",
        str(frames),
    ]


def acquire(paths: dict[str, Path], frames: int, uart_device: str) -> Path:
    if frames < 80:
        raise ValueError("At least 80 frames are required for warm-up and one valid window")
    output_dir = acquisition_output_dir(paths["config"], paths["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("*.csv"))
    command = acquisition_command(paths, frames, uart_device)
    lines = stream(command, paths["root"], project_environment(paths["root"]))
    return find_acquired_csv(lines, output_dir, before, paths["root"])


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def status_display_state(
    food_type: str,
    freshness_level: str,
    system_status: str,
) -> dict[str, Any]:
    """Create a valid display state before the first prediction is available."""
    return {
        "food_type": food_type,
        "freshness_level": freshness_level,
        "confidence": 0.0,
        "temperature_c": None,
        "humidity_rh": None,
        "voc_raw": None,
        "nox_raw": None,
        "nh3_value": None,
        "h2s_value": None,
        "nh3_unit": "mV",
        "h2s_unit": "mV",
        "system_status": system_status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def prompt_yes_no(question: str, *, default: bool = True) -> bool:
    """Ask a yes/no question until the operator enters a valid response."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def run_model(
    paths: dict[str, Path],
    sample_csv: Path,
    baseline_csv: Path | None,
    args: argparse.Namespace,
    *,
    latest_window: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(paths["test_script"]),
        "--model",
        str(paths["model"]),
        "--input-csv",
        str(sample_csv),
        "--output-json",
        str(paths["result"]),
        "--confidence-threshold",
        str(args.confidence_threshold),
    ]
    if latest_window:
        # The improved model was trained on one median vector per recording.
        # For live data, aggregate all windows in the currently available
        # recording prefix instead of using only the newest window.
        command.append("--streaming-prefix")
    if baseline_csv is not None:
        command += ["--baseline-csv", str(baseline_csv)]
    if args.expected_food:
        command += ["--expected-food", args.expected_food]
    if args.expected_freshness:
        command += ["--expected-freshness", args.expected_freshness]
    completed = subprocess.run(
        command,
        cwd=paths["root"],
        env=project_environment(paths["root"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Live model inference failed with status {completed.returncode}:\n"
            f"{completed.stdout.strip()}"
        )
    with paths["result"].open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("test_model.py produced an invalid JSON result")
    return result


def numeric(row: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if number == number:
            return number
    return None


def sensor_summary(path: Path) -> dict[str, float | None]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[-SUMMARY_ROWS:]
    if not rows:
        raise ValueError(f"Sensor CSV has no rows: {path}")

    def mean(names: tuple[str, ...], multiplier: float = 1.0) -> float | None:
        values = [numeric(row, names) for row in rows]
        valid = [value * multiplier for value in values if value is not None]
        return sum(valid) / len(valid) if valid else None

    return {
        "temperature_c": mean(("svm41_temperature_c", "sht45_temperature_c", "bme690_temperature_c")),
        "humidity_rh": mean(("svm41_relative_humidity_pct", "sht45_relative_humidity_pct", "bme690_relative_humidity_pct")),
        "voc_raw": mean(("svm41_voc_index", "sgp41_voc_index", "sgp41_sraw_voc")),
        "nox_raw": mean(("svm41_nox_index", "sgp41_nox_index", "sgp41_sraw_nox")),
        "nh3_value": mean(("nh3_diff_voltage_v",), 1000.0),
        "h2s_value": mean(("h2s_diff_voltage_v",), 1000.0),
    }


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "ok"}


def valid_model_row_count(path: Path) -> int:
    """Count complete rows accepted by the training-time validity filters."""
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            count = 0
            for row in rows:
                if "ads7828_ok" in row and not truthy(row.get("ads7828_ok")):
                    continue
                if "svm41_ok" in row and not truthy(row.get("svm41_ok")):
                    continue
                count += 1
            return count
    except (OSError, csv.Error):
        # The writer may still be creating the header; retry on the next poll.
        return 0


def streaming_parameters(
    model_path: Path, requested_step: int | None
) -> tuple[int, int]:
    bundle = joblib.load(model_path)
    try:
        config = bundle["feature_config"]
        first_prediction = int(config["minimum_warmup"]) + int(
            config["minimum_window"]
        )
        model_step = int(config["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Model bundle has an invalid feature_config") from exc
    step = model_step if requested_step is None else requested_step
    if step < 1:
        raise ValueError("--prediction-step must be at least 1")
    return first_prediction, step


def make_display_state(
    result: dict[str, Any],
    summary: dict[str, Any],
    *,
    prediction_number: int = 1,
    acquisition_complete: bool = True,
) -> dict[str, Any]:
    evaluation = result.get("evaluation")
    if evaluation is not None and not evaluation["all_requested_labels_correct"]:
        status = "TEST FAIL"
    elif result.get("blank_like_warning", False):
        status = "BLANK-LIKE"
    elif result["low_confidence"]:
        status = "LOW CONF"
    elif evaluation is not None:
        status = "TEST PASS"
    else:
        status = "OK"
    return {
        "food_type": str(result["food_type"]).replace("_", " ").title(),
        "freshness_level": str(result["freshness_level"]).replace("_", " ").title(),
        "confidence": float(result["overall_confidence"]),
        **summary,
        "nh3_unit": "mV",
        "h2s_unit": "mV",
        "system_status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "food_probabilities": result["food_probabilities"],
        "freshness_probabilities": result["freshness_probabilities"],
        "inference_seconds": result["inference_seconds"],
        "evaluation": evaluation,
        "blank_reference_probability": result.get("blank_reference_probability"),
        "blank_like_warning": bool(result.get("blank_like_warning", False)),
        "prediction_number": prediction_number,
        "sample_frames_seen": result["frames"]["total_valid_frames"],
        "window_mode": result["frames"]["prediction_mode"],
        "acquisition_complete": acquisition_complete,
    }


def show_dashboard(
    paths: dict[str, Path], args: argparse.Namespace, *, hold: bool
) -> None:
    require_file("CO5300 dashboard", paths["dashboard"])
    require_file("CO5300 init file", paths["display_init"])
    command = [
        sys.executable,
        str(paths["dashboard"]),
        "--state-file", str(paths["display_state"]),
        "--init", str(paths["display_init"]),
        "--gpiochip", "auto",
        "--clk", "21",
        "--sio0", "20",
        "--sio1", "19",
        "--sio2", "16",
        "--sio3", "26",
        "--cs", "18",
        "--rst", "25",
        "--te", "24",
        "--half-period-us", "5",
        "--chunk-bytes", "1024",
        "--once",
    ]
    if hold:
        command.append("--hold")
    if args.skip_te_check:
        command.append("--skip-te-check")
    stream(command, paths["root"], project_environment(paths["root"]))


def dashboard_command(
    paths: dict[str, Path], args: argparse.Namespace
) -> list[str]:
    """Build a continuously refreshing dashboard command."""
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
        "--refresh-seconds",
        "0.5",
    ]
    if args.skip_te_check:
        command.append("--skip-te-check")
    return command


def start_dashboard(
    paths: dict[str, Path], args: argparse.Namespace
) -> subprocess.Popen[str]:
    """Start one dashboard process that watches display_state.json."""
    require_file("CO5300 dashboard", paths["dashboard"])
    require_file("CO5300 init file", paths["display_init"])
    command = dashboard_command(paths, args)
    print("\nStarting continuously refreshing CO5300 dashboard...", flush=True)
    return subprocess.Popen(
        command,
        cwd=paths["root"],
        env=project_environment(paths["root"]),
        text=True,
    )


def stop_dashboard(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def publish_live_prediction(
    paths: dict[str, Path],
    sample: Path,
    baseline: Path | None,
    args: argparse.Namespace,
    prediction_number: int,
    *,
    acquisition_complete: bool,
) -> dict[str, Any]:
    result = run_model(
        paths,
        sample,
        baseline,
        args,
        latest_window=True,
    )
    state = make_display_state(
        result,
        sensor_summary(sample),
        prediction_number=prediction_number,
        acquisition_complete=acquisition_complete,
    )
    atomic_json(paths["display_state"], state)
    frames_seen = result["frames"]["total_valid_frames"]
    print(
        f"\n[LIVE {prediction_number}] frames={frames_seen} | "
        f"food={result['food_type']} ({result['food_confidence']:.1%}) | "
        f"freshness={result['freshness_level']} "
        f"({result['freshness_confidence']:.1%}) | "
        f"blank_ref={result.get('blank_reference_probability', 0.0):.1%}",
        flush=True,
    )
    return result


def acquire_with_live_predictions(
    paths: dict[str, Path],
    frames: int,
    baseline: Path | None,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], int]:
    first_prediction, prediction_step = streaming_parameters(
        paths["model"], args.prediction_step
    )
    if frames < first_prediction:
        raise ValueError(
            f"--frames={frames} is too short for live prediction; need at least "
            f"{first_prediction} valid frames"
        )

    output_dir = acquisition_output_dir(paths["config"], paths["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("*.csv"))
    command = acquisition_command(paths, frames, args.uart_device)
    print("\n$ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=paths["root"],
        env=project_environment(paths["root"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line.rstrip("\n"))

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    sample: Path | None = None
    last_successful_rows = -1
    last_attempted_rows = -1
    prediction_number = 0
    latest_result: dict[str, Any] | None = None

    print(
        f"Recording-prefix prediction starts at {first_prediction} valid frames and refreshes "
        f"every {prediction_step} new valid frame(s).",
        flush=True,
    )
    try:
        while process.poll() is None:
            if sample is None:
                sample = find_live_csv(
                    output_lines, output_dir, before, paths["root"]
                )
            if sample is not None:
                valid_rows = valid_model_row_count(sample)
                ready = valid_rows >= first_prediction
                moved = (
                    last_successful_rows < 0
                    or valid_rows - last_successful_rows >= prediction_step
                )
                not_retried_same_file = valid_rows != last_attempted_rows
                if ready and moved and not_retried_same_file:
                    last_attempted_rows = valid_rows
                    try:
                        prediction_number += 1
                        latest_result = publish_live_prediction(
                            paths,
                            sample,
                            baseline,
                            args,
                            prediction_number,
                            acquisition_complete=False,
                        )
                        # Acquisition continues while inference runs, so the
                        # model may have consumed more rows than were present
                        # at the trigger check.
                        last_successful_rows = int(
                            latest_result["frames"]["total_valid_frames"]
                        )
                    except RuntimeError as error:
                        prediction_number -= 1
                        print(f"\nLive prediction postponed: {error}", file=sys.stderr)
            time.sleep(0.5)

        reader.join(timeout=5.0)
        if process.returncode:
            raise RuntimeError(
                f"Sensor acquisition failed with status {process.returncode}"
            )
        if sample is None:
            sample = find_acquired_csv(
                output_lines, output_dir, before, paths["root"]
            )

        final_valid_rows = valid_model_row_count(sample)
        if final_valid_rows < first_prediction:
            raise ValueError(
                f"Only {final_valid_rows} valid frames were recorded; need "
                f"{first_prediction} for the first prediction"
            )
        if latest_result is None or final_valid_rows != last_successful_rows:
            prediction_number += 1
            latest_result = publish_live_prediction(
                paths,
                sample,
                baseline,
                args,
                prediction_number,
                acquisition_complete=True,
            )
        else:
            # Mark the last already-displayed prediction as complete without
            # re-running the same window.
            state = make_display_state(
                latest_result,
                sensor_summary(sample),
                prediction_number=prediction_number,
                acquisition_complete=True,
            )
            atomic_json(paths["display_state"], state)
        return sample, latest_result, prediction_number
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def print_result(result: dict[str, Any], sample: Path, baseline: Path | None) -> None:
    print("\n" + "=" * 72)
    print("E-NOSE CLASSIFICATION RESULT")
    print("=" * 72)
    print(f"Sample CSV       : {sample}")
    print(f"Baseline CSV     : {baseline or 'not used'}")
    print(f"Food type        : {result['food_type']} ({result['food_confidence']:.1%})")
    print(f"Freshness        : {result['freshness_level']} ({result['freshness_confidence']:.1%})")
    print(f"Overall confidence: {result['overall_confidence']:.1%}")
    print(f"Blank reference   : {result.get('blank_reference_probability', 0.0):.1%}")
    print(f"Blank-like warning: {result.get('blank_like_warning', False)}")
    print(f"Inference time   : {result['inference_seconds']:.3f} s")
    if result.get("evaluation"):
        outcome = "PASS" if result["evaluation"]["all_requested_labels_correct"] else "FAIL"
        print(f"Known-label test : {outcome}")


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_paths(args.project_root)
    require_file("test script", paths["test_script"])
    require_file("trained model", paths["model"])
    dashboard_process: subprocess.Popen[str] | None = None

    try:
        if args.input_csv:
            sample = args.input_csv.expanduser().resolve()
            require_file("input CSV", sample)
            if args.baseline_csv:
                baseline = args.baseline_csv.expanduser().resolve()
                require_file("baseline CSV", baseline)
            else:
                # The improved deployed model uses absolute recording-level
                # features, so an existing CSV can be tested without a baseline.
                baseline = None
            result = run_model(
                paths,
                sample,
                baseline,
                args,
                latest_window=True,
            )
            prediction_count = 1
            state = make_display_state(
                result,
                sensor_summary(sample),
                prediction_number=prediction_count,
                acquisition_complete=True,
            )
            atomic_json(paths["display_state"], state)
            if (
                not args.no_display
                and (
                    dashboard_process is None
                    or dashboard_process.poll() is not None
                )
            ):
                show_dashboard(paths, args, hold=not args.no_hold)
        else:
            if args.baseline_csv:
                raise ValueError("--baseline-csv is only used with --input-csv")
            require_file("acquisition config", paths["config"])
            args.uart_device = resolve_uart_device(args.uart)
            print(f"SVM41 UART      : {args.uart_device}")

            if args.collect_baseline:
                use_baseline = True
            elif args.no_baseline or args.yes:
                # The improved model selected absolute recording-level features.
                # In unattended mode, do not spend an extra acquisition on a
                # baseline unless --collect-baseline was explicitly requested.
                use_baseline = False
            else:
                use_baseline = prompt_yes_no(
                    "Collect an optional diagnostic baseline / 是否采集参考 baseline?",
                    default=False,
                )

            if not args.no_display:
                initial_state = (
                    status_display_state("Clean Air", "Baseline", "BASELINE")
                    if use_baseline
                    else status_display_state("Insert Food", "Ready", "READY")
                )
                atomic_json(paths["display_state"], initial_state)
                dashboard_process = start_dashboard(paths, args)

            if not use_baseline:
                baseline = None
                print("Baseline skipped; the improved absolute recording-level model will be used.")
            else:
                if not args.yes:
                    input("\nPrepare clean air and press Enter to collect the baseline...")
                baseline = acquire(
                    paths,
                    args.baseline_frames,
                    args.uart_device,
                )

            if not args.no_display:
                atomic_json(
                    paths["display_state"],
                    status_display_state("Insert Food", "Ready", "READY"),
                )
            if not args.yes:
                input("\nInsert the food sample and press Enter to start live prediction...")

            if not args.no_display:
                atomic_json(
                    paths["display_state"],
                    status_display_state("Collecting", "Waiting", "SAMPLING"),
                )
            sample, result, prediction_count = acquire_with_live_predictions(
                paths,
                args.frames,
                baseline,
                args,
            )

        print_result(result, sample, baseline)
        print(f"Live predictions  : {prediction_count}")
        print(f"Result JSON      : {paths['result']}")
        print(f"Display state    : {paths['display_state']}")

        if dashboard_process is not None and not args.no_hold:
            print("Final result is on the display. Press Ctrl+C to exit.")
            dashboard_process.wait()
            dashboard_process = None
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        try:
            atomic_json(
                paths["display_state"],
                {
                    "food_type": "Unknown",
                    "freshness_level": "Unknown",
                    "confidence": 0.0,
                    "system_status": "ERROR",
                    "error_message": str(error),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            if (
                not args.no_display
                and (
                    dashboard_process is None
                    or dashboard_process.poll() is not None
                )
            ):
                show_dashboard(paths, args, hold=not args.no_hold)
        except Exception as display_error:
            print(f"Could not update display: {display_error}", file=sys.stderr)
        return 1
    finally:
        stop_dashboard(dashboard_process)


if __name__ == "__main__":
    raise SystemExit(main())

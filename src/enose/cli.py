from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

from .acquisition import Acquisition, Sensors
from .ads7828 import ADS7828
from .bme690 import BME690
from .config import AppConfig, DeviceConfig, load_config
from .csv_logger import (
    CSV_COLUMNS,
    NO_SGP41_BME690_SHT45_CSV_COLUMNS,
    TGS_SHT45_CSV_COLUMNS,
    CSVLogger,
    frame_to_row,
)
from .i2c_bus import DriverError, I2CBus
from .mcp3421 import MCP3421
from .records import Frame
from .sgp41 import SGP41
from .sht45 import SHT45
from .svm41_acquisition import run_svm41_acquisition

LOGGER = logging.getLogger(__name__)
REDUCED_ACQUISITION_COMMANDS = {
    "acquire-no-sgp41-bme690-sht45",
    "acquire-classify",
}
REDUCED_SVM41_COMMAND = "acquire-no-sgp41-bme690-sht45-with-svm41"
TGS_SVM41_COMMAND = "acquire-tgs-svm41"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m enose")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "probe",
        "diagnose",
        "acquire",
        "acquire-no-sgp41-bme690-sht45",
        "acquire-classify",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--verbose", action="store_true")
        if name == "acquire" or name in REDUCED_ACQUISITION_COMMANDS:
            command.add_argument("--frames", type=int)
        if name == "acquire-classify":
            command.add_argument("--uart", default="/dev/ttyUSB0", help="SVM41 USB-UART device.")
            command.add_argument(
                "--classification-window-rows",
                type=int,
                default=10,
                help="Most recent TGS rows to average (default: 10).",
            )
            command.add_argument(
                "--classification-update-rows",
                type=int,
                default=1,
                help="New rows between rule evaluations (default: 1).",
            )
            command.add_argument(
                "--classification-min-elapsed-s",
                type=float,
                default=60.0,
                help="Wait this long before classification (default: 60 s).",
            )
            command.add_argument(
                "--classification-confirmations",
                type=int,
                default=5,
                help="Matching rule outputs needed to confirm a final label (default: 5).",
            )
            command.add_argument(
                "--display-state",
                type=Path,
                help="atomically publish live classification JSON here",
            )
    svm41 = subparsers.add_parser("acquire-svm41")
    svm41.add_argument("--config", required=True, type=Path)
    svm41.add_argument("--uart", default="/dev/ttyUSB0")
    svm41.add_argument("--verbose", action="store_true")
    reduced_svm41 = subparsers.add_parser(REDUCED_SVM41_COMMAND)
    reduced_svm41.add_argument("--config", required=True, type=Path)
    reduced_svm41.add_argument("--uart", default="/dev/ttyUSB0")
    reduced_svm41.add_argument("--frames", type=int)
    reduced_svm41.add_argument("--verbose", action="store_true")
    tgs_svm41 = subparsers.add_parser(TGS_SVM41_COMMAND)
    tgs_svm41.add_argument("--config", required=True, type=Path)
    tgs_svm41.add_argument("--uart", default="/dev/ttyUSB0")
    tgs_svm41.add_argument("--frames", type=int)
    tgs_svm41.add_argument("--verbose", action="store_true")
    return parser


def build_sensors(config: AppConfig, bus: I2CBus) -> Sensors:
    return Sensors(
        sht45=SHT45(bus, config.sht45.address) if config.sht45.enabled else None,
        ads7828=(
            ADS7828(
                bus,
                config.ads7828.address,
                config.ads7828.saturation_low,
                config.ads7828.saturation_high,
            )
            if config.ads7828.enabled
            else None
        ),
        sgp41=SGP41(bus, config.sgp41.address) if config.sgp41.enabled else None,
        nh3=(
            MCP3421(
                bus,
                config.nh3.address,
                config.nh3.resolution_bits,
                config.nh3.gain,
                config.nh3.continuous,
            )
            if config.nh3.enabled
            else None
        ),
        h2s=(
            MCP3421(
                bus,
                config.h2s.address,
                config.h2s.resolution_bits,
                config.h2s.gain,
                config.h2s.continuous,
            )
            if config.h2s.enabled
            else None
        ),
        bme690=(
            BME690(
                bus,
                config.bme690.address,
                config.bme690.heater_temperature_c,
                config.bme690.heater_duration_ms,
                config.bme690.profile,
            )
            if config.bme690.enabled
            else None
        ),
    )


def _identity(name: str, sensors: Sensors) -> str:
    sensor = getattr(sensors, name)
    if name == "sht45":
        return f"serial=0x{sensor.serial_number():08X}"
    if name == "sgp41":
        sensor.self_test()
        return f"serial=0x{sensor.serial_number():012X}"
    if name == "bme690":
        chip, variant = sensor.identity()
        return f"chip=0x{chip:02X} variant=0x{variant:02X}"
    return "ACK"


def _probe(config: AppConfig, bus: I2CBus, sensors: Sensors) -> int:
    print("DEVICE    ADDRESS  REQUIRED  STATUS  IDENTITY")
    required_failed = False
    for name, device in config.devices.items():
        if not device.enabled:
            print(f"{name:<9} 0x{device.address:02X}     {str(device.required):<8} SKIP    disabled")
            continue
        if not bus.probe(device.address):
            print(f"{name:<9} 0x{device.address:02X}     {str(device.required):<8} FAIL    NACK")
            required_failed |= device.required
            continue
        try:
            identity = _identity(name, sensors)
        except Exception as exc:
            print(
                f"{name:<9} 0x{device.address:02X}     "
                f"{str(device.required):<8} FAIL    {exc}"
            )
            required_failed |= device.required
        else:
            print(
                f"{name:<9} 0x{device.address:02X}     "
                f"{str(device.required):<8} OK      {identity}"
            )
    return 1 if required_failed else 0


def _required_sample_failed(config: AppConfig, row: dict[str, object]) -> bool:
    status_fields: dict[str, str] = {
        "sht45": "sht45_ok",
        "ads7828": "ads7828_ok",
        "sgp41": "sgp41_ok",
        "nh3": "nh3_ok",
        "h2s": "h2s_ok",
        "bme690": "bme690_ok",
    }
    return any(
        device.enabled and device.required and not row[status_fields[name]]
        for name, device in config.devices.items()
    )


def _without_sgp41_bme690_sht45(config: AppConfig) -> AppConfig:
    return replace(
        config,
        sgp41=replace(config.sgp41, enabled=False, required=False),
        bme690=replace(config.bme690, enabled=False, required=False),
        sht45=replace(config.sht45, enabled=False, required=False),
    )


def _tgs_only(config: AppConfig) -> AppConfig:
    """Keep TGS plus SHT45 temperature/humidity; disable other devices."""
    return replace(
        config,
        nh3=replace(config.nh3, enabled=False, required=False),
        h2s=replace(config.h2s, enabled=False, required=False),
        sgp41=replace(config.sgp41, enabled=False, required=False),
        bme690=replace(config.bme690, enabled=False, required=False),
    )


def _format_frame(
    frame: Frame,
    columns: tuple[str, ...] = CSV_COLUMNS,
) -> str:
    row = frame_to_row(frame)
    return " ".join(
        f"{name}={'-' if row[name] == '' else row[name]}" for name in columns
    )


def _print_frame(frame: Frame, columns: tuple[str, ...] = CSV_COLUMNS) -> None:
    print(_format_frame(frame, columns), flush=True)


def _print_classification(frame: Frame, result: dict[str, Any]) -> None:
    predictions = result["predictions"]
    if "rule_status" in result:
        averages = result.get("sensor_averages", {})
        print(
            " ".join(
                (
                    "CLASSIFICATION",
                    "mode=hard_code",
                    f"sequence={frame.sequence}",
                    f"rule_status={result['rule_status']}",
                    f"confirmed={result.get('confirmed', False)}",
                    f"food_type={predictions['food_type']['overall_prediction']}",
                    f"freshness={predictions['freshness']['overall_prediction']}",
                    f"combined_class={predictions['combined_class']['overall_prediction']}",
                    f"rule_confidence={result.get('rule_confidence', 0.0):.4f}",
                    f"tgs2603_avg={averages.get('tgs2603_raw', float('nan')):.1f}",
                    f"tgs2620_avg={averages.get('tgs2620_raw', float('nan')):.1f}",
                    f"tgs2602_avg={averages.get('tgs2602_raw', float('nan')):.1f}",
                )
            ),
            flush=True,
        )
        return
    fields = [
        "CLASSIFICATION",
        f"sequence={frame.sequence}",
        f"input_rows={result['raw_rows']}",
        f"valid_rows={result['valid_rows']}",
        f"model_windows={result['window_count']}",
    ]
    for task_name in ("food_type", "freshness", "combined_class"):
        prediction = predictions[task_name]
        fields.append(f"{task_name}={prediction['overall_prediction']}")
        fields.append(
            f"{task_name}_confidence={prediction['confidence']:.4f}"
        )
    print(" ".join(fields), flush=True)


def _build_display_state(
    frame: Frame,
    result: dict[str, Any],
) -> dict[str, Any]:
    predictions = result["predictions"]
    food = predictions["food_type"]
    freshness = predictions["freshness"]
    combined = predictions["combined_class"]
    food_label = str(food["overall_prediction"])
    freshness_label = str(freshness["overall_prediction"])
    combined_label = str(combined["overall_prediction"])
    consistent = combined_label == f"{freshness_label}_{food_label}"
    complete = result["valid_rows"] == result["raw_rows"]
    return {
        "food_type": food_label.replace("_", " ").title(),
        "freshness_level": freshness_label.replace("_", " ").title(),
        "combined_class": combined_label.replace("_", " ").title(),
        "food_confidence": float(food["confidence"]),
        "freshness_confidence": float(freshness["confidence"]),
        "combined_confidence": float(combined["confidence"]),
        "confidence": float(result.get("rule_confidence", combined["confidence"])),
        "input_rows": int(result["raw_rows"]),
        "valid_rows": int(result["valid_rows"]),
        "model_windows": int(result["window_count"]),
        "temperature_c": None if frame.sht45 is None else frame.sht45.temperature_c,
        "humidity_rh": None if frame.sht45 is None else frame.sht45.relative_humidity_pct,
        "system_status": str(result.get("rule_status", "OK" if consistent and complete else "WARNING")),
        "rule_reason": str(result.get("rule_reason", "")),
        "rule_confirmed": bool(result.get("confirmed", False)),
        **result.get("sensor_averages", {}),
        "updated_at": frame.timestamp_utc,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _starting_display_state(window_rows: int) -> dict[str, Any]:
    return {
        "food_type": "Collecting",
        "freshness_level": "Waiting",
        "combined_class": f"Need {window_rows} Samples",
        "food_confidence": 0.0,
        "freshness_confidence": 0.0,
        "combined_confidence": 0.0,
        "input_rows": 0,
        "valid_rows": 0,
        "model_windows": 0,
        "system_status": "STARTING",
        "updated_at": "",
    }


def _update_live_classification(
    classifier: Any,
    frame: Frame,
    display_state_path: Path | None = None,
) -> None:
    try:
        result = classifier.add_frame(frame)
    except Exception:
        LOGGER.exception("classification failed; acquisition will continue")
        return
    if result is not None:
        _print_classification(frame, result)
        if display_state_path is not None:
            try:
                _write_json_atomic(
                    display_state_path,
                    _build_display_state(frame, result),
                )
            except Exception:
                LOGGER.exception(
                    "display state update failed; acquisition will continue"
                )


def _diagnose(config: AppConfig, sensors: Sensors) -> int:
    acquisition = Acquisition(config, sensors)
    try:
        acquisition.initialize()
        start = acquisition._monotonic()
        frame = acquisition.read_frame(0, start, start)
    finally:
        acquisition.shutdown()
    row = frame_to_row(frame)
    for key, value in row.items():
        print(f"{key}={value}")
    return 1 if _required_sample_failed(config, row) else 0


def _acquire(
    config: AppConfig,
    sensors: Sensors,
    max_frames: int | None,
    columns: tuple[str, ...] = CSV_COLUMNS,
    on_frame: Callable[[Frame], None] | None = None,
) -> int:
    if max_frames is not None and max_frames < 1:
        raise ValueError("--frames must be at least 1")
    enabled = [name for name, device in config.devices.items() if device.enabled]
    with CSVLogger(
        Path(config.acquisition.output_dir),
        effective_config=config.as_dict(),
        enabled_devices=enabled,
        flush_rows=config.acquisition.flush_rows,
        columns=columns,
    ) as csv_logger:
        print(f"writing {csv_logger.path}")
        acquisition = Acquisition(config, sensors)

        def handle_frame(frame: Frame) -> None:
            _print_frame(frame, columns)
            if on_frame is not None:
                on_frame(frame)

        try:
            count = acquisition.run(
                csv_logger,
                max_frames=max_frames,
                on_frame=handle_frame,
            )
        except KeyboardInterrupt:
            print("acquisition stopped", file=sys.stderr)
            return 130
        print(f"wrote {count} frames")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.command == "acquire-classify":
            from .classification import FoodFreshnessClassifier, SlidingWindowClassifier

            classifier = SlidingWindowClassifier(
                FoodFreshnessClassifier(),
                window_rows=args.classification_window_rows,
                update_rows=args.classification_update_rows,
                min_elapsed_s=args.classification_min_elapsed_s,
                confirmations=args.classification_confirmations,
            )
            if args.display_state is not None:
                _write_json_atomic(args.display_state, _starting_display_state(args.classification_window_rows))

            def classify_svm41_row(row: dict[str, object]) -> None:
                elapsed_s = float(row["elapsed_s"])
                sequence = int(row["sequence"])
                tgs2603 = float(row["tgs2603_raw"])
                tgs2620 = float(row["tgs2620_raw"])
                tgs2602 = float(row["tgs2602_raw"])
                try:
                    result = classifier.add_values(
                        elapsed_s, sequence, tgs2603, tgs2620, tgs2602,
                    )
                except (KeyError, TypeError, ValueError):
                    return
                frame = Frame(str(row["timestamp_utc"]), elapsed_s, sequence, 0.0, 0.0,
                              None, None, None, None, None, None)
                if args.display_state is not None:
                    if result is None:
                        progress = min(100, int(elapsed_s / args.classification_min_elapsed_s * 100))
                        if sequence == 0 or (sequence + 1) % 10 == 0:
                            print(
                                f"COLLECTING elapsed_s={elapsed_s:.0f} "
                                f"progress={progress}% "
                                f"tgs2603_raw={tgs2603:.0f}",
                                flush=True,
                            )
                        state = {
                            "food_type": "Collecting",
                            "freshness_level": f"{progress}%",
                            "combined_class": "Waiting for stable sample",
                            "confidence": 0.0,
                            "temperature_c": row.get("svm41_temperature_c") or None,
                            "humidity_rh": row.get("svm41_relative_humidity_pct") or None,
                            "tgs2603_raw": tgs2603,
                            "tgs2620_raw": tgs2620,
                            "tgs2602_raw": tgs2602,
                            "system_status": "COLLECTING",
                            "updated_at": frame.timestamp_utc,
                        }
                    else:
                        state = _build_display_state(frame, result)
                        _print_classification(frame, result)
                    state["temperature_c"] = row.get("svm41_temperature_c") or None
                    state["humidity_rh"] = row.get("svm41_relative_humidity_pct") or None
                    _write_json_atomic(args.display_state, state)
                elif result is not None:
                    _print_classification(frame, result)

            try:
                return run_svm41_acquisition(
                    config, args.uart, max_frames=args.frames, include_ads7828=True,
                    include_mcp3421=False, on_row=classify_svm41_row,
                )
            except KeyboardInterrupt:
                print("acquisition stopped", file=sys.stderr)
                return 130
        if args.command in {
            "acquire-svm41",
            REDUCED_SVM41_COMMAND,
            TGS_SVM41_COMMAND,
        }:
            try:
                run_svm41_acquisition(
                    config,
                    args.uart,
                    max_frames=getattr(args, "frames", None),
                    include_ads7828=args.command != "acquire-svm41",
                    include_mcp3421=args.command != TGS_SVM41_COMMAND,
                )
            except KeyboardInterrupt:
                print("acquisition stopped", file=sys.stderr)
                return 130
            return 0
        if args.command == "acquire-classify":
            config = _tgs_only(config)
        elif args.command in REDUCED_ACQUISITION_COMMANDS:
            config = _without_sgp41_bme690_sht45(config)
        live_classifier = None
        display_state_path = None
        if args.command == "acquire-classify":
            try:
                from .classification import (
                    FoodFreshnessClassifier,
                    SlidingWindowClassifier,
                )

                live_classifier = SlidingWindowClassifier(
                    FoodFreshnessClassifier(),
                    window_rows=args.classification_window_rows,
                    update_rows=args.classification_update_rows,
                    min_elapsed_s=args.classification_min_elapsed_s,
                    confirmations=args.classification_confirmations,
                )
                display_state_path = args.display_state
                if display_state_path is not None:
                    _write_json_atomic(
                        display_state_path,
                        _starting_display_state(
                            args.classification_window_rows
                        ),
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"classification initialization failed: {exc}"
                ) from exc
        with I2CBus(config.acquisition.bus) as bus:
            sensors = build_sensors(config, bus)
            if args.command == "probe":
                return _probe(config, bus, sensors)
            if args.command == "diagnose":
                return _diagnose(config, sensors)
            if args.command == "acquire-classify":
                columns = TGS_SHT45_CSV_COLUMNS
            elif args.command in REDUCED_ACQUISITION_COMMANDS:
                columns = NO_SGP41_BME690_SHT45_CSV_COLUMNS
            else:
                columns = CSV_COLUMNS
            on_frame = (
                (
                    lambda frame: _update_live_classification(
                        live_classifier,
                        frame,
                        display_state_path,
                    )
                )
                if live_classifier is not None
                else None
            )
            return _acquire(
                config,
                sensors,
                args.frames,
                columns,
                on_frame,
            )
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    except (DriverError, OSError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

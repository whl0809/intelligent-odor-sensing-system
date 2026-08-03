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
from .config import AppConfig, load_config
from .csv_logger import (
    ACQUISITION_SENSOR_NAMES,
    CLASSIFICATION_CSV_COLUMNS,
    CSV_COLUMNS,
    CSVLogger,
    columns_for_sensors,
    frame_to_row,
)
from .i2c_bus import DriverError, I2CBus
from .mcp3421 import MCP3421
from .records import Frame
from .sgp41 import SGP41
from .sht45 import SHT45
from .svm41_uart import SVM41UART, UART_BAUDRATE

LOGGER = logging.getLogger(__name__)
I2C_SENSOR_NAMES = frozenset(ACQUISITION_SENSOR_NAMES) - {"svm41"}


def _parse_sensor_selection(value: str) -> tuple[str, ...]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if requested == {"all"}:
        requested = set(ACQUISITION_SENSOR_NAMES)
    unknown = requested.difference(ACQUISITION_SENSOR_NAMES)
    if unknown:
        choices = ",".join(ACQUISITION_SENSOR_NAMES)
        raise argparse.ArgumentTypeError(
            f"unknown sensor(s): {', '.join(sorted(unknown))}; choose from {choices}"
        )
    if not requested:
        raise argparse.ArgumentTypeError("--sensors must contain at least one sensor")
    return tuple(name for name in ACQUISITION_SENSOR_NAMES if name in requested)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m enose")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "diagnose", "acquire", "acquire-classify"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--verbose", action="store_true")
        if name in {"acquire", "acquire-classify"}:
            command.add_argument("--frames", type=int)
        if name == "acquire":
            command.add_argument(
                "--sensors",
                type=_parse_sensor_selection,
                help=(
                    "comma-separated sensors: tgs,nh3,h2s,bme690,sgp41,"
                    "sht45,svm41; omit to use enabled TOML I2C sensors"
                ),
            )
            command.add_argument(
                "--uart",
                default="/dev/ttyUSB0",
                help="SVM41 UART device (default: /dev/ttyUSB0)",
            )
        if name == "acquire-classify":
            command.add_argument(
                "--classification-window-rows",
                type=int,
                default=60,
                help="CSV rows per classification input (default: 60, minimum: 20)",
            )
            command.add_argument(
                "--classification-update-rows",
                type=int,
                default=10,
                help="new rows between classifications (default: 10)",
            )
            command.add_argument(
                "--display-state",
                type=Path,
                help="atomically publish live classification JSON here",
            )
    return parser


def build_sensors(
    config: AppConfig,
    bus: I2CBus | None,
    svm41_uart: str | None = None,
) -> Sensors:
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
        svm41=SVM41UART(svm41_uart) if svm41_uart is not None else None,
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


def _config_for_sensors(
    config: AppConfig,
    sensor_names: tuple[str, ...],
) -> AppConfig:
    selected = set(sensor_names)
    return replace(
        config,
        ads7828=replace(
            config.ads7828,
            enabled="tgs" in selected,
            required=config.ads7828.required if "tgs" in selected else False,
        ),
        nh3=replace(
            config.nh3,
            enabled="nh3" in selected,
            required=config.nh3.required if "nh3" in selected else False,
        ),
        h2s=replace(
            config.h2s,
            enabled="h2s" in selected,
            required=config.h2s.required if "h2s" in selected else False,
        ),
        bme690=replace(
            config.bme690,
            enabled="bme690" in selected,
            required=config.bme690.required if "bme690" in selected else False,
        ),
        sgp41=replace(
            config.sgp41,
            enabled="sgp41" in selected,
            required=config.sgp41.required if "sgp41" in selected else False,
        ),
        sht45=replace(
            config.sht45,
            enabled="sht45" in selected,
            required=config.sht45.required if "sht45" in selected else False,
        ),
    )


def _default_acquisition_sensors(config: AppConfig) -> tuple[str, ...]:
    enabled = {
        "tgs" if name == "ads7828" else name
        for name, device in config.devices.items()
        if device.enabled
    }
    return tuple(name for name in ACQUISITION_SENSOR_NAMES if name in enabled)


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
        "input_rows": int(result["raw_rows"]),
        "valid_rows": int(result["valid_rows"]),
        "model_windows": int(result["window_count"]),
        "nh3_value": (
            None
            if frame.nh3 is None
            else frame.nh3.differential_voltage_v * 1000.0
        ),
        "nh3_unit": "mV",
        "h2s_value": (
            None
            if frame.h2s is None
            else frame.h2s.differential_voltage_v * 1000.0
        ),
        "h2s_unit": "mV",
        "system_status": "OK" if consistent and complete else "WARNING",
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
        "nh3_value": None,
        "nh3_unit": "mV",
        "h2s_value": None,
        "h2s_unit": "mV",
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
    sensor_names: tuple[str, ...],
    columns: tuple[str, ...],
    svm41_uart: str | None = None,
    on_frame: Callable[[Frame], None] | None = None,
) -> int:
    if max_frames is not None and max_frames < 1:
        raise ValueError("--frames must be at least 1")
    effective_config = config.as_dict()
    if svm41_uart is not None:
        effective_config["svm41"] = {
            "uart_device": svm41_uart,
            "baudrate": UART_BAUDRATE,
        }
    with CSVLogger(
        Path(config.acquisition.output_dir),
        effective_config=effective_config,
        enabled_devices=list(sensor_names),
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
        if args.command in {"probe", "diagnose"}:
            with I2CBus(config.acquisition.bus) as bus:
                sensors = build_sensors(config, bus)
                if args.command == "probe":
                    return _probe(config, bus, sensors)
                return _diagnose(config, sensors)

        if args.command == "acquire":
            sensor_names = args.sensors or _default_acquisition_sensors(config)
            if not sensor_names:
                raise ValueError(
                    "no sensors selected and no I2C sensors are enabled in the config"
                )
            config = _config_for_sensors(config, sensor_names)
            columns = columns_for_sensors(sensor_names)
            svm41_uart = args.uart if "svm41" in sensor_names else None
            if set(sensor_names).intersection(I2C_SENSOR_NAMES):
                with I2CBus(config.acquisition.bus) as bus:
                    sensors = build_sensors(config, bus, svm41_uart)
                    return _acquire(
                        config,
                        sensors,
                        args.frames,
                        sensor_names,
                        columns,
                        svm41_uart,
                    )
            sensors = build_sensors(config, None, svm41_uart)
            return _acquire(
                config,
                sensors,
                args.frames,
                sensor_names,
                columns,
                svm41_uart,
            )

        classification_sensors = ("tgs", "nh3", "h2s")
        config = _config_for_sensors(config, classification_sensors)
        live_classifier = None
        display_state_path = None
        try:
            from .classification import (
                FoodFreshnessClassifier,
                SlidingWindowClassifier,
            )

            live_classifier = SlidingWindowClassifier(
                FoodFreshnessClassifier(),
                window_rows=args.classification_window_rows,
                update_rows=args.classification_update_rows,
            )
            display_state_path = args.display_state
            if display_state_path is not None:
                _write_json_atomic(
                    display_state_path,
                    _starting_display_state(args.classification_window_rows),
                )
        except Exception as exc:
            raise RuntimeError(
                f"classification initialization failed: {exc}"
            ) from exc
        with I2CBus(config.acquisition.bus) as bus:
            sensors = build_sensors(config, bus)
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
                classification_sensors,
                CLASSIFICATION_CSV_COLUMNS,
                on_frame=on_frame,
            )
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    except (DriverError, OSError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

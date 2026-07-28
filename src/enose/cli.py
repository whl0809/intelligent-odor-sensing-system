from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

from .acquisition import Acquisition, Sensors
from .ads7828 import ADS7828
from .bme690 import BME690
from .config import AppConfig, DeviceConfig, load_config
from .csv_logger import (
    CSV_COLUMNS,
    NO_SGP41_BME690_SHT45_CSV_COLUMNS,
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
    svm41 = subparsers.add_parser("acquire-svm41")
    svm41.add_argument("--config", required=True, type=Path)
    svm41.add_argument("--uart", default="/dev/ttyUSB0")
    svm41.add_argument("--verbose", action="store_true")
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


def _update_live_classification(classifier: Any, frame: Frame) -> None:
    try:
        result = classifier.add_frame(frame)
    except Exception:
        LOGGER.exception("classification failed; acquisition will continue")
        return
    if result is not None:
        _print_classification(frame, result)


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
        if args.command == "acquire-svm41":
            try:
                run_svm41_acquisition(config, args.uart)
            except KeyboardInterrupt:
                print("acquisition stopped", file=sys.stderr)
                return 130
            return 0
        if args.command in REDUCED_ACQUISITION_COMMANDS:
            config = _without_sgp41_bme690_sht45(config)
        live_classifier = None
        if args.command == "acquire-classify":
            try:
                from .classification import (
                    FoodFreshnessClassifier,
                    SlidingWindowClassifier,
                )

                live_classifier = SlidingWindowClassifier(
                    FoodFreshnessClassifier(),
                    window_rows=60,
                    update_rows=10,
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
            columns = (
                NO_SGP41_BME690_SHT45_CSV_COLUMNS
                if args.command in REDUCED_ACQUISITION_COMMANDS
                else CSV_COLUMNS
            )
            on_frame = (
                (
                    lambda frame: _update_live_classification(
                        live_classifier,
                        frame,
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

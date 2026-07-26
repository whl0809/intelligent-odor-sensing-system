from __future__ import annotations

import csv
import json
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import AppConfig
from .csv_logger import _git_commit
from .i2c_bus import I2CBus, NotReadyError
from .mcp3421 import MCP3421
from .records import MCP3421Sample
from .svm41_uart import SVM41Sample, SVM41UART, UART_BAUDRATE

LOGGER = logging.getLogger(__name__)

SVM41_CSV_COLUMNS = (
    "timestamp_utc",
    "elapsed_s",
    "nh3_raw",
    "nh3_diff_voltage_v",
    "h2s_raw",
    "h2s_diff_voltage_v",
    "svm41_temperature_c",
    "svm41_relative_humidity_pct",
    "svm41_voc_index",
    "svm41_nox_index",
    "nh3_error",
    "h2s_error",
    "svm41_error",
)


class _CSV:
    def __init__(self, config: AppConfig, uart_device: str) -> None:
        output_dir = Path(config.acquisition.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(UTC)
        stem = started.strftime("enose_nh3_h2s_svm41_%Y%m%dT%H%M%S_%fZ")
        self.path = output_dir / f"{stem}.csv"
        self.metadata_path = output_dir / f"{stem}.metadata.json"
        self._handle = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=SVM41_CSV_COLUMNS)
        self._writer.writeheader()
        self._flush_rows = config.acquisition.flush_rows
        self._rows_since_flush = 0
        metadata = {
            "csv_schema_version": 1,
            "mode": "acquire-svm41",
            "effective_configuration": {
                "acquisition": asdict(config.acquisition),
                "nh3": asdict(config.nh3),
                "h2s": asdict(config.h2s),
                "svm41": {
                    "uart_device": uart_device,
                    "baudrate": UART_BAUDRATE,
                },
            },
            "software_commit": _git_commit(),
            "hostname": socket.gethostname(),
            "start_time_utc": started.isoformat().replace("+00:00", "Z"),
            "enabled_devices": ["nh3", "h2s", "svm41"],
        }
        with self.metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        self._handle.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        if not self._handle.closed:
            self.flush()
            self._handle.close()


def _error_code(phase: str, exc: Exception) -> str:
    if isinstance(exc, NotReadyError):
        suffix = "not_ready"
    elif isinstance(exc, OSError):
        suffix = "io"
    else:
        suffix = type(exc).__name__.lower()
    return f"{phase}_{suffix}"


def _read(
    name: str,
    action: Callable[[], Any],
) -> tuple[Any | None, str]:
    try:
        return action(), ""
    except Exception as exc:
        LOGGER.error(
            "%s read failed",
            name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return None, _error_code("read", exc)


def _display(value: object, precision: int | None = None) -> str:
    if value == "":
        return "-"
    if precision is not None and isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def run_svm41_acquisition(
    config: AppConfig,
    uart_device: str,
    *,
    max_frames: int | None = None,
    bus_factory: Callable[[int], Any] = I2CBus,
    mcp_factory: Callable[..., Any] = MCP3421,
    svm41_factory: Callable[[str], Any] = SVM41UART,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    utcnow_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    print_fn: Callable[[str], None] = print,
) -> int:
    csv_logger = _CSV(config, uart_device)
    print_fn(f"writing {csv_logger.path}")
    bus: Any | None = None
    nh3: Any | None = None
    h2s: Any | None = None
    svm41: Any | None = None
    persistent_errors = {"nh3": "", "h2s": "", "svm41": ""}
    count = 0

    try:
        try:
            bus = bus_factory(config.acquisition.bus)
        except Exception as exc:
            LOGGER.exception("I2C bus initialization failed")
            code = _error_code("initialize", exc)
            persistent_errors["nh3"] = code
            persistent_errors["h2s"] = code
        else:
            for name, device_config in (("nh3", config.nh3), ("h2s", config.h2s)):
                try:
                    sensor = mcp_factory(
                        bus,
                        device_config.address,
                        device_config.resolution_bits,
                        device_config.gain,
                        device_config.continuous,
                    )
                    sensor.configure()
                except Exception as exc:
                    LOGGER.exception("%s initialization failed", name)
                    persistent_errors[name] = _error_code("initialize", exc)
                else:
                    if name == "nh3":
                        nh3 = sensor
                    else:
                        h2s = sensor

        try:
            svm41 = svm41_factory(uart_device)
            svm41.start()
        except Exception as exc:
            LOGGER.exception("SVM41 UART initialization failed")
            persistent_errors["svm41"] = _error_code("initialize", exc)

        start = monotonic_fn()
        while max_frames is None or count < max_frames:
            deadline = start + (count + 1) * config.acquisition.interval_s
            sleep_fn(max(0.0, deadline - monotonic_fn()))
            timestamp = utcnow_fn().isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )

            nh3_sample: MCP3421Sample | None = None
            nh3_error = persistent_errors["nh3"]
            if nh3 is not None:
                nh3_sample, nh3_error = _read("nh3", nh3.read)

            h2s_sample: MCP3421Sample | None = None
            h2s_error = persistent_errors["h2s"]
            if h2s is not None:
                h2s_sample, h2s_error = _read("h2s", h2s.read)

            svm41_sample: SVM41Sample | None = None
            svm41_error = persistent_errors["svm41"]
            if svm41 is not None and not svm41_error:
                svm41_sample, svm41_error = _read("svm41", svm41.read)

            row: dict[str, object] = {
                column: "" for column in SVM41_CSV_COLUMNS
            }
            row.update(
                {
                    "timestamp_utc": timestamp,
                    "elapsed_s": monotonic_fn() - start,
                    "nh3_error": nh3_error,
                    "h2s_error": h2s_error,
                    "svm41_error": svm41_error,
                }
            )
            if nh3_sample is not None:
                row["nh3_raw"] = nh3_sample.raw
                row["nh3_diff_voltage_v"] = nh3_sample.differential_voltage_v
            if h2s_sample is not None:
                row["h2s_raw"] = h2s_sample.raw
                row["h2s_diff_voltage_v"] = h2s_sample.differential_voltage_v
            if svm41_sample is not None:
                row["svm41_temperature_c"] = svm41_sample.temperature_c
                row["svm41_relative_humidity_pct"] = (
                    svm41_sample.relative_humidity_pct
                )
                row["svm41_voc_index"] = svm41_sample.voc_index
                row["svm41_nox_index"] = svm41_sample.nox_index

            csv_logger.write(row)
            errors = ",".join(
                f"{name}:{error}"
                for name, error in (
                    ("nh3", nh3_error),
                    ("h2s", h2s_error),
                    ("svm41", svm41_error),
                )
                if error
            )
            print_fn(
                f"{timestamp} "
                f"NH3 raw={_display(row['nh3_raw'])} "
                f"V={_display(row['nh3_diff_voltage_v'], 6)} | "
                f"H2S raw={_display(row['h2s_raw'])} "
                f"V={_display(row['h2s_diff_voltage_v'], 6)} | "
                f"SVM41 T={_display(row['svm41_temperature_c'], 2)} C "
                f"RH={_display(row['svm41_relative_humidity_pct'], 2)} % "
                f"VOC Index={_display(row['svm41_voc_index'], 1)} "
                f"NOx Index={_display(row['svm41_nox_index'], 1)} "
                f"errors={errors or '-'}"
            )
            count += 1
    finally:
        if svm41 is not None:
            try:
                svm41.stop()
            except Exception:
                LOGGER.exception("failed to stop SVM41 measurement")
            try:
                svm41.close()
            except Exception:
                LOGGER.exception("failed to close SVM41 UART")
        if bus is not None:
            try:
                bus.close()
            except Exception:
                LOGGER.exception("failed to close I2C bus")
        csv_logger.close()
    return count

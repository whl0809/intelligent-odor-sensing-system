from __future__ import annotations

import csv
import json
import socket
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .records import Frame

CSV_SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "timestamp_utc",
    "elapsed_s",
    "sequence",
    "frame_duration_ms",
    "deadline_miss_ms",
    "sht45_temperature_c",
    "sht45_relative_humidity_pct",
    "sht45_ok",
    "tgs2620_raw",
    "tgs2620_voltage_v",
    "tgs2610_raw",
    "tgs2610_voltage_v",
    "tgs2611_raw",
    "tgs2611_voltage_v",
    "tgs2600_raw",
    "tgs2600_voltage_v",
    "tgs2602_raw",
    "tgs2602_voltage_v",
    "tgs2603_raw",
    "tgs2603_voltage_v",
    "ads7828_ok",
    "nh3_raw",
    "nh3_diff_voltage_v",
    "nh3_ok",
    "h2s_raw",
    "h2s_diff_voltage_v",
    "h2s_ok",
    "sgp41_sraw_voc",
    "sgp41_sraw_nox",
    "sgp41_voc_index",
    "sgp41_nox_index",
    "sgp41_compensated",
    "sgp41_ok",
    "bme690_temperature_c",
    "bme690_relative_humidity_pct",
    "bme690_pressure_pa",
    "bme690_gas_resistance_ohm",
    "bme690_gas_valid",
    "bme690_heater_stable",
    "bme690_ok",
    "error_codes",
)
NO_SGP41_BME690_SHT45_CSV_COLUMNS = tuple(
    column
    for column in CSV_COLUMNS
    if not column.startswith(("sgp41_", "bme690_", "sht45_"))
)
TGS_ONLY_CSV_COLUMNS = tuple(
    column
    for column in CSV_COLUMNS
    if not column.startswith(("sht45_", "nh3_", "h2s_", "sgp41_", "bme690_"))
)


def frame_to_row(frame: Frame) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "timestamp_utc": frame.timestamp_utc,
            "elapsed_s": frame.elapsed_s,
            "sequence": frame.sequence,
            "frame_duration_ms": frame.frame_duration_ms,
            "deadline_miss_ms": frame.deadline_miss_ms,
            "sht45_ok": frame.sht45 is not None,
            "ads7828_ok": frame.ads7828 is not None,
            "nh3_ok": frame.nh3 is not None,
            "h2s_ok": frame.h2s is not None,
            "sgp41_ok": frame.sgp41 is not None,
            "bme690_ok": frame.bme690 is not None,
            "error_codes": ";".join(frame.error_codes),
        }
    )
    if frame.sht45 is not None:
        row["sht45_temperature_c"] = frame.sht45.temperature_c
        row["sht45_relative_humidity_pct"] = frame.sht45.relative_humidity_pct
    if frame.ads7828 is not None:
        for sensor, reading in frame.ads7828.by_sensor().items():
            row[f"{sensor}_raw"] = reading.raw
            row[f"{sensor}_voltage_v"] = reading.voltage_v
    if frame.nh3 is not None:
        row["nh3_raw"] = frame.nh3.raw
        row["nh3_diff_voltage_v"] = frame.nh3.differential_voltage_v
    if frame.h2s is not None:
        row["h2s_raw"] = frame.h2s.raw
        row["h2s_diff_voltage_v"] = frame.h2s.differential_voltage_v
    if frame.sgp41 is not None:
        row["sgp41_sraw_voc"] = frame.sgp41.sraw_voc
        row["sgp41_sraw_nox"] = frame.sgp41.sraw_nox
        row["sgp41_voc_index"] = (
            "" if frame.sgp41.voc_index is None else frame.sgp41.voc_index
        )
        row["sgp41_nox_index"] = (
            "" if frame.sgp41.nox_index is None else frame.sgp41.nox_index
        )
        row["sgp41_compensated"] = frame.sgp41.compensated
    if frame.bme690 is not None:
        row["bme690_temperature_c"] = frame.bme690.temperature_c
        row["bme690_relative_humidity_pct"] = frame.bme690.relative_humidity_pct
        row["bme690_pressure_pa"] = frame.bme690.pressure_pa
        row["bme690_gas_resistance_ohm"] = frame.bme690.gas_resistance_ohm
        row["bme690_gas_valid"] = frame.bme690.gas_valid
        row["bme690_heater_stable"] = frame.bme690.heater_stable
    return row


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass
class CSVLogger:
    output_dir: Path
    effective_config: dict[str, Any]
    enabled_devices: list[str]
    flush_rows: int = 10
    start_time: datetime | None = None
    columns: tuple[str, ...] = CSV_COLUMNS

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = self.start_time or datetime.now(UTC)
        self.start_time = started
        stem = started.strftime("enose_%Y%m%dT%H%M%S_%fZ")
        self.path = self.output_dir / f"{stem}.csv"
        self.metadata_path = self.output_dir / f"{stem}.metadata.json"
        self._handle = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.columns)
        self._writer.writeheader()
        self._rows_since_flush = 0
        metadata = {
            "csv_schema_version": CSV_SCHEMA_VERSION,
            "effective_configuration": self.effective_config,
            "software_commit": _git_commit(),
            "hostname": socket.gethostname(),
            "start_time_utc": started.isoformat().replace("+00:00", "Z"),
            "enabled_devices": self.enabled_devices,
        }
        with self.metadata_path.open("x", encoding="utf-8") as metadata_handle:
            json.dump(metadata, metadata_handle, indent=2, sort_keys=True)
            metadata_handle.write("\n")

    def write(self, frame: Frame) -> None:
        row = frame_to_row(frame)
        self._writer.writerow({column: row[column] for column in self.columns})
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        self._handle.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        if not self._handle.closed:
            self.flush()
            self._handle.close()

    def __enter__(self) -> CSVLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

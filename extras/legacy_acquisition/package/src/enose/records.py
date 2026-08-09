from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SHT45Sample:
    temperature_c: float
    relative_humidity_pct: float


@dataclass(frozen=True)
class ADS7828Reading:
    sensor: str
    channel: int
    raw: int
    voltage_v: float
    saturated: bool


@dataclass(frozen=True)
class ADS7828Sample:
    readings: tuple[ADS7828Reading, ...]

    def by_sensor(self) -> dict[str, ADS7828Reading]:
        return {reading.sensor: reading for reading in self.readings}


@dataclass(frozen=True)
class MCP3421Sample:
    raw: int
    differential_voltage_v: float
    resolution_bits: int
    gain: int


@dataclass(frozen=True)
class SGP41Sample:
    sraw_voc: int
    sraw_nox: int
    compensated: bool
    voc_index: float | None = None
    nox_index: float | None = None


@dataclass(frozen=True)
class BME690Sample:
    temperature_c: float
    relative_humidity_pct: float
    pressure_pa: float
    gas_resistance_ohm: float
    gas_valid: bool
    heater_stable: bool


@dataclass(frozen=True)
class Frame:
    timestamp_utc: str
    elapsed_s: float
    sequence: int
    frame_duration_ms: float
    deadline_miss_ms: float
    sht45: SHT45Sample | None
    ads7828: ADS7828Sample | None
    nh3: MCP3421Sample | None
    h2s: MCP3421Sample | None
    sgp41: SGP41Sample | None
    bme690: BME690Sample | None
    error_codes: tuple[str, ...] = ()


from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class AcquisitionConfig:
    bus: int = 1
    interval_s: float = 1.0
    output_dir: str = "data/raw"
    flush_rows: int = 10
    retries: int = 0


@dataclass(frozen=True)
class DeviceConfig:
    enabled: bool
    required: bool
    address: int


@dataclass(frozen=True)
class ADS7828Config(DeviceConfig):
    saturation_low: int = 1
    saturation_high: int = 4094


@dataclass(frozen=True)
class MCP3421Config(DeviceConfig):
    resolution_bits: int = 18
    gain: int = 1
    continuous: bool = True


@dataclass(frozen=True)
class SGP41Config(DeviceConfig):
    conditioning_s: float = 10.0


@dataclass(frozen=True)
class BME690Config(DeviceConfig):
    heater_temperature_c: int = 320
    heater_duration_ms: int = 150
    profile: str = "forced"


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    acquisition: AcquisitionConfig
    sht45: DeviceConfig
    ads7828: ADS7828Config
    sgp41: SGP41Config
    nh3: MCP3421Config
    h2s: MCP3421Config
    bme690: BME690Config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def devices(self) -> dict[str, DeviceConfig]:
        return {
            "sht45": self.sht45,
            "ads7828": self.ads7828,
            "sgp41": self.sgp41,
            "nh3": self.nh3,
            "h2s": self.h2s,
            "bme690": self.bme690,
        }


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing TOML section [{name}]")
    return value


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)

    config = AppConfig(
        schema_version=int(data.get("schema_version", 1)),
        acquisition=AcquisitionConfig(**_section(data, "acquisition")),
        sht45=DeviceConfig(**_section(data, "sht45")),
        ads7828=ADS7828Config(**_section(data, "ads7828")),
        sgp41=SGP41Config(**_section(data, "sgp41")),
        nh3=MCP3421Config(**_section(data, "nh3")),
        h2s=MCP3421Config(**_section(data, "h2s")),
        bme690=BME690Config(**_section(data, "bme690")),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.schema_version != 1:
        raise ValueError(f"unsupported configuration schema {config.schema_version}")
    if config.acquisition.interval_s <= 0:
        raise ValueError("acquisition interval_s must be positive")
    if config.acquisition.flush_rows < 1:
        raise ValueError("flush_rows must be at least 1")
    if config.acquisition.retries not in (0, 1):
        raise ValueError("retries must be 0 or 1")

    enabled_addresses: dict[int, str] = {}
    for name, device in config.devices.items():
        if not 0x03 <= device.address <= 0x77:
            raise ValueError(f"{name} address is not a valid 7-bit I2C address")
        if device.required and not device.enabled:
            raise ValueError(f"{name} cannot be required while disabled")
        if device.enabled:
            previous = enabled_addresses.get(device.address)
            if previous is not None:
                raise ValueError(f"{name} and {previous} share address 0x{device.address:02X}")
            enabled_addresses[device.address] = name

    for name, device in (("nh3", config.nh3), ("h2s", config.h2s)):
        if device.resolution_bits not in (12, 14, 16, 18):
            raise ValueError(f"{name} resolution_bits must be 12, 14, 16, or 18")
        if device.gain not in (1, 2, 4, 8):
            raise ValueError(f"{name} gain must be 1, 2, 4, or 8")
    if not 0 <= config.sgp41.conditioning_s <= 10:
        raise ValueError("sgp41 conditioning_s must be between 0 and 10")
    if config.sgp41.enabled and config.sgp41.conditioning_s != 10.0:
        raise ValueError("enabled SGP41 must use exactly 10 seconds of conditioning")
    if not 0 <= config.ads7828.saturation_low < config.ads7828.saturation_high <= 4095:
        raise ValueError("invalid ADS7828 saturation thresholds")

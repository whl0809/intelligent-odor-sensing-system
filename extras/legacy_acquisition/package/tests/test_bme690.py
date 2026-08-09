from __future__ import annotations

from dataclasses import replace

import pytest

from enose.acquisition import Acquisition, Sensors
from enose.bme690 import BME690
from enose.config import load_config
from enose.i2c_bus import IdentityError
from enose.records import BME690Sample, SHT45Sample
from conftest import FakeBus


SAMPLE = BME690Sample(
    temperature_c=24.5,
    relative_humidity_pct=41.25,
    pressure_pa=100_850.0,
    gas_resistance_ohm=123_456.0,
    gas_valid=True,
    heater_stable=False,
)


class StubBackend:
    def __init__(
        self,
        identity: tuple[int, int] = (0x61, 0x02),
        sample: BME690Sample = SAMPLE,
    ) -> None:
        self.identity = identity
        self.sample = sample
        self.initializations: list[tuple[int, int]] = []

    def initialize(
        self,
        heater_temperature_c: int,
        heater_duration_ms: int,
    ) -> tuple[int, int]:
        self.initializations.append((heater_temperature_c, heater_duration_ms))
        return self.identity

    def read(self) -> BME690Sample:
        return self.sample


class FailingBME690:
    def initialize(self) -> tuple[int, int]:
        raise OSError("BME690 NACK")


class FailingReadBackend(StubBackend):
    def read(self) -> BME690Sample:
        raise OSError("BME690 NACK")


class WorkingSHT45:
    def read(self) -> SHT45Sample:
        return SHT45Sample(22.0, 40.0)


def test_bosch_backend_initializes_once_and_returns_all_fields() -> None:
    backend = StubBackend()
    sensor = BME690(
        FakeBus(),
        heater_temperature_c=320,
        heater_duration_ms=150,
        backend=backend,
    )

    assert sensor.identity() == (0x61, 0x02)
    assert sensor.read() == SAMPLE
    assert backend.initializations == [(320, 150)]


def test_native_extension_runs_bosch_identity_path() -> None:
    _bme69x = pytest.importorskip("enose._bme69x")
    bus = FakeBus()
    bus.registers[(0x76, 0xD0)] = b"\x00"
    sleeps: list[float] = []
    driver = _bme69x.Driver(bus, sleeps.append, 0x76)

    with pytest.raises(_bme69x.SensorAPIError) as error:
        driver.initialize(320, 150)

    assert _bme69x.SENSORAPI_VERSION == "1.1.0"
    assert error.value.args == ("bme69x_init", -3)
    assert bus.writes == [(0x76, b"\xE0\xB6")]
    assert sleeps == [0.01]


def test_bme690_requires_bme690_variant() -> None:
    sensor = BME690(FakeBus(), backend=StubBackend(identity=(0x61, 0x01)))

    with pytest.raises(IdentityError):
        sensor.initialize()


def test_optional_bme690_initialization_failure_preserves_other_sensors() -> None:
    config = load_config("config/rpi5.toml")
    config = replace(
        config,
        sht45=replace(config.sht45, required=False),
        bme690=replace(config.bme690, required=False),
    )
    sensors = Sensors(sht45=WorkingSHT45(), bme690=FailingBME690())  # type: ignore[arg-type]
    acquisition = Acquisition(config, sensors)

    acquisition.initialize()
    frame = acquisition.read_frame(0, 0.0, 0.0)

    assert frame.sht45 == SHT45Sample(22.0, 40.0)
    assert frame.bme690 is None
    assert "bme690_nack" in frame.error_codes
    assert sensors.bme690 is None


def test_optional_bme690_read_failure_preserves_other_sensors() -> None:
    config = load_config("config/rpi5.toml")
    sensor = BME690(FakeBus(), backend=FailingReadBackend())
    acquisition = Acquisition(
        config,
        Sensors(sht45=WorkingSHT45(), bme690=sensor),
    )

    acquisition.initialize()
    frame = acquisition.read_frame(0, 0.0, 0.0)

    assert frame.sht45 == SHT45Sample(22.0, 40.0)
    assert frame.bme690 is None
    assert "bme690_nack" in frame.error_codes

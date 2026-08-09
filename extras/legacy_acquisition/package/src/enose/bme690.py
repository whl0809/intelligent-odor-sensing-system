from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from .i2c_bus import BusIO, DriverError, IdentityError, NotReadyError
from .records import BME690Sample

EXPECTED_CHIP_ID = 0x61
EXPECTED_VARIANT_ID = 0x02


class BME690Backend(Protocol):
    def initialize(
        self,
        heater_temperature_c: int,
        heater_duration_ms: int,
    ) -> tuple[int, int]: ...

    def read(self) -> BME690Sample: ...


class _BoschSensorAPIBackend:
    def __init__(
        self,
        bus: BusIO,
        address: int,
        sleep_fn: Callable[[float], None],
    ) -> None:
        try:
            from . import _bme69x
        except ImportError as exc:
            raise RuntimeError(
                "the Bosch BME690 native extension is not installed; "
                "reinstall the project so its C extension is built"
            ) from exc

        self._api = _bme69x
        self._driver = _bme69x.Driver(bus, sleep_fn, address)

    def _translate_api_error(self, exc: Exception) -> None:
        if len(exc.args) >= 2 and exc.args[1] == -3:
            raise IdentityError("Bosch SensorAPI did not find a BME69x device") from exc
        operation = exc.args[0] if exc.args else "Bosch SensorAPI"
        result = exc.args[1] if len(exc.args) >= 2 else "unknown"
        raise DriverError(f"{operation} failed with result {result}") from exc

    def initialize(
        self,
        heater_temperature_c: int,
        heater_duration_ms: int,
    ) -> tuple[int, int]:
        try:
            chip_id, variant_id = self._driver.initialize(
                heater_temperature_c,
                heater_duration_ms,
            )
        except self._api.SensorAPIError as exc:
            self._translate_api_error(exc)
        return int(chip_id), int(variant_id)

    def read(self) -> BME690Sample:
        try:
            (
                temperature_c,
                relative_humidity_pct,
                pressure_pa,
                gas_resistance_ohm,
                gas_valid,
                heater_stable,
            ) = self._driver.read()
        except self._api.NoDataError as exc:
            raise NotReadyError("BME690 returned no new sample") from exc
        except self._api.SensorAPIError as exc:
            self._translate_api_error(exc)
        return BME690Sample(
            temperature_c=float(temperature_c),
            relative_humidity_pct=float(relative_humidity_pct),
            pressure_pa=float(pressure_pa),
            gas_resistance_ohm=float(gas_resistance_ohm),
            gas_valid=bool(gas_valid),
            heater_stable=bool(heater_stable),
        )


class BME690:
    def __init__(
        self,
        bus: BusIO,
        address: int = 0x76,
        heater_temperature_c: int = 320,
        heater_duration_ms: int = 150,
        profile: str = "forced",
        backend: BME690Backend | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if profile != "forced":
            raise ValueError("BME690 profile must be 'forced'")
        self.bus = bus
        self.address = address
        self.heater_temperature_c = heater_temperature_c
        self.heater_duration_ms = heater_duration_ms
        self.profile = profile
        self.backend = backend
        self._sleep = sleep_fn
        self._initialized = False
        self._identity: tuple[int, int] | None = None

    def _get_backend(self) -> BME690Backend:
        if self.backend is None:
            self.backend = _BoschSensorAPIBackend(self.bus, self.address, self._sleep)
        return self.backend

    def initialize(self) -> tuple[int, int]:
        if self._initialized:
            assert self._identity is not None
            return self._identity

        chip_id, variant_id = self._get_backend().initialize(
            self.heater_temperature_c,
            self.heater_duration_ms,
        )
        if chip_id != EXPECTED_CHIP_ID or variant_id != EXPECTED_VARIANT_ID:
            raise IdentityError(
                "BME690 identity mismatch: "
                f"chip=0x{chip_id:02X}, variant=0x{variant_id:02X}"
            )
        self._identity = (chip_id, variant_id)
        self._initialized = True
        return self._identity

    def identity(self) -> tuple[int, int]:
        return self.initialize()

    def read(self) -> BME690Sample:
        self.initialize()
        return self._get_backend().read()

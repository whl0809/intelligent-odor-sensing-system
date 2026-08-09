from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

from .i2c_bus import BusIO, IdentityError, ParseError
from .records import SGP41Sample
from .sht45 import crc8, validate_word

LOGGER = logging.getLogger(__name__)

DEFAULT_RH_TICKS = 0x8000
DEFAULT_T_TICKS = 0x6666


def humidity_ticks(relative_humidity_pct: float) -> int:
    value = round(min(100.0, max(0.0, relative_humidity_pct)) * 65535.0 / 100.0)
    return min(65535, max(0, value))


def temperature_ticks(temperature_c: float) -> int:
    value = round((min(130.0, max(-45.0, temperature_c)) + 45.0) * 65535.0 / 175.0)
    return min(65535, max(0, value))


def encoded_word(value: int) -> bytes:
    word = value.to_bytes(2, "big")
    return word + bytes([crc8(word)])


def parse_raw_signals(data: bytes, compensated: bool) -> SGP41Sample:
    if len(data) != 6:
        raise ParseError(f"SGP41 expected 6 bytes, received {len(data)}")
    return SGP41Sample(
        sraw_voc=validate_word(data[:3]),
        sraw_nox=validate_word(data[3:]),
        compensated=compensated,
    )


class SGP41:
    def __init__(
        self,
        bus: BusIO,
        address: int = 0x59,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bus = bus
        self.address = address
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

    def _write_command(self, command: int, payload: bytes = b"") -> None:
        data = command.to_bytes(2, "big") + payload
        self.bus.write(self.address, data)
        LOGGER.debug("SGP41 write=%s", data.hex())

    def condition(self, duration_s: float = 10.0) -> None:
        if not 0 <= duration_s <= 10.0:
            raise ValueError("SGP41 conditioning duration must be between 0 and 10 seconds")
        iterations = math.ceil(duration_s)
        start = self._monotonic()
        payload = encoded_word(DEFAULT_RH_TICKS) + encoded_word(DEFAULT_T_TICKS)
        for index in range(iterations):
            self._write_command(0x2612, payload)
            self._sleep(0.05)
            response = self.bus.read(self.address, 3)
            LOGGER.debug("SGP41 conditioning response=%s", response.hex())
            validate_word(response)
            deadline = start + min(duration_s, index + 1.0)
            self._sleep(max(0.0, deadline - self._monotonic()))

    def prepare(self, duration_s: float = 10.0) -> None:
        self.condition(duration_s)
        # Sensirion specifies discarding the first raw result when no Gas Index
        # Algorithm is used. Prime measurement mode with compensation disabled.
        prime_start = self._monotonic()
        self.measure(None, None)
        self._sleep(max(0.0, prime_start + 1.0 - self._monotonic()))

    def measure(
        self,
        relative_humidity_pct: float | None,
        temperature_c: float | None,
    ) -> SGP41Sample:
        compensated = relative_humidity_pct is not None and temperature_c is not None
        if compensated:
            rh_ticks = humidity_ticks(relative_humidity_pct)
            t_ticks = temperature_ticks(temperature_c)
        else:
            rh_ticks = DEFAULT_RH_TICKS
            t_ticks = DEFAULT_T_TICKS
        payload = encoded_word(rh_ticks) + encoded_word(t_ticks)
        self._write_command(0x2619, payload)
        self._sleep(0.05)
        data = self.bus.read(self.address, 6)
        LOGGER.debug("SGP41 measure response=%s", data.hex())
        return parse_raw_signals(data, compensated)

    def heater_off(self) -> None:
        self._write_command(0x3615)

    def serial_number(self) -> int:
        self._write_command(0x3682)
        self._sleep(0.001)
        data = self.bus.read(self.address, 9)
        LOGGER.debug("SGP41 serial response=%s", data.hex())
        if len(data) != 9:
            raise ParseError(f"SGP41 serial number expected 9 bytes, received {len(data)}")
        return (
            (validate_word(data[:3]) << 32)
            | (validate_word(data[3:6]) << 16)
            | validate_word(data[6:])
        )

    def self_test(self) -> None:
        self._write_command(0x280E)
        self._sleep(0.32)
        data = self.bus.read(self.address, 3)
        LOGGER.debug("SGP41 self-test response=%s", data.hex())
        result = validate_word(data)
        if result & 0x03:
            raise IdentityError(f"SGP41 self-test failed with status 0x{result:04X}")

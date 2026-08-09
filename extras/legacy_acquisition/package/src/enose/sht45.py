from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .i2c_bus import BusIO, CRCError, ParseError
from .records import SHT45Sample

LOGGER = logging.getLogger(__name__)


def crc8(data: bytes) -> int:
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def validate_word(data: bytes) -> int:
    if len(data) != 3:
        raise ParseError(f"Sensirion word needs 3 bytes, received {len(data)}")
    if crc8(data[:2]) != data[2]:
        raise CRCError(
            f"Sensirion CRC mismatch: got 0x{data[2]:02X}, "
            f"expected 0x{crc8(data[:2]):02X}"
        )
    return int.from_bytes(data[:2], "big")


def parse_measurement(data: bytes) -> SHT45Sample:
    if len(data) != 6:
        raise ParseError(f"SHT45 expected 6 bytes, received {len(data)}")
    temperature_ticks = validate_word(data[:3])
    humidity_ticks = validate_word(data[3:])
    temperature_c = -45.0 + 175.0 * temperature_ticks / 65535.0
    humidity = -6.0 + 125.0 * humidity_ticks / 65535.0
    return SHT45Sample(temperature_c, min(100.0, max(0.0, humidity)))


class SHT45:
    def __init__(
        self,
        bus: BusIO,
        address: int = 0x44,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.bus = bus
        self.address = address
        self._sleep = sleep_fn

    def read(self) -> SHT45Sample:
        self.bus.write(self.address, [0xFD])
        self._sleep(0.01)
        data = self.bus.read(self.address, 6)
        LOGGER.debug("SHT45 command=FD response=%s", data.hex())
        return parse_measurement(data)

    def serial_number(self) -> int:
        self.bus.write(self.address, [0x89])
        self._sleep(0.001)
        data = self.bus.read(self.address, 6)
        LOGGER.debug("SHT45 command=89 response=%s", data.hex())
        if len(data) != 6:
            raise ParseError(f"SHT45 serial number expected 6 bytes, received {len(data)}")
        return (validate_word(data[:3]) << 16) | validate_word(data[3:])


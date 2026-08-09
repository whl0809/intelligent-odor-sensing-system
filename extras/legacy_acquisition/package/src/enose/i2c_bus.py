from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class DriverError(Exception):
    """Base class for sensor protocol failures."""


class CRCError(DriverError):
    pass


class NotReadyError(DriverError):
    pass


class IdentityError(DriverError):
    pass


class SaturationError(DriverError):
    pass


class ParseError(DriverError):
    pass


class BusIO(Protocol):
    def write(self, address: int, data: Sequence[int]) -> None: ...

    def read(self, address: int, length: int) -> bytes: ...

    def read_register(self, address: int, register: int, length: int = 1) -> bytes: ...

    def probe(self, address: int) -> bool: ...


class I2CBus:
    """Small smbus2 adapter that performs raw I2C transfers."""

    def __init__(self, bus_number: int = 1) -> None:
        try:
            from smbus2 import SMBus, i2c_msg
        except ImportError as exc:  # pragma: no cover - installation failure
            raise RuntimeError("smbus2 is required on the Raspberry Pi") from exc
        self._i2c_msg = i2c_msg
        self._bus = SMBus(bus_number)

    def write(self, address: int, data: Sequence[int]) -> None:
        message = self._i2c_msg.write(address, list(data))
        self._bus.i2c_rdwr(message)

    def read(self, address: int, length: int) -> bytes:
        message = self._i2c_msg.read(address, length)
        self._bus.i2c_rdwr(message)
        return bytes(message)

    def read_register(self, address: int, register: int, length: int = 1) -> bytes:
        write_message = self._i2c_msg.write(address, [register])
        read_message = self._i2c_msg.read(address, length)
        self._bus.i2c_rdwr(write_message, read_message)
        return bytes(read_message)

    def probe(self, address: int) -> bool:
        try:
            self._bus.write_quick(address)
        except OSError:
            return False
        return True

    def close(self) -> None:
        self._bus.close()

    def __enter__(self) -> I2CBus:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

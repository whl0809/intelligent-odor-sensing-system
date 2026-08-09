from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .i2c_bus import BusIO, NotReadyError, ParseError
from .records import MCP3421Sample

LOGGER = logging.getLogger(__name__)

RESOLUTION_BITS = {12: 0b00, 14: 0b01, 16: 0b10, 18: 0b11}
BITS_RESOLUTION = {value: key for key, value in RESOLUTION_BITS.items()}
GAIN_BITS = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}
BITS_GAIN = {value: key for key, value in GAIN_BITS.items()}
SAMPLES_PER_SECOND = {12: 240.0, 14: 60.0, 16: 15.0, 18: 3.75}


def configuration_byte(
    resolution_bits: int,
    gain: int,
    continuous: bool,
    start: bool = False,
) -> int:
    try:
        sample_bits = RESOLUTION_BITS[resolution_bits]
        gain_bits = GAIN_BITS[gain]
    except KeyError as exc:
        raise ValueError("invalid MCP3421 resolution or gain") from exc
    return (
        (0x80 if start else 0)
        | (0x10 if continuous else 0)
        | (sample_bits << 2)
        | gain_bits
    )


def response_length(resolution_bits: int) -> int:
    if resolution_bits not in RESOLUTION_BITS:
        raise ValueError("invalid MCP3421 resolution")
    return 4 if resolution_bits == 18 else 3


def decode_signed(data_bytes: bytes, resolution_bits: int) -> int:
    expected = 3 if resolution_bits == 18 else 2
    if len(data_bytes) != expected:
        raise ParseError(
            f"MCP3421 {resolution_bits}-bit data needs {expected} bytes, "
            f"received {len(data_bytes)}"
        )
    value = int.from_bytes(data_bytes, byteorder="big", signed=False)
    mask = (1 << resolution_bits) - 1
    value &= mask
    sign_bit = 1 << (resolution_bits - 1)
    return value - (1 << resolution_bits) if value & sign_bit else value


def raw_to_voltage(raw: int, resolution_bits: int, gain: int) -> float:
    return raw * 2.048 / ((2 ** (resolution_bits - 1)) * gain)


def parse_response(
    data: bytes,
    resolution_bits: int,
    gain: int,
) -> MCP3421Sample:
    expected = response_length(resolution_bits)
    if len(data) != expected:
        raise ParseError(f"MCP3421 expected {expected} bytes, received {len(data)}")
    config = data[-1]
    if config & 0x80:
        raise NotReadyError("MCP3421 conversion is not ready")
    returned_resolution = BITS_RESOLUTION[(config >> 2) & 0x03]
    returned_gain = BITS_GAIN[config & 0x03]
    if returned_resolution != resolution_bits or returned_gain != gain:
        raise ParseError(
            "MCP3421 returned configuration does not match requested "
            f"{resolution_bits}-bit gain {gain}"
        )
    raw = decode_signed(data[:-1], resolution_bits)
    return MCP3421Sample(
        raw=raw,
        differential_voltage_v=raw_to_voltage(raw, resolution_bits, gain),
        resolution_bits=resolution_bits,
        gain=gain,
    )


class MCP3421:
    def __init__(
        self,
        bus: BusIO,
        address: int,
        resolution_bits: int = 18,
        gain: int = 1,
        continuous: bool = True,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        configuration_byte(resolution_bits, gain, continuous)
        self.bus = bus
        self.address = address
        self.resolution_bits = resolution_bits
        self.gain = gain
        self.continuous = continuous
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

    @property
    def config_byte(self) -> int:
        return configuration_byte(self.resolution_bits, self.gain, self.continuous)

    def configure(self) -> None:
        value = self.config_byte
        self.bus.write(self.address, [value])
        LOGGER.debug("MCP3421 address=%02X config=%02X", self.address, value)

    def read(self) -> MCP3421Sample:
        if not self.continuous:
            trigger = configuration_byte(
                self.resolution_bits, self.gain, continuous=False, start=True
            )
            self.bus.write(self.address, [trigger])
            LOGGER.debug("MCP3421 address=%02X trigger=%02X", self.address, trigger)

        conversion_s = 1.0 / SAMPLES_PER_SECOND[self.resolution_bits]
        deadline = self._monotonic() + conversion_s * 1.25 + 0.01
        while True:
            data = self.bus.read(self.address, response_length(self.resolution_bits))
            LOGGER.debug("MCP3421 address=%02X response=%s", self.address, data.hex())
            try:
                return parse_response(data, self.resolution_bits, self.gain)
            except NotReadyError:
                if self._monotonic() >= deadline:
                    raise
                self._sleep(min(0.01, max(0.0, deadline - self._monotonic())))


from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .i2c_bus import BusIO, ParseError
from .records import ADS7828Reading, ADS7828Sample

LOGGER = logging.getLogger(__name__)

CHANNEL_COMMANDS = (0x8C, 0xCC, 0x9C, 0xDC, 0xAC, 0xEC, 0xBC, 0xFC)
SENSOR_CHANNELS = (
    ("tgs2620", 0),
    ("tgs2610", 1),
    ("tgs2611", 2),
    ("tgs2600", 3),
    ("tgs2602", 4),
    ("tgs2603", 5),
)
REFERENCE_V = 2.5


def command_for_channel(channel: int) -> int:
    try:
        return CHANNEL_COMMANDS[channel]
    except IndexError as exc:
        raise ValueError("ADS7828 channel must be between 0 and 7") from exc


def parse_conversion(data: bytes) -> int:
    if len(data) != 2:
        raise ParseError(f"ADS7828 expected 2 bytes, received {len(data)}")
    return ((data[0] & 0x0F) << 8) | data[1]


def raw_to_voltage(raw: int) -> float:
    if not 0 <= raw <= 4095:
        raise ValueError("ADS7828 raw code must be 12-bit")
    return raw * REFERENCE_V / 4096.0


class ADS7828:
    def __init__(
        self,
        bus: BusIO,
        address: int = 0x48,
        saturation_low: int = 1,
        saturation_high: int = 4094,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.bus = bus
        self.address = address
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self._sleep = sleep_fn
        self._initialized = False

    def initialize(self) -> None:
        command = command_for_channel(0)
        self.bus.write(self.address, [command])
        self._sleep(0.002)
        discarded = self.bus.read(self.address, 2)
        LOGGER.debug("ADS7828 startup command=%02X discarded=%s", command, discarded.hex())
        parse_conversion(discarded)
        self._initialized = True

    def read_channel(self, channel: int, sensor: str = "") -> ADS7828Reading:
        if not self._initialized:
            self.initialize()
        command = command_for_channel(channel)
        self.bus.write(self.address, [command])
        data = self.bus.read(self.address, 2)
        raw = parse_conversion(data)
        LOGGER.debug(
            "ADS7828 channel=%d command=%02X response=%s raw=%d",
            channel,
            command,
            data.hex(),
            raw,
        )
        return ADS7828Reading(
            sensor=sensor,
            channel=channel,
            raw=raw,
            voltage_v=raw_to_voltage(raw),
            saturated=raw <= self.saturation_low or raw >= self.saturation_high,
        )

    def read_all(self) -> ADS7828Sample:
        return ADS7828Sample(
            tuple(self.read_channel(channel, sensor) for sensor, channel in SENSOR_CHANNELS)
        )


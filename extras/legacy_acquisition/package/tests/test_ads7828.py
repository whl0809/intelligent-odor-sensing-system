from __future__ import annotations

import pytest

from enose.ads7828 import (
    CHANNEL_COMMANDS,
    SENSOR_CHANNELS,
    ADS7828,
    command_for_channel,
    parse_conversion,
    raw_to_voltage,
)
from conftest import FakeBus


def _encoded(raw: int) -> bytes:
    return bytes([(raw >> 8) & 0x0F, raw & 0xFF])


def test_channel_command_mapping() -> None:
    assert [command_for_channel(channel) for channel in range(8)] == list(
        CHANNEL_COMMANDS
    )
    assert CHANNEL_COMMANDS == (0x8C, 0xCC, 0x9C, 0xDC, 0xAC, 0xEC, 0xBC, 0xFC)
    with pytest.raises(ValueError):
        command_for_channel(8)


def test_parse_and_voltage() -> None:
    assert parse_conversion(bytes([0xFA, 0x55])) == 0xA55
    assert raw_to_voltage(4095) == pytest.approx(4095 * 2.5 / 4096)


def test_startup_sensor_order_and_saturation() -> None:
    values = [123, 0, 1000, 2000, 3000, 4095]
    bus = FakeBus([_encoded(456), *[_encoded(value) for value in values]])
    adc = ADS7828(bus, sleep_fn=lambda _: None)

    sample = adc.read_all()

    assert [reading.sensor for reading in sample.readings] == [
        pair[0] for pair in SENSOR_CHANNELS
    ]
    assert [reading.raw for reading in sample.readings] == values
    assert [reading.saturated for reading in sample.readings] == [
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert [data[0] for _, data in bus.writes] == [
        CHANNEL_COMMANDS[0],
        *CHANNEL_COMMANDS[:6],
    ]


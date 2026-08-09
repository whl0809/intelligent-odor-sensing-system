from __future__ import annotations

import pytest

from enose.i2c_bus import CRCError
from enose.sgp41 import (
    SGP41,
    encoded_word,
    humidity_ticks,
    parse_raw_signals,
    temperature_ticks,
)
from enose.sht45 import crc8, parse_measurement
from conftest import FakeBus, FakeClock


def _word(value: int) -> bytes:
    data = value.to_bytes(2, "big")
    return data + bytes([crc8(data)])


def test_sht45_crc_and_conversion() -> None:
    sample = parse_measurement(_word(0x6666) + _word(0x8000))
    assert sample.temperature_c == pytest.approx(25.0, abs=0.01)
    assert sample.relative_humidity_pct == pytest.approx(56.5, abs=0.01)
    broken = bytearray(_word(0x6666) + _word(0x8000))
    broken[2] ^= 1
    with pytest.raises(CRCError):
        parse_measurement(bytes(broken))


def test_sgp41_ticks_crc_and_raw_parse() -> None:
    assert humidity_ticks(0) == 0
    assert humidity_ticks(100) == 65535
    assert humidity_ticks(50) in (32767, 32768)
    assert temperature_ticks(-45) == 0
    assert temperature_ticks(130) == 65535
    assert temperature_ticks(25) == 0x6666
    assert encoded_word(0x8000) == bytes.fromhex("8000a2")
    sample = parse_raw_signals(_word(100) + _word(200), compensated=True)
    assert (sample.sraw_voc, sample.sraw_nox, sample.compensated) == (100, 200, True)
    broken = bytearray(_word(100) + _word(200))
    broken[-1] ^= 1
    with pytest.raises(CRCError):
        parse_raw_signals(bytes(broken), compensated=False)


def test_conditioning_is_one_hz_and_never_exceeds_ten_seconds() -> None:
    clock = FakeClock()
    response = _word(123)
    bus = FakeBus([response, response, response])
    sensor = SGP41(
        bus,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    sensor.condition(2.2)

    assert clock.value == pytest.approx(2.2)
    assert len(bus.writes) == 3
    assert all(data[:2] == bytes.fromhex("2612") for _, data in bus.writes)
    with pytest.raises(ValueError):
        sensor.condition(10.01)


def test_prepare_spaces_priming_measurement_at_one_hz() -> None:
    clock = FakeClock()
    conditioning = _word(123)
    raw = _word(100) + _word(200)
    bus = FakeBus([conditioning, raw])
    sensor = SGP41(
        bus,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    sensor.prepare(1.0)

    assert clock.value == pytest.approx(2.0)
    assert bus.writes[-1][1][:2] == bytes.fromhex("2619")

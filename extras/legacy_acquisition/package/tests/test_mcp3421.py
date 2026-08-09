from __future__ import annotations

import pytest

from enose.i2c_bus import NotReadyError, ParseError
from enose.mcp3421 import (
    configuration_byte,
    decode_signed,
    parse_response,
    raw_to_voltage,
    response_length,
)


def _container(raw: int, resolution: int) -> bytes:
    byte_count = 3 if resolution == 18 else 2
    if raw < 0:
        value = (1 << (byte_count * 8)) + raw
    else:
        value = raw
    return value.to_bytes(byte_count, "big")


@pytest.mark.parametrize("resolution", [12, 14, 16, 18])
@pytest.mark.parametrize("raw", [-1, 0, 1])
def test_signed_parsing_all_resolutions(resolution: int, raw: int) -> None:
    assert decode_signed(_container(raw, resolution), resolution) == raw


@pytest.mark.parametrize(
    ("resolution", "length"),
    [(12, 3), (14, 3), (16, 3), (18, 4)],
)
def test_response_byte_count(resolution: int, length: int) -> None:
    assert response_length(resolution) == length
    config = configuration_byte(resolution, 1, continuous=True)
    response = _container(42, resolution) + bytes([config])
    assert parse_response(response, resolution, 1).raw == 42
    with pytest.raises(ParseError):
        parse_response(response[:-1], resolution, 1)


def test_rdy_gain_and_voltage() -> None:
    config = configuration_byte(18, 8, continuous=True)
    sample = parse_response(_container(-131072, 18) + bytes([config]), 18, 8)
    assert sample.raw == -131072
    assert sample.differential_voltage_v == pytest.approx(-2.048 / 8)
    assert raw_to_voltage(1, 18, 1) == pytest.approx(2.048 / 131072)
    with pytest.raises(NotReadyError):
        parse_response(_container(0, 18) + bytes([config | 0x80]), 18, 8)


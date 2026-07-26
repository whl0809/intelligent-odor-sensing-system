from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

UART_BAUDRATE = 115200


@dataclass(frozen=True)
class SVM41Sample:
    temperature_c: float
    relative_humidity_pct: float
    voc_index: float
    nox_index: float


class SVM41UART:
    """Small lifecycle adapter for Sensirion's official SVM4x UART driver."""

    def __init__(
        self,
        device: str = "/dev/ttyUSB0",
        *,
        port_factory: Callable[..., Any] | None = None,
        channel_factory: Callable[[Any], Any] | None = None,
        sensor_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if port_factory is None or channel_factory is None or sensor_factory is None:
            try:
                from sensirion_driver_adapters.shdlc_adapter.shdlc_channel import (
                    ShdlcChannel,
                )
                from sensirion_shdlc_driver import ShdlcSerialPort
                from sensirion_uart_svm4x.device import Svm4xDevice
            except ImportError as exc:
                raise RuntimeError(
                    "sensirion-uart-svm4x is required for SVM41 UART acquisition"
                ) from exc
            port_factory = port_factory or ShdlcSerialPort
            channel_factory = channel_factory or ShdlcChannel
            sensor_factory = sensor_factory or Svm4xDevice

        self._port = port_factory(port=device, baudrate=UART_BAUDRATE)
        self._sensor = sensor_factory(channel_factory(self._port))
        self._started = False

    def start(self) -> None:
        self._sensor.start_measurement()
        self._started = True

    def read(self) -> SVM41Sample:
        if not self._started:
            raise RuntimeError("SVM41 measurement has not been started")
        humidity, temperature, voc_index, nox_index = (
            self._sensor.read_measured_values()
        )
        return SVM41Sample(
            temperature_c=float(temperature.value),
            relative_humidity_pct=float(humidity.value),
            voc_index=float(voc_index.value),
            nox_index=float(nox_index.value),
        )

    def stop(self) -> None:
        if self._started:
            try:
                self._sensor.stop_measurement()
            finally:
                self._started = False

    def close(self) -> None:
        self._port.close()

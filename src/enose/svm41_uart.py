from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .records import SVM41Sample

UART_BAUDRATE = 115200
SVM41_MEASUREMENT_INTERVAL_S = 1.0


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
        self._device = device
        self._port_factory = port_factory
        self._channel_factory = channel_factory
        self._sensor_factory = sensor_factory
        self._port: Any | None = None
        self._sensor: Any | None = None
        self._started = False

    def _open(self) -> None:
        if self._sensor is not None:
            return
        if (
            self._port_factory is None
            or self._channel_factory is None
            or self._sensor_factory is None
        ):
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
            self._port_factory = self._port_factory or ShdlcSerialPort
            self._channel_factory = self._channel_factory or ShdlcChannel
            self._sensor_factory = self._sensor_factory or Svm4xDevice

        self._port = self._port_factory(
            port=self._device,
            baudrate=UART_BAUDRATE,
        )
        self._sensor = self._sensor_factory(self._channel_factory(self._port))

    def start(self) -> None:
        self._open()
        assert self._sensor is not None
        self._sensor.start_measurement()
        self._started = True

    def read(self) -> SVM41Sample:
        if not self._started:
            raise RuntimeError("SVM41 measurement has not been started")
        assert self._sensor is not None
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
            assert self._sensor is not None
            try:
                self._sensor.stop_measurement()
            finally:
                self._started = False

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None
            self._sensor = None

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from conftest import FakeClock
from enose.acquisition import Acquisition, Sensors
from enose.config import load_config
from enose.records import ADS7828Reading, ADS7828Sample, Frame, SVM41Sample
from enose.svm41_uart import SVM41UART, UART_BAUDRATE


class Signal:
    def __init__(self, value: float) -> None:
        self.value = value


class FakePort:
    def __init__(self, port: str, baudrate: int) -> None:
        self.device = port
        self.baudrate = baudrate
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeOfficialDevice:
    def __init__(self, channel: object) -> None:
        self.channel = channel
        self.start_calls = 0
        self.stop_calls = 0

    def start_measurement(self) -> None:
        self.start_calls += 1

    def read_measured_values(self):
        return Signal(48.5), Signal(23.25), Signal(101.0), Signal(2.0)

    def stop_measurement(self) -> None:
        self.stop_calls += 1


def test_official_uart_adapter_lifecycle_and_value_order() -> None:
    ports: list[FakePort] = []
    devices: list[FakeOfficialDevice] = []

    def port_factory(**kwargs) -> FakePort:
        port = FakePort(**kwargs)
        ports.append(port)
        return port

    def device_factory(channel: object) -> FakeOfficialDevice:
        device = FakeOfficialDevice(channel)
        devices.append(device)
        return device

    driver = SVM41UART(
        "/dev/test-svm41",
        port_factory=port_factory,
        channel_factory=lambda port: port,
        sensor_factory=device_factory,
    )
    driver.start()
    sample = driver.read()
    driver.stop()
    driver.close()

    assert ports[0].device == "/dev/test-svm41"
    assert ports[0].baudrate == UART_BAUDRATE
    assert ports[0].closed
    assert devices[0].start_calls == 1
    assert devices[0].stop_calls == 1
    assert sample == SVM41Sample(23.25, 48.5, 101.0, 2.0)


class FakeADS7828:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def read_all(self) -> ADS7828Sample:
        names = (
            "tgs2620",
            "tgs2610",
            "tgs2611",
            "tgs2600",
            "tgs2602",
            "tgs2603",
        )
        return ADS7828Sample(
            tuple(
                ADS7828Reading(name, index, 100 + index, 0.1, False)
                for index, name in enumerate(names)
            )
        )


class FlakySVM41:
    def __init__(self) -> None:
        self.start_calls = 0
        self.read_calls = 0
        self.stop_calls = 0
        self.closed = False

    def start(self) -> None:
        self.start_calls += 1

    def read(self) -> SVM41Sample:
        self.read_calls += 1
        if self.read_calls == 2:
            raise OSError("SVM41 unavailable")
        return SVM41Sample(24.0, 50.0, 100.0, 1.0)

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.closed = True


class MemoryLogger:
    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.flushed = False

    def write(self, frame: Frame) -> None:
        self.frames.append(frame)

    def flush(self) -> None:
        self.flushed = True


def test_unified_loop_isolates_svm41_read_failure_and_closes_uart() -> None:
    config = load_config("config/rpi5.toml")
    config = replace(
        config,
        acquisition=replace(config.acquisition, interval_s=1.0),
    )
    clock = FakeClock()
    ads7828 = FakeADS7828()
    svm41 = FlakySVM41()
    logger = MemoryLogger()
    acquisition = Acquisition(
        config,
        Sensors(ads7828=ads7828, svm41=svm41),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        utcnow_fn=lambda: datetime(2026, 8, 2, tzinfo=UTC),
    )

    count = acquisition.run(logger, max_frames=2)

    assert count == 2
    assert ads7828.initialized
    assert logger.frames[0].ads7828 is not None
    assert logger.frames[0].svm41 == SVM41Sample(24.0, 50.0, 100.0, 1.0)
    assert logger.frames[1].ads7828 is not None
    assert logger.frames[1].svm41 is None
    assert "svm41_nack" in logger.frames[1].error_codes
    assert svm41.start_calls == 1
    assert svm41.stop_calls == 1
    assert svm41.closed
    assert logger.flushed


def test_unified_loop_continues_when_svm41_initialization_fails() -> None:
    class FailingStartSVM41(FlakySVM41):
        def start(self) -> None:
            self.start_calls += 1
            raise OSError("UART unavailable")

    config = load_config("config/rpi5.toml")
    clock = FakeClock()
    svm41 = FailingStartSVM41()
    logger = MemoryLogger()
    acquisition = Acquisition(
        config,
        Sensors(ads7828=FakeADS7828(), svm41=svm41),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        utcnow_fn=lambda: datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert acquisition.run(logger, max_frames=1) == 1
    assert logger.frames[0].ads7828 is not None
    assert logger.frames[0].svm41 is None
    assert "svm41_nack" in logger.frames[0].error_codes
    assert svm41.start_calls == 1
    assert svm41.stop_calls == 0
    assert svm41.closed

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime
import json

from conftest import FakeClock
from enose.config import load_config
from enose.records import ADS7828Reading, ADS7828Sample, MCP3421Sample
from enose.svm41_acquisition import (
    REDUCED_WITH_SVM41_CSV_COLUMNS,
    TGS_SVM41_CSV_COLUMNS,
    run_svm41_acquisition,
)
from enose.svm41_uart import SVM41Sample, SVM41UART, UART_BAUDRATE


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


class FakeBus:
    def __init__(self, bus_number: int) -> None:
        self.bus_number = bus_number
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeMCP3421:
    def __init__(self, address: int) -> None:
        self.address = address
        self.read_calls = 0
        self.configured = False

    def configure(self) -> None:
        self.configured = True

    def read(self) -> MCP3421Sample:
        self.read_calls += 1
        if self.address == 0x6A and self.read_calls == 2:
            raise OSError("H2S unavailable")
        return MCP3421Sample(
            raw=self.address,
            differential_voltage_v=self.address / 1_000_000,
            resolution_bits=18,
            gain=1,
        )


class FakeSVM41:
    def __init__(self, uart_device: str) -> None:
        self.uart_device = uart_device
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


class FakeADS7828:
    def __init__(self, _bus, address: int, *_args) -> None:
        self.address = address
        self.initialized = False
        self.read_calls = 0

    def initialize(self) -> None:
        self.initialized = True

    def read_all(self) -> ADS7828Sample:
        self.read_calls += 1
        if self.read_calls == 2:
            raise OSError("ADS7828 unavailable")
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
                ADS7828Reading(
                    sensor=name,
                    channel=index,
                    raw=100 + index,
                    voltage_v=(100 + index) * 2.5 / 4096.0,
                    saturated=False,
                )
                for index, name in enumerate(names)
            )
        )


class StableSVM41(FakeSVM41):
    def read(self) -> SVM41Sample:
        self.read_calls += 1
        return SVM41Sample(24.0, 50.0, 100.0, 1.0)


def test_isolated_loop_keeps_other_values_when_one_read_fails(tmp_path) -> None:
    config = load_config("config/rpi5.toml")
    config = replace(
        config,
        acquisition=replace(
            config.acquisition,
            output_dir=str(tmp_path),
            interval_s=1.0,
            flush_rows=1,
        ),
    )
    clock = FakeClock()
    buses: list[FakeBus] = []
    mcps: dict[int, FakeMCP3421] = {}
    svm41_devices: list[FakeSVM41] = []
    terminal: list[str] = []

    def bus_factory(bus_number: int) -> FakeBus:
        bus = FakeBus(bus_number)
        buses.append(bus)
        return bus

    def mcp_factory(_bus, address, *_args) -> FakeMCP3421:
        sensor = FakeMCP3421(address)
        mcps[address] = sensor
        return sensor

    def svm41_factory(uart_device: str) -> FakeSVM41:
        sensor = FakeSVM41(uart_device)
        svm41_devices.append(sensor)
        return sensor

    count = run_svm41_acquisition(
        config,
        "/dev/test-svm41",
        max_frames=2,
        bus_factory=bus_factory,
        mcp_factory=mcp_factory,
        svm41_factory=svm41_factory,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        utcnow_fn=lambda: datetime(2026, 7, 26, tzinfo=UTC),
        print_fn=terminal.append,
    )

    assert count == 2
    assert clock.value == 2.0
    assert buses[0].bus_number == 1
    assert buses[0].closed
    assert mcps[0x69].configured and mcps[0x6A].configured
    assert svm41_devices[0].start_calls == 1
    assert svm41_devices[0].stop_calls == 1
    assert svm41_devices[0].closed
    assert len(terminal) == 3

    csv_path = next(tmp_path.glob("*.csv"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["nh3_raw"] == str(0x69)
    assert rows[0]["h2s_raw"] == str(0x6A)
    assert rows[0]["svm41_temperature_c"] == "24.0"
    assert rows[1]["nh3_raw"] == str(0x69)
    assert rows[1]["h2s_raw"] == ""
    assert rows[1]["h2s_error"] == "read_io"
    assert rows[1]["svm41_temperature_c"] == ""
    assert rows[1]["svm41_error"] == "read_io"
    assert next(tmp_path.glob("*.metadata.json")).exists()


def test_reduced_svm41_mode_adds_tgs_and_isolates_ads_failure(
    tmp_path,
) -> None:
    config = load_config("config/rpi5.toml")
    config = replace(
        config,
        acquisition=replace(
            config.acquisition,
            output_dir=str(tmp_path),
            interval_s=1.0,
            flush_rows=1,
        ),
    )
    clock = FakeClock()
    ads_devices: list[FakeADS7828] = []

    def ads_factory(*args) -> FakeADS7828:
        sensor = FakeADS7828(*args)
        ads_devices.append(sensor)
        return sensor

    count = run_svm41_acquisition(
        config,
        "/dev/test-svm41",
        max_frames=2,
        bus_factory=FakeBus,
        mcp_factory=lambda _bus, address, *_args: FakeMCP3421(address),
        svm41_factory=StableSVM41,
        ads7828_factory=ads_factory,
        include_ads7828=True,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        utcnow_fn=lambda: datetime(2026, 7, 30, tzinfo=UTC),
        print_fn=lambda _line: None,
    )

    assert count == 2
    assert ads_devices[0].initialized
    csv_path = next(tmp_path.glob("*.csv"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == REDUCED_WITH_SVM41_CSV_COLUMNS
    assert not any(
        column.startswith(("sgp41_", "bme690_", "sht45_"))
        for column in REDUCED_WITH_SVM41_CSV_COLUMNS
    )
    assert rows[0]["tgs2620_raw"] == "100"
    assert rows[0]["ads7828_ok"] == "True"
    assert rows[0]["svm41_temperature_c"] == "24.0"
    assert rows[1]["tgs2620_raw"] == ""
    assert rows[1]["ads7828_ok"] == "False"
    assert "ads7828_read_io" in rows[1]["error_codes"]
    assert rows[1]["nh3_raw"] == str(0x69)
    assert rows[1]["svm41_temperature_c"] == "24.0"

    metadata_path = next(tmp_path.glob("*.metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == (
        "acquire-no-sgp41-bme690-sht45-with-svm41"
    )
    assert metadata["enabled_devices"] == [
        "ads7828",
        "nh3",
        "h2s",
        "svm41",
    ]


def test_tgs_svm41_mode_never_initializes_or_logs_mcp3421(
    tmp_path,
) -> None:
    config = load_config("config/rpi5.toml")
    config = replace(
        config,
        acquisition=replace(
            config.acquisition,
            output_dir=str(tmp_path),
            interval_s=1.0,
            flush_rows=1,
        ),
    )
    clock = FakeClock()
    terminal: list[str] = []

    def forbidden_mcp_factory(*_args):
        raise AssertionError("MCP3421 must not be initialized")

    count = run_svm41_acquisition(
        config,
        "/dev/test-svm41",
        max_frames=1,
        bus_factory=FakeBus,
        mcp_factory=forbidden_mcp_factory,
        svm41_factory=StableSVM41,
        ads7828_factory=FakeADS7828,
        include_ads7828=True,
        include_mcp3421=False,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        utcnow_fn=lambda: datetime(2026, 8, 1, tzinfo=UTC),
        print_fn=terminal.append,
    )

    assert count == 1
    csv_path = next(tmp_path.glob("*.csv"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == TGS_SVM41_CSV_COLUMNS
    assert not any(
        column.startswith(("nh3_", "h2s_"))
        for column in TGS_SVM41_CSV_COLUMNS
    )
    assert rows[0]["tgs2620_raw"] == "100"
    assert rows[0]["svm41_temperature_c"] == "24.0"
    assert "nh3" not in terminal[1].lower()
    assert "h2s" not in terminal[1].lower()

    metadata_path = next(tmp_path.glob("*.metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == "acquire-tgs-svm41"
    assert metadata["enabled_devices"] == ["ads7828", "svm41"]
    assert "nh3" not in metadata["effective_configuration"]
    assert "h2s" not in metadata["effective_configuration"]


def test_ctrl_c_stops_svm41_and_closes_outputs(tmp_path) -> None:
    config = load_config("config/rpi5.toml")
    config = replace(
        config,
        acquisition=replace(config.acquisition, output_dir=str(tmp_path)),
    )
    bus = FakeBus(1)
    svm41 = FakeSVM41("/dev/test-svm41")

    def interrupt_sleep(_duration: float) -> None:
        raise KeyboardInterrupt

    try:
        run_svm41_acquisition(
            config,
            "/dev/test-svm41",
            bus_factory=lambda _number: bus,
            mcp_factory=lambda _bus, address, *_args: FakeMCP3421(address),
            svm41_factory=lambda _device: svm41,
            sleep_fn=interrupt_sleep,
            print_fn=lambda _line: None,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert svm41.start_calls == 1
    assert svm41.stop_calls == 1
    assert svm41.closed
    assert bus.closed
    csv_path = next(tmp_path.glob("*.csv"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []

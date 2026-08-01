from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime
import json

import pytest

import enose.acquisition as acquisition_module
from enose.acquisition import Acquisition, Sensors
from enose.cli import (
    _format_frame,
    _parser,
    _update_live_classification,
    _without_sgp41_bme690_sht45,
)
from enose.config import load_config
from enose.csv_logger import (
    CSV_COLUMNS,
    NO_SGP41_BME690_SHT45_CSV_COLUMNS,
    CSVLogger,
    frame_to_row,
)
from enose.records import (
    BME690Sample,
    Frame,
    MCP3421Sample,
    SGP41Sample,
    SHT45Sample,
)
from conftest import FakeClock


class FlakySHT45:
    def __init__(self) -> None:
        self.calls = 0

    def read(self) -> SHT45Sample:
        self.calls += 1
        if self.calls == 1:
            return SHT45Sample(22.0, 40.0)
        raise OSError("NACK")


class StubSGP41:
    def __init__(self) -> None:
        self.compensation: list[tuple[float | None, float | None]] = []
        self.heater_was_turned_off = False

    def prepare(self, duration: float) -> None:
        pass

    def measure(
        self, humidity: float | None, temperature: float | None
    ) -> SGP41Sample:
        self.compensation.append((humidity, temperature))
        return SGP41Sample(100, 200, humidity is not None and temperature is not None)

    def heater_off(self) -> None:
        self.heater_was_turned_off = True


class MemoryLogger:
    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.flushed = False

    def write(self, frame: Frame) -> None:
        self.frames.append(frame)

    def flush(self) -> None:
        self.flushed = True


class FailingPrepareSGP41(StubSGP41):
    def prepare(self, duration: float) -> None:
        raise OSError("conditioning failed")


class FailingReadSGP41(StubSGP41):
    def measure(
        self, humidity: float | None, temperature: float | None
    ) -> SGP41Sample:
        raise OSError("measurement failed")


def _test_config():
    config = load_config("config/rpi5.toml")
    acquisition = replace(config.acquisition, retries=0)
    return replace(config, acquisition=acquisition)


def test_failed_values_are_missing_not_stale() -> None:
    config = _test_config()
    sht45 = FlakySHT45()
    sgp41 = StubSGP41()
    acquisition = Acquisition(config, Sensors(sht45=sht45, sgp41=sgp41))

    first = acquisition.read_frame(0, 0.0, 0.0)
    second = acquisition.read_frame(1, 0.0, 0.0)

    assert first.sht45 == SHT45Sample(22.0, 40.0)
    assert second.sht45 is None
    assert frame_to_row(second)["sht45_temperature_c"] == ""
    assert sgp41.compensation == [(40.0, 22.0), (None, None)]
    assert first.sgp41 is not None and first.sgp41.compensated
    assert second.sgp41 is not None and not second.sgp41.compensated
    assert "sht45_nack" in second.error_codes


def test_gas_index_algorithms_are_persistent_and_receive_matching_raw_signals(
    monkeypatch,
) -> None:
    events: list[str] = []

    class PreparedSGP41(StubSGP41):
        def prepare(self, duration: float) -> None:
            events.append("conditioned")

    class FakeVocAlgorithm:
        instances = 0
        inputs: list[int] = []

        def __init__(self) -> None:
            FakeVocAlgorithm.instances += 1
            events.append("voc_created")

        def process(self, raw: int) -> int:
            FakeVocAlgorithm.inputs.append(raw)
            return 101

    class FakeNoxAlgorithm:
        instances = 0
        inputs: list[int] = []

        def __init__(self) -> None:
            FakeNoxAlgorithm.instances += 1
            events.append("nox_created")

        def process(self, raw: int) -> int:
            FakeNoxAlgorithm.inputs.append(raw)
            return 2

    monkeypatch.setattr(acquisition_module, "VocAlgorithm", FakeVocAlgorithm)
    monkeypatch.setattr(acquisition_module, "NoxAlgorithm", FakeNoxAlgorithm)
    acquisition = Acquisition(_test_config(), Sensors(sgp41=PreparedSGP41()))

    acquisition.initialize()
    first = acquisition.read_frame(0, 0.0, 0.0)
    second = acquisition.read_frame(1, 0.0, 1.0)

    assert FakeVocAlgorithm.instances == 1
    assert FakeNoxAlgorithm.instances == 1
    assert events == ["conditioned", "voc_created", "nox_created"]
    assert FakeVocAlgorithm.inputs == [100, 100]
    assert FakeNoxAlgorithm.inputs == [200, 200]
    assert first.sgp41 is not None
    assert (first.sgp41.voc_index, first.sgp41.nox_index) == (101.0, 2.0)
    assert second.sgp41 is not None
    row = frame_to_row(first)
    assert row["sgp41_sraw_voc"] == 100
    assert row["sgp41_sraw_nox"] == 200
    assert row["sgp41_voc_index"] == 101.0
    assert row["sgp41_nox_index"] == 2.0


def test_missing_gas_index_package_keeps_raw_signals_and_empty_indices(
    monkeypatch,
) -> None:
    monkeypatch.setattr(acquisition_module, "VocAlgorithm", None)
    monkeypatch.setattr(acquisition_module, "NoxAlgorithm", None)
    acquisition = Acquisition(_test_config(), Sensors(sgp41=StubSGP41()))

    acquisition.initialize()
    frame = acquisition.read_frame(0, 0.0, 0.0)

    assert frame.sgp41 == SGP41Sample(100, 200, False)
    row = frame_to_row(frame)
    assert row["sgp41_sraw_voc"] == 100
    assert row["sgp41_sraw_nox"] == 200
    assert row["sgp41_voc_index"] == ""
    assert row["sgp41_nox_index"] == ""
    assert "sgp41_index_unavailable" in frame.error_codes


def test_failed_sgp41_read_does_not_stop_other_sensors(monkeypatch) -> None:
    monkeypatch.setattr(acquisition_module, "VocAlgorithm", None)
    monkeypatch.setattr(acquisition_module, "NoxAlgorithm", None)
    acquisition = Acquisition(
        _test_config(),
        Sensors(sht45=FlakySHT45(), sgp41=FailingReadSGP41()),
    )

    acquisition.initialize()
    frame = acquisition.read_frame(0, 0.0, 0.0)

    assert frame.sht45 == SHT45Sample(22.0, 40.0)
    assert frame.sgp41 is None
    assert "sgp41_nack" in frame.error_codes
    row = frame_to_row(frame)
    assert row["sht45_ok"] is True
    assert row["sgp41_voc_index"] == ""
    assert row["sgp41_nox_index"] == ""


def test_absolute_deadline_scheduler_and_shutdown() -> None:
    config = _test_config()
    clock = FakeClock()
    sgp41 = StubSGP41()
    logger = MemoryLogger()
    terminal_frames: list[Frame] = []
    acquisition = Acquisition(
        config,
        Sensors(sgp41=sgp41),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        utcnow_fn=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    count = acquisition.run(logger, max_frames=3, on_frame=terminal_frames.append)

    assert count == 3
    assert [frame.elapsed_s for frame in logger.frames] == [0.0, 1.0, 2.0]
    assert [frame.deadline_miss_ms for frame in logger.frames] == [0.0, 0.0, 0.0]
    assert clock.value == 2.0
    assert logger.flushed
    assert sgp41.heater_was_turned_off
    assert terminal_frames == logger.frames


def test_initialization_failure_flushes_and_turns_off_heater() -> None:
    config = _test_config()
    sgp41 = FailingPrepareSGP41()
    logger = MemoryLogger()
    acquisition = Acquisition(config, Sensors(sgp41=sgp41))

    with pytest.raises(OSError):
        acquisition.run(logger, max_frames=1)

    assert logger.flushed
    assert sgp41.heater_was_turned_off


def test_stable_csv_header_and_sidecar(tmp_path) -> None:
    frame = Frame(
        timestamp_utc="2026-01-01T00:00:00.000Z",
        elapsed_s=0.0,
        sequence=0,
        frame_duration_ms=1.0,
        deadline_miss_ms=0.0,
        sht45=None,
        ads7828=None,
        nh3=None,
        h2s=None,
        sgp41=None,
        bme690=None,
        error_codes=("sht45_crc",),
    )
    with CSVLogger(tmp_path, {"schema_version": 1}, ["sht45"], flush_rows=1) as logger:
        logger.write(frame)
        csv_path = logger.path
        metadata_path = logger.metadata_path

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        row = next(reader)
    assert row["sht45_temperature_c"] == ""
    assert row["sht45_ok"] == "False"
    assert row["error_codes"] == "sht45_crc"
    assert metadata_path.exists()


def test_terminal_format_uses_csv_order_and_marks_missing_values() -> None:
    frame = Frame(
        timestamp_utc="2026-01-01T00:00:00.000Z",
        elapsed_s=0.0,
        sequence=0,
        frame_duration_ms=1.0,
        deadline_miss_ms=0.0,
        sht45=None,
        ads7828=None,
        nh3=None,
        h2s=None,
        sgp41=None,
        bme690=None,
        error_codes=("sgp41_nack",),
    )

    fields = _format_frame(frame).split()

    assert len(fields) == len(CSV_COLUMNS)
    assert fields[0] == "timestamp_utc=2026-01-01T00:00:00.000Z"
    assert "sgp41_sraw_voc=-" in fields
    assert "bme690_temperature_c=-" in fields
    assert fields[-1] == "error_codes=sgp41_nack"


def test_reduced_mode_disables_sgp41_bme690_and_sht45() -> None:
    config = _test_config()

    result = _without_sgp41_bme690_sht45(config)

    assert result.sgp41.enabled is False
    assert result.sgp41.required is False
    assert result.bme690.enabled is False
    assert result.bme690.required is False
    assert result.sht45.enabled is False
    assert result.sht45.required is False
    assert result.ads7828 == config.ads7828
    assert result.nh3 == config.nh3
    assert result.h2s == config.h2s
    assert config.sgp41.enabled is True
    assert config.bme690.enabled is True
    assert config.sht45.enabled is True


def test_reduced_mode_command_name_replaces_old_name() -> None:
    args = _parser().parse_args(
        [
            "acquire-no-sgp41-bme690-sht45",
            "--config",
            "config/rpi5.toml",
            "--frames",
            "1",
        ]
    )

    assert args.command == "acquire-no-sgp41-bme690-sht45"
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["acquire-no-sgp41-bme690", "--config", "config/rpi5.toml"]
        )

    live_args = _parser().parse_args(
        [
            "acquire-classify",
            "--config",
            "config/rpi5.toml",
            "--frames",
            "60",
            "--classification-window-rows",
            "20",
            "--classification-update-rows",
            "3",
            "--display-state",
            "runtime/display_state.json",
        ]
    )
    assert live_args.command == "acquire-classify"
    assert live_args.frames == 60
    assert live_args.classification_window_rows == 20
    assert live_args.classification_update_rows == 3
    assert live_args.display_state.as_posix() == "runtime/display_state.json"

    default_live_args = _parser().parse_args(
        ["acquire-classify", "--config", "config/rpi5.toml"]
    )
    assert default_live_args.classification_window_rows == 60
    assert default_live_args.classification_update_rows == 10

    svm41_args = _parser().parse_args(
        [
            "acquire-no-sgp41-bme690-sht45-with-svm41",
            "--config",
            "config/rpi5.toml",
            "--uart",
            "/dev/ttyUSB1",
            "--frames",
            "30",
        ]
    )
    assert (
        svm41_args.command
        == "acquire-no-sgp41-bme690-sht45-with-svm41"
    )
    assert svm41_args.uart == "/dev/ttyUSB1"
    assert svm41_args.frames == 30

    tgs_svm41_args = _parser().parse_args(
        [
            "acquire-tgs-svm41",
            "--config",
            "config/rpi5.toml",
            "--uart",
            "/dev/ttyUSB2",
            "--frames",
            "20",
        ]
    )
    assert tgs_svm41_args.command == "acquire-tgs-svm41"
    assert tgs_svm41_args.uart == "/dev/ttyUSB2"
    assert tgs_svm41_args.frames == 20


def test_live_classification_failure_does_not_stop_acquisition(caplog) -> None:
    class FailingClassifier:
        def add_frame(self, frame):
            raise ValueError("bad model input")

    frame = Frame(
        timestamp_utc="2026-01-01T00:00:00.000Z",
        elapsed_s=0.0,
        sequence=0,
        frame_duration_ms=1.0,
        deadline_miss_ms=0.0,
        sht45=None,
        ads7828=None,
        nh3=None,
        h2s=None,
        sgp41=None,
        bme690=None,
    )

    _update_live_classification(FailingClassifier(), frame)

    assert "classification failed; acquisition will continue" in caplog.text


def test_live_classification_is_printed_and_published(
    capsys,
    tmp_path,
) -> None:
    frame = Frame(
        timestamp_utc="2026-01-01T00:00:59.000Z",
        elapsed_s=59.0,
        sequence=59,
        frame_duration_ms=1.0,
        deadline_miss_ms=0.0,
        sht45=None,
        ads7828=None,
        nh3=MCP3421Sample(100, 0.0124, 18, 1),
        h2s=MCP3421Sample(50, 0.0032, 18, 1),
        sgp41=None,
        bme690=None,
    )
    result = {
        "raw_rows": 60,
        "valid_rows": 60,
        "window_count": 9,
        "predictions": {
            "food_type": {
                "overall_prediction": "banana",
                "confidence": 0.8,
            },
            "freshness": {
                "overall_prediction": "fresh",
                "confidence": 0.7,
            },
            "combined_class": {
                "overall_prediction": "fresh_banana",
                "confidence": 0.6,
            },
        },
    }

    class ReturningClassifier:
        def add_frame(self, candidate):
            assert candidate is frame
            return result

    state_path = tmp_path / "runtime" / "display_state.json"
    _update_live_classification(ReturningClassifier(), frame, state_path)
    output = capsys.readouterr().out
    assert output.startswith("CLASSIFICATION sequence=59 input_rows=60")
    assert "food_type=banana food_type_confidence=0.8000" in output
    assert "freshness=fresh freshness_confidence=0.7000" in output
    assert "combined_class=fresh_banana" in output

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["combined_class"] == "Fresh Banana"
    assert state["nh3_value"] == pytest.approx(12.4)
    assert state["h2s_value"] == pytest.approx(3.2)
    assert state["system_status"] == "OK"
    assert not list(state_path.parent.glob("*.tmp"))


def test_reduced_mode_omits_disabled_sensors_from_output(tmp_path) -> None:
    frame = Frame(
        timestamp_utc="2026-01-01T00:00:00.000Z",
        elapsed_s=0.0,
        sequence=0,
        frame_duration_ms=1.0,
        deadline_miss_ms=0.0,
        sht45=SHT45Sample(22.0, 40.0),
        ads7828=None,
        nh3=None,
        h2s=None,
        sgp41=SGP41Sample(100, 200, True, 101.0, 2.0),
        bme690=BME690Sample(22.0, 40.0, 101325.0, 50000.0, True, True),
        error_codes=(),
    )

    terminal_line = _format_frame(frame, NO_SGP41_BME690_SHT45_CSV_COLUMNS)
    assert "sgp41" not in terminal_line
    assert "bme690" not in terminal_line
    assert "sht45" not in terminal_line

    with CSVLogger(
        tmp_path,
        {"schema_version": 1},
        ["sht45"],
        flush_rows=1,
        columns=NO_SGP41_BME690_SHT45_CSV_COLUMNS,
    ) as logger:
        logger.write(frame)
        csv_path = logger.path

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == NO_SGP41_BME690_SHT45_CSV_COLUMNS
        next(reader)
    assert not any(
        column.startswith(("sgp41_", "bme690_", "sht45_"))
        for column in NO_SGP41_BME690_SHT45_CSV_COLUMNS
    )

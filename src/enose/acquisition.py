from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, TypeVar

try:
    from sensirion_gas_index_algorithm.nox_algorithm import NoxAlgorithm
    from sensirion_gas_index_algorithm.voc_algorithm import VocAlgorithm
except ImportError:  # Raw SGP41 acquisition remains available without the package.
    NoxAlgorithm = None  # type: ignore[assignment,misc]
    VocAlgorithm = None  # type: ignore[assignment,misc]

from .ads7828 import ADS7828
from .bme690 import BME690
from .config import AppConfig, DeviceConfig
from .csv_logger import CSVLogger
from .i2c_bus import (
    CRCError,
    IdentityError,
    NotReadyError,
    ParseError,
)
from .mcp3421 import MCP3421
from .records import (
    ADS7828Sample,
    BME690Sample,
    Frame,
    MCP3421Sample,
    SGP41Sample,
    SHT45Sample,
    SVM41Sample,
)
from .sgp41 import SGP41
from .sht45 import SHT45
from .svm41_uart import SVM41UART, SVM41_MEASUREMENT_INTERVAL_S

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class Sensors:
    sht45: SHT45 | None = None
    ads7828: ADS7828 | None = None
    sgp41: SGP41 | None = None
    nh3: MCP3421 | None = None
    h2s: MCP3421 | None = None
    bme690: BME690 | None = None
    svm41: SVM41UART | None = None


def error_suffix(exc: Exception) -> str:
    if isinstance(exc, CRCError):
        return "crc"
    if isinstance(exc, NotReadyError):
        return "not_ready"
    if isinstance(exc, IdentityError):
        return "identity"
    if isinstance(exc, ParseError):
        return "parse"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, OSError):
        return "nack"
    return "error"


class Acquisition:
    def __init__(
        self,
        config: AppConfig,
        sensors: Sensors,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        utcnow_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.config = config
        self.sensors = sensors
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._utcnow = utcnow_fn
        self._persistent_errors: list[str] = []
        self._voc_algorithm: Any | None = None
        self._nox_algorithm: Any | None = None

    def _initialize_gas_index_algorithms(self) -> None:
        if VocAlgorithm is None or NoxAlgorithm is None:
            LOGGER.warning(
                "sensirion-gas-index-algorithm is unavailable; "
                "SGP41 index fields will remain empty"
            )
            self._persistent_errors.append("sgp41_index_unavailable")
            return
        try:
            self._voc_algorithm = VocAlgorithm()
            self._nox_algorithm = NoxAlgorithm()
        except Exception:
            self._voc_algorithm = None
            self._nox_algorithm = None
            self._persistent_errors.append("sgp41_index_unavailable")
            LOGGER.exception("failed to initialize SGP41 Gas Index Algorithms")

    def _initialize_device(
        self,
        name: str,
        device_config: DeviceConfig,
        action: Callable[[], object],
        cleanup: Callable[[], object] | None = None,
    ) -> bool:
        try:
            action()
        except Exception as exc:
            LOGGER.exception("%s initialization failed", name)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    LOGGER.exception("%s cleanup after initialization failure failed", name)
            if device_config.required:
                raise
            self._persistent_errors.append(f"{name}_{error_suffix(exc)}")
            setattr(self.sensors, name, None)
            return False
        return True

    def initialize(self) -> None:
        if self.sensors.ads7828 is not None:
            self._initialize_device(
                "ads7828", self.config.ads7828, self.sensors.ads7828.initialize
            )
        if self.sensors.nh3 is not None:
            self._initialize_device("nh3", self.config.nh3, self.sensors.nh3.configure)
        if self.sensors.h2s is not None:
            self._initialize_device("h2s", self.config.h2s, self.sensors.h2s.configure)
        if self.sensors.sgp41 is not None:
            if self._initialize_device(
                "sgp41",
                self.config.sgp41,
                lambda: self.sensors.sgp41.prepare(self.config.sgp41.conditioning_s),
                self.sensors.sgp41.heater_off,
            ):
                self._initialize_gas_index_algorithms()
        if self.sensors.bme690 is not None:
            self._initialize_device(
                "bme690", self.config.bme690, self.sensors.bme690.initialize
            )
        if self.sensors.svm41 is not None:
            try:
                self.sensors.svm41.start()
                self._sleep(SVM41_MEASUREMENT_INTERVAL_S)
            except Exception as exc:
                LOGGER.exception("svm41 initialization failed")
                self._persistent_errors.append(f"svm41_{error_suffix(exc)}")
                try:
                    self.sensors.svm41.close()
                except Exception:
                    LOGGER.exception("svm41 cleanup after initialization failure failed")
                self.sensors.svm41 = None

    def _read(
        self,
        name: str,
        action: Callable[[], T],
        errors: list[str],
    ) -> T | None:
        last_error: Exception | None = None
        for _ in range(self.config.acquisition.retries + 1):
            try:
                return action()
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        errors.append(f"{name}_{error_suffix(last_error)}")
        LOGGER.error(
            "%s read failed",
            name,
            exc_info=(
                type(last_error),
                last_error,
                last_error.__traceback__,
            ),
        )
        return None

    def read_frame(
        self,
        sequence: int,
        run_start: float,
        deadline: float,
    ) -> Frame:
        frame_start = self._monotonic()
        timestamp = self._utcnow().isoformat(timespec="milliseconds").replace("+00:00", "Z")
        errors = list(self._persistent_errors)

        sht45_sample: SHT45Sample | None = None
        if self.sensors.sht45 is not None:
            sht45_sample = self._read("sht45", self.sensors.sht45.read, errors)

        sgp41_sample: SGP41Sample | None = None
        if self.sensors.sgp41 is not None:
            humidity = (
                sht45_sample.relative_humidity_pct if sht45_sample is not None else None
            )
            temperature = sht45_sample.temperature_c if sht45_sample is not None else None
            sgp41_sample = self._read(
                "sgp41",
                lambda: self.sensors.sgp41.measure(humidity, temperature),
                errors,
            )
            if (
                sgp41_sample is not None
                and self._voc_algorithm is not None
                and self._nox_algorithm is not None
            ):
                try:
                    voc_index = self._voc_algorithm.process(sgp41_sample.sraw_voc)
                    nox_index = self._nox_algorithm.process(sgp41_sample.sraw_nox)
                except Exception:
                    errors.append("sgp41_index_error")
                    LOGGER.exception("SGP41 Gas Index Algorithm processing failed")
                else:
                    sgp41_sample = replace(
                        sgp41_sample,
                        voc_index=float(voc_index),
                        nox_index=float(nox_index),
                    )

        ads7828_sample: ADS7828Sample | None = None
        if self.sensors.ads7828 is not None:
            ads7828_sample = self._read("ads7828", self.sensors.ads7828.read_all, errors)
            if ads7828_sample is not None:
                for reading in ads7828_sample.readings:
                    if reading.saturated:
                        errors.append(f"ads7828_saturation_{reading.sensor}")

        nh3_sample: MCP3421Sample | None = None
        if self.sensors.nh3 is not None:
            nh3_sample = self._read("nh3", self.sensors.nh3.read, errors)

        h2s_sample: MCP3421Sample | None = None
        if self.sensors.h2s is not None:
            h2s_sample = self._read("h2s", self.sensors.h2s.read, errors)

        bme690_sample: BME690Sample | None = None
        if self.sensors.bme690 is not None:
            bme690_sample = self._read("bme690", self.sensors.bme690.read, errors)

        svm41_sample: SVM41Sample | None = None
        if self.sensors.svm41 is not None:
            svm41_sample = self._read("svm41", self.sensors.svm41.read, errors)

        frame_end = self._monotonic()
        return Frame(
            timestamp_utc=timestamp,
            elapsed_s=frame_start - run_start,
            sequence=sequence,
            frame_duration_ms=(frame_end - frame_start) * 1000.0,
            deadline_miss_ms=max(0.0, (frame_start - deadline) * 1000.0),
            sht45=sht45_sample,
            ads7828=ads7828_sample,
            nh3=nh3_sample,
            h2s=h2s_sample,
            sgp41=sgp41_sample,
            bme690=bme690_sample,
            svm41=svm41_sample,
            error_codes=tuple(errors),
        )

    def shutdown(self) -> None:
        if self.sensors.sgp41 is not None:
            try:
                self.sensors.sgp41.heater_off()
            except Exception:
                LOGGER.exception("failed to turn off SGP41 heater")
        if self.sensors.svm41 is not None:
            try:
                self.sensors.svm41.stop()
            except Exception:
                LOGGER.exception("failed to stop SVM41 measurement")
            try:
                self.sensors.svm41.close()
            except Exception:
                LOGGER.exception("failed to close SVM41 UART")

    def run(
        self,
        logger: CSVLogger,
        max_frames: int | None = None,
        on_frame: Callable[[Frame], None] | None = None,
    ) -> int:
        sequence = 0
        try:
            self.initialize()
            run_start = self._monotonic()
            while max_frames is None or sequence < max_frames:
                deadline = run_start + sequence * self.config.acquisition.interval_s
                wait_s = deadline - self._monotonic()
                if wait_s > 0:
                    self._sleep(wait_s)
                frame = self.read_frame(sequence, run_start, deadline)
                logger.write(frame)
                if on_frame is not None:
                    on_frame(frame)
                sequence += 1
        finally:
            self.shutdown()
            logger.flush()
        return sequence

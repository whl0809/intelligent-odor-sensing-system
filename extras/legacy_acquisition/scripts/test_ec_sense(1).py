#!/usr/bin/env python3
"""Read EC Sense, six TGS channels, and a Sensirion SVM41.

Hardware interfaces
-------------------
* EC Sense NH3/H2S: MCP3421 ADCs on the Raspberry Pi I2C bus.
* Six TGS channels: the existing TGS acquisition board over USB serial.
* SVM41 evaluation board: its supplied USB-UART cable (SHDLC protocol).

The TGS serial parser accepts these common formats:
* ``100,200,300,400,500,600``
* ``TGS2600=100,TGS2602=200,...``
* ``{"tgs2600": 100, "tgs2602": 200, ...}``
* A CSV header followed by rows of values.

Place this file in ``ECE450_software/tools``. It adds the repository's
``src`` directory to ``sys.path``, so the local ``enose`` package does not
need to be installed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence, TextIO

# Allow direct execution from ECE450_software/tools without pip install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enose.i2c_bus import I2CBus
from enose.mcp3421 import MCP3421


TGS_NAMES = (
    "tgs2600",
    "tgs2602",
    "tgs2603",
    "tgs2610",
    "tgs2611",
    "tgs2620",
)

TGS_ALIASES = {
    "tgs2600": ("tgs2600", "ch1", "channel1", "adc1"),
    "tgs2602": ("tgs2602", "ch2", "channel2", "adc2"),
    "tgs2603": ("tgs2603", "ch3", "channel3", "adc3"),
    "tgs2610": ("tgs2610", "ch4", "channel4", "adc4"),
    "tgs2611": ("tgs2611", "ch5", "channel5", "adc5"),
    "tgs2620": ("tgs2620", "ch6", "channel6", "adc6"),
}

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_i2c_address(value: str) -> int:
    """Parse a decimal or 0x-prefixed I2C address."""
    try:
        address = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid I2C address {value!r}; use a value such as 0x69"
        ) from exc
    if not 0x03 <= address <= 0x77:
        raise argparse.ArgumentTypeError(
            f"I2C address 0x{address:02X} is outside the normal 7-bit range"
        )
    return address


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read EC Sense, six TGS channels, and SVM41 measurements."
    )

    ec_group = parser.add_argument_group("EC Sense (I2C)")
    ec_group.add_argument(
        "--bus", type=int, default=1, help="EC Sense I2C bus (default: 1)"
    )
    ec_group.add_argument(
        "--nh3-address",
        type=parse_i2c_address,
        default=0x6A,
        help="NH3 MCP3421 address (default: 0x6A)",
    )
    ec_group.add_argument(
        "--h2s-address",
        type=parse_i2c_address,
        default=0x69,
        help="H2S MCP3421 address (default: 0x69)",
    )
    ec_group.add_argument(
        "--no-ec", action="store_true", help="disable NH3/H2S acquisition"
    )
    ec_group.add_argument(
        "--skip-probe",
        action="store_true",
        help="skip SMBus quick-write probing and attempt direct MCP3421 reads",
    )

    tgs_group = parser.add_argument_group("TGS board (USB serial)")
    tgs_group.add_argument(
        "--tgs-port",
        default="auto",
        help="TGS serial port, e.g. /dev/ttyACM0; default: auto",
    )
    tgs_group.add_argument(
        "--tgs-baud",
        type=positive_int,
        default=115200,
        help="TGS serial baud rate (default: 115200)",
    )
    tgs_group.add_argument(
        "--tgs-timeout",
        type=positive_float,
        default=2.0,
        help="maximum seconds to wait for a valid TGS row (default: 2.0)",
    )
    tgs_group.add_argument(
        "--no-tgs", action="store_true", help="disable the six TGS channels"
    )

    svm_group = parser.add_argument_group("SVM41 (USB-UART)")
    svm_group.add_argument(
        "--svm41-port",
        default="auto",
        help="SVM41 USB-UART port, e.g. /dev/ttyUSB0; default: auto",
    )
    svm_group.add_argument(
        "--svm41-baud",
        type=positive_int,
        default=115200,
        help="SVM41 UART baud rate (default: 115200)",
    )
    svm_group.add_argument(
        "--no-svm41", action="store_true", help="disable SVM41 acquisition"
    )
    svm_group.add_argument(
        "--no-svm41-reset",
        action="store_true",
        help="do not reset SVM41 before starting measurement",
    )

    output_group = parser.add_argument_group("Sampling and output")
    output_group.add_argument(
        "--interval",
        type=positive_float,
        default=1.0,
        help="seconds between output rows (default: 1.0)",
    )
    output_group.add_argument(
        "--samples",
        type=nonnegative_int,
        default=0,
        help="number of rows; 0 means run until Ctrl+C (default: 0)",
    )
    output_group.add_argument(
        "--csv",
        type=Path,
        help="optional CSV path, e.g. data/raw/all_sensor_test.csv",
    )
    return parser


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def parse_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a sensor value")
    return float(value)


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


class TgsSerialReader:
    """Read and parse six-channel rows from the TGS acquisition board."""

    def __init__(self, port: str, baudrate: int, timeout_s: float) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for the TGS board. Install it with: "
                "python3 -m pip install pyserial"
            ) from exc

        self.port = port
        self.timeout_s = timeout_s
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=min(0.25, timeout_s),
        )
        self._header_indices: tuple[int, ...] | None = None
        self._parsed_header = False
        self._serial.reset_input_buffer()

    def close(self) -> None:
        self._serial.close()

    @staticmethod
    def _values_from_mapping(mapping: dict[Any, Any]) -> tuple[float, ...] | None:
        normalized = {normalize_key(key): value for key, value in mapping.items()}
        values: list[float] = []
        for name in TGS_NAMES:
            found = None
            for alias in TGS_ALIASES[name]:
                key = normalize_key(alias)
                if key in normalized:
                    found = normalized[key]
                    break
            if found is None:
                return None
            values.append(parse_number(found))
        return tuple(values)

    def _parse_delimited(self, line: str) -> tuple[float, ...] | None:
        self._parsed_header = False
        delimiter = None
        for candidate in (",", ";", "\t"):
            if candidate in line:
                delimiter = candidate
                break
        if delimiter is None:
            return None

        fields = next(csv.reader([line], delimiter=delimiter))
        fields = [field.strip() for field in fields]

        # A header can contain extra timestamp/status columns. Remember only
        # the positions of the six TGS columns.
        normalized_fields = [normalize_key(field) for field in fields]
        indices: list[int] = []
        for name in TGS_NAMES:
            index = None
            aliases = {normalize_key(alias) for alias in TGS_ALIASES[name]}
            for candidate_index, field in enumerate(normalized_fields):
                if field in aliases:
                    index = candidate_index
                    break
            if index is None:
                indices = []
                break
            indices.append(index)
        if len(indices) == len(TGS_NAMES):
            self._header_indices = tuple(indices)
            self._parsed_header = True
            return None

        if self._header_indices is not None:
            try:
                return tuple(parse_number(fields[index]) for index in self._header_indices)
            except (IndexError, ValueError):
                pass

        if len(fields) == len(TGS_NAMES):
            try:
                return tuple(parse_number(field) for field in fields)
            except ValueError:
                return None
        return None

    def parse_line(self, line: str) -> tuple[float, ...] | None:
        line = line.strip()
        if not line:
            return None

        # JSON object or six-element JSON list.
        if line[0] in "[{":
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                result = self._values_from_mapping(payload)
                if result is not None:
                    return result
            if isinstance(payload, list) and len(payload) == len(TGS_NAMES):
                try:
                    return tuple(parse_number(value) for value in payload)
                except ValueError:
                    pass

        # Labeled values such as "TGS2600=123" or "CH1:123".
        labeled: dict[str, float] = {}
        for name in TGS_NAMES:
            for alias in TGS_ALIASES[name]:
                pattern = (
                    rf"(?<![A-Za-z0-9]){re.escape(alias)}"
                    rf"\s*[:=]\s*({NUMBER_PATTERN})"
                )
                match = re.search(pattern, line, flags=re.IGNORECASE)
                if match is not None:
                    labeled[name] = float(match.group(1))
                    break
        if len(labeled) == len(TGS_NAMES):
            return tuple(labeled[name] for name in TGS_NAMES)

        delimited = self._parse_delimited(line)
        if delimited is not None:
            return delimited
        if self._parsed_header:
            return None

        # Also accept six whitespace-separated numbers, optionally following
        # a prefix such as "ADC:" or "TGS:".
        numeric_part = line.rsplit(":", 1)[-1]
        tokens = re.findall(NUMBER_PATTERN, numeric_part)
        if len(tokens) == len(TGS_NAMES):
            return tuple(float(token) for token in tokens)
        return None

    def read(self) -> tuple[float, ...]:
        deadline = time.monotonic() + self.timeout_s
        last_line = ""
        while time.monotonic() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            last_line = raw.decode("utf-8", errors="replace").strip()
            parsed = self.parse_line(last_line)
            if parsed is not None:
                return parsed
        detail = f" Last line: {last_line!r}" if last_line else ""
        raise TimeoutError(
            f"no valid six-channel TGS row from {self.port} "
            f"within {self.timeout_s:.1f} s.{detail}"
        )


class Svm41UartReader:
    """Read processed and raw SVM41 values using Sensirion's UART driver."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        stack: ExitStack,
        reset: bool,
    ) -> None:
        try:
            from sensirion_driver_adapters.shdlc_adapter.shdlc_channel import (
                ShdlcChannel,
            )
            from sensirion_shdlc_driver import ShdlcSerialPort
            from sensirion_uart_svm4x.device import Svm4xDevice
        except ImportError as exc:
            raise RuntimeError(
                "Sensirion's SVM41 UART driver is required. Install it with: "
                "python3 -m pip install sensirion-uart-svm4x"
            ) from exc

        serial_port = stack.enter_context(
            ShdlcSerialPort(port=port, baudrate=baudrate)
        )
        self.port = port
        self.sensor = Svm4xDevice(ShdlcChannel(serial_port))

        if reset:
            self.sensor.device_reset()
            time.sleep(2.0)

        self.serial_number = str(self.sensor.get_serial_number()).rstrip("\x00")
        self.product_name = str(self.sensor.get_product_name()).rstrip("\x00")
        self.sensor.start_measurement()
        stack.callback(self._stop_safely)

        # SVM41 publishes the first non-zero measurement after one second.
        time.sleep(1.1)

    def _stop_safely(self) -> None:
        try:
            self.sensor.stop_measurement()
        except Exception:
            pass

    def read(self) -> dict[str, float | int]:
        humidity, temperature, voc_index, nox_index = (
            self.sensor.read_measured_values()
        )
        raw_humidity, raw_temperature, raw_voc, raw_nox = (
            self.sensor.read_measured_raw_values()
        )
        return {
            "humidity_rh": float(humidity),
            "temperature_c": float(temperature),
            "voc_index": float(voc_index),
            "nox_index": float(nox_index),
            "raw_humidity_rh": float(raw_humidity) / 100.0,
            "raw_temperature_c": float(raw_temperature) / 200.0,
            "raw_voc_ticks": int(raw_voc),
            "raw_nox_ticks": int(raw_nox),
        }


def serial_candidates() -> list[str]:
    candidates: list[str] = []
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    ):
        candidates.extend(glob.glob(pattern))
    # Resolve by-id links while preserving stable by-id names first.
    unique: list[str] = []
    resolved: set[str] = set()
    for candidate in sorted(candidates):
        real_path = str(Path(candidate).resolve())
        if real_path not in resolved:
            unique.append(candidate)
            resolved.add(real_path)
    return unique


def resolve_serial_port(
    requested: str,
    purpose: str,
    excluded_real_paths: set[str],
) -> str:
    if requested != "auto":
        path = str(Path(requested))
        real_path = str(Path(path).resolve())
        if real_path in excluded_real_paths:
            raise RuntimeError(f"{purpose} port {path} is already in use")
        if not Path(path).exists():
            raise FileNotFoundError(f"{purpose} serial port does not exist: {path}")
        return path

    available = [
        path
        for path in serial_candidates()
        if str(Path(path).resolve()) not in excluded_real_paths
    ]
    if not available:
        raise FileNotFoundError(
            f"no unused serial port found for {purpose}; connect the USB cable "
            f"or pass --{purpose.lower()}-port explicitly"
        )

    # TGS boards are commonly ttyACM devices, while the SVM41 USB-UART cable
    # is commonly ttyUSB. The fallback remains deterministic.
    preferred_token = "ttyACM" if purpose == "TGS" else "ttyUSB"
    for path in available:
        if preferred_token in str(Path(path).resolve()):
            return path
    return available[0]


CSV_HEADER = [
    "timestamp",
    "monotonic_s",
    "tgs2600_raw",
    "tgs2602_raw",
    "tgs2603_raw",
    "tgs2610_raw",
    "tgs2611_raw",
    "tgs2620_raw",
    "nh3_raw",
    "nh3_voltage_v",
    "nh3_voltage_mv",
    "h2s_raw",
    "h2s_voltage_v",
    "h2s_voltage_mv",
    "svm41_humidity_rh",
    "svm41_temperature_c",
    "svm41_voc_index",
    "svm41_nox_index",
    "svm41_raw_humidity_rh",
    "svm41_raw_temperature_c",
    "svm41_raw_voc_ticks",
    "svm41_raw_nox_ticks",
]


def open_csv(
    path: Path | None, stack: ExitStack
) -> tuple[TextIO | None, csv.writer | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
    writer = csv.writer(handle)
    writer.writerow(CSV_HEADER)
    handle.flush()
    return handle, writer


def optional_csv_values(
    tgs: Sequence[float] | None,
    nh3: Any | None,
    h2s: Any | None,
    svm41: dict[str, float | int] | None,
) -> list[Any]:
    row: list[Any] = []
    row.extend("" if tgs is None else format_number(float(value)) for value in (tgs or [0] * 6))
    if nh3 is None or h2s is None:
        row.extend([""] * 6)
    else:
        nh3_mv = nh3.differential_voltage_v * 1000.0
        h2s_mv = h2s.differential_voltage_v * 1000.0
        row.extend(
            [
                nh3.raw,
                f"{nh3.differential_voltage_v:.9f}",
                f"{nh3_mv:.6f}",
                h2s.raw,
                f"{h2s.differential_voltage_v:.9f}",
                f"{h2s_mv:.6f}",
            ]
        )
    if svm41 is None:
        row.extend([""] * 8)
    else:
        row.extend(
            [
                f"{float(svm41['humidity_rh']):.3f}",
                f"{float(svm41['temperature_c']):.3f}",
                f"{float(svm41['voc_index']):.3f}",
                f"{float(svm41['nox_index']):.3f}",
                f"{float(svm41['raw_humidity_rh']):.3f}",
                f"{float(svm41['raw_temperature_c']):.3f}",
                int(svm41["raw_voc_ticks"]),
                int(svm41["raw_nox_ticks"]),
            ]
        )
    return row


def main() -> int:
    args = build_parser().parse_args()

    if not args.no_ec and args.nh3_address == args.h2s_address:
        print("ERROR: NH3 and H2S addresses must be different.", file=sys.stderr)
        return 2
    if args.no_ec and args.no_tgs and args.no_svm41:
        print("ERROR: all sensor groups are disabled.", file=sys.stderr)
        return 2

    print("ECE450 combined sensor test")
    print(f"Sampling interval: {args.interval:.3f} s")

    try:
        with ExitStack() as stack:
            csv_handle, csv_writer = open_csv(args.csv, stack)

            used_serial_paths: set[str] = set()
            tgs_reader: TgsSerialReader | None = None
            if not args.no_tgs:
                tgs_port = resolve_serial_port(
                    args.tgs_port, "TGS", used_serial_paths
                )
                tgs_reader = TgsSerialReader(
                    tgs_port, args.tgs_baud, args.tgs_timeout
                )
                stack.callback(tgs_reader.close)
                used_serial_paths.add(str(Path(tgs_port).resolve()))
                print(f"TGS board: {tgs_port} @ {args.tgs_baud} baud")
                print("TGS order: " + ", ".join(name.upper() for name in TGS_NAMES))

            svm41_reader: Svm41UartReader | None = None
            if not args.no_svm41:
                svm41_port = resolve_serial_port(
                    args.svm41_port, "SVM41", used_serial_paths
                )
                svm41_reader = Svm41UartReader(
                    svm41_port,
                    args.svm41_baud,
                    stack,
                    reset=not args.no_svm41_reset,
                )
                used_serial_paths.add(str(Path(svm41_port).resolve()))
                print(
                    f"SVM41: {svm41_reader.product_name} "
                    f"SN={svm41_reader.serial_number} on {svm41_port}"
                )

            ec_sensors: dict[str, MCP3421] = {}
            if not args.no_ec:
                bus = stack.enter_context(I2CBus(args.bus))
                sensor_specs = (
                    ("NH3", args.nh3_address),
                    ("H2S", args.h2s_address),
                )
                print(f"EC Sense I2C device: /dev/i2c-{args.bus}")
                if not args.skip_probe:
                    print("EC Sense address probe:")
                    for name, address in sensor_specs:
                        result = "ACK" if bus.probe(address) else "NACK"
                        print(f"  {name}: 0x{address:02X} -> {result}")
                    print(
                        "A quick-probe NACK is not fatal; direct MCP3421 "
                        "transfers will still be attempted."
                    )

                for name, address in sensor_specs:
                    sensor = MCP3421(
                        bus=bus,
                        address=address,
                        resolution_bits=18,
                        gain=1,
                        continuous=True,
                    )
                    sensor.configure()
                    ec_sensors[name] = sensor
                    print(
                        f"Configured {name} at 0x{address:02X}; "
                        f"config=0x{sensor.config_byte:02X}"
                    )
                time.sleep(0.35)

            print("\nPress Ctrl+C to stop.\n")
            sample_index = 0
            while args.samples == 0 or sample_index < args.samples:
                cycle_start = time.monotonic()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                tgs_values = tgs_reader.read() if tgs_reader is not None else None
                nh3 = ec_sensors["NH3"].read() if ec_sensors else None
                h2s = ec_sensors["H2S"].read() if ec_sensors else None
                svm41_values = (
                    svm41_reader.read() if svm41_reader is not None else None
                )

                sections = [timestamp]
                if tgs_values is not None:
                    sections.append(
                        "TGS=[" + ", ".join(format_number(v) for v in tgs_values) + "]"
                    )
                if nh3 is not None and h2s is not None:
                    sections.append(
                        f"NH3={nh3.raw:+d}/{nh3.differential_voltage_v * 1000:+.3f}mV"
                    )
                    sections.append(
                        f"H2S={h2s.raw:+d}/{h2s.differential_voltage_v * 1000:+.3f}mV"
                    )
                if svm41_values is not None:
                    sections.append(
                        "SVM41="
                        f"{float(svm41_values['temperature_c']):.2f}C/"
                        f"{float(svm41_values['humidity_rh']):.2f}%RH "
                        f"VOC={float(svm41_values['voc_index']):.1f}"
                        f"({int(svm41_values['raw_voc_ticks'])}) "
                        f"NOx={float(svm41_values['nox_index']):.1f}"
                        f"({int(svm41_values['raw_nox_ticks'])})"
                    )
                print("  ".join(sections), flush=True)

                if csv_writer is not None and csv_handle is not None:
                    csv_writer.writerow(
                        [timestamp, f"{cycle_start:.6f}"]
                        + optional_csv_values(
                            tgs_values, nh3, h2s, svm41_values
                        )
                    )
                    csv_handle.flush()

                sample_index += 1
                elapsed = time.monotonic() - cycle_start
                time.sleep(max(0.0, args.interval - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except PermissionError as exc:
        print(
            f"ERROR: permission denied: {exc}\n"
            "Add your user to the i2c and dialout groups, then log out and in.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"ERROR: hardware communication failed: {exc}\n"
            "Check power, common ground, USB cables, I2C wiring, and ports.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.csv is not None:
        print(f"CSV saved to: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

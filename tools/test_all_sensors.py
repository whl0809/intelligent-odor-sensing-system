#!/usr/bin/env python3
"""Acquire all six TGS channels, EC Sense NH3/H2S, and an external SVM41.

Hardware communication follows AGENTS.md:

* TGS board: ADS7828 at I2C address 0x48 on /dev/i2c-1.
* NH3: MCP3421 at I2C address 0x69.
* H2S: MCP3421 at I2C address 0x6A.
* External SVM41: USB-UART/SHDLC at 115200 baud.

The SVM41 must not be connected to the shared I2C bus because its I2C address
0x6A conflicts with the H2S MCP3421. TGS readings are ADC codes and voltages,
not ppm.

Place this file in ``ECE450_software/tools``. It adds the repository's ``src``
directory to ``sys.path`` so the local ``enose`` package need not be installed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

# Allow direct execution from ECE450_software/tools without pip installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enose.i2c_bus import I2CBus
from enose.mcp3421 import MCP3421


ADS7828_CHANNEL_COMMANDS = {
    0: 0x8C,
    1: 0xCC,
    2: 0x9C,
    3: 0xDC,
    4: 0xAC,
    5: 0xEC,
    6: 0xBC,
    7: 0xFC,
}

# Stable sensor/CSV order required by AGENTS.md.
TGS_CHANNELS = (
    ("tgs2620", 0),
    ("tgs2610", 1),
    ("tgs2611", 2),
    ("tgs2600", 3),
    ("tgs2602", 4),
    ("tgs2603", 5),
)

TGS_NEAR_RAIL_LOW = 4
TGS_NEAR_RAIL_HIGH = 4091


@dataclass(frozen=True)
class Ads7828Sample:
    raw: int
    voltage_v: float
    near_rail: bool


@dataclass(frozen=True)
class Svm41Sample:
    humidity_rh: float
    temperature_c: float
    voc_index: float
    nox_index: float
    raw_humidity_rh: float
    raw_temperature_c: float
    raw_voc_ticks: int
    raw_nox_ticks: int


class Ads7828:
    """Small ADS7828 I2C driver for the six single-ended TGS channels."""

    REFERENCE_V = 2.5
    FULL_SCALE_CODES = 4096

    def __init__(self, bus_number: int, address: int) -> None:
        try:
            from smbus2 import SMBus, i2c_msg
        except ImportError as exc:
            raise RuntimeError(
                "smbus2 is required. Install it with: "
                "sudo apt install -y python3-smbus2"
            ) from exc

        self.address = address
        self._i2c_msg = i2c_msg
        self._bus = SMBus(bus_number)

        # Enable the internal reference, allow it to settle, and discard the
        # first conversion as required by the ADS7828 startup procedure.
        self._send_command(ADS7828_CHANNEL_COMMANDS[0])
        time.sleep(0.002)
        self.read_channel(0)

    def close(self) -> None:
        self._bus.close()

    def _send_command(self, command: int) -> None:
        message = self._i2c_msg.write(self.address, [command])
        self._bus.i2c_rdwr(message)

    def _transfer(self, command: int) -> tuple[int, int]:
        write_message = self._i2c_msg.write(self.address, [command])
        read_message = self._i2c_msg.read(self.address, 2)
        self._bus.i2c_rdwr(write_message, read_message)
        data = list(read_message)
        if len(data) != 2:
            raise OSError(f"ADS7828 returned {len(data)} bytes; expected 2")
        return int(data[0]), int(data[1])

    @staticmethod
    def parse_raw(byte0: int, byte1: int) -> int:
        return ((byte0 & 0x0F) << 8) | byte1

    def read_channel(self, channel: int) -> Ads7828Sample:
        try:
            command = ADS7828_CHANNEL_COMMANDS[channel]
        except KeyError as exc:
            raise ValueError(f"invalid ADS7828 channel: {channel}") from exc
        byte0, byte1 = self._transfer(command)
        raw = self.parse_raw(byte0, byte1)
        voltage_v = raw * self.REFERENCE_V / self.FULL_SCALE_CODES
        near_rail = raw <= TGS_NEAR_RAIL_LOW or raw >= TGS_NEAR_RAIL_HIGH
        return Ads7828Sample(raw=raw, voltage_v=voltage_v, near_rail=near_rail)

    def read_tgs(self) -> dict[str, Ads7828Sample]:
        return {
            sensor_name: self.read_channel(channel)
            for sensor_name, channel in TGS_CHANNELS
        }


class Svm41UartReader:
    """Read processed and raw SVM41 values with Sensirion's SHDLC driver."""

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
                "python3 -m pip install --user --break-system-packages "
                "sensirion-uart-svm4x"
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

        # First non-zero SVM41 measurement is available after about one second.
        time.sleep(1.1)

    def _stop_safely(self) -> None:
        try:
            self.sensor.stop_measurement()
        except Exception:
            pass

    def read(self) -> Svm41Sample:
        humidity, temperature, voc_index, nox_index = (
            self.sensor.read_measured_values()
        )
        raw_humidity, raw_temperature, raw_voc, raw_nox = (
            self.sensor.read_measured_raw_values()
        )
        return Svm41Sample(
            humidity_rh=float(humidity),
            temperature_c=float(temperature),
            voc_index=float(voc_index),
            nox_index=float(nox_index),
            raw_humidity_rh=float(raw_humidity) / 100.0,
            raw_temperature_c=float(raw_temperature) / 200.0,
            raw_voc_ticks=int(raw_voc),
            raw_nox_ticks=int(raw_nox),
        )


def parse_i2c_address(value: str) -> int:
    try:
        address = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid I2C address {value!r}; use a value such as 0x69"
        ) from exc
    if not 0x03 <= address <= 0x77:
        raise argparse.ArgumentTypeError(
            f"I2C address 0x{address:02X} is outside the 7-bit range"
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
        description=(
            "Read six ADS7828/TGS channels and EC Sense over I2C, plus "
            "SVM41 over USB-UART."
        )
    )

    i2c_group = parser.add_argument_group("Shared I2C bus")
    i2c_group.add_argument(
        "--bus", type=int, default=1, help="I2C bus number (default: 1)"
    )
    i2c_group.add_argument(
        "--ads-address",
        type=parse_i2c_address,
        default=0x48,
        help="ADS7828 address (default: 0x48)",
    )
    i2c_group.add_argument(
        "--no-tgs", action="store_true", help="disable ADS7828/TGS acquisition"
    )
    i2c_group.add_argument(
        "--nh3-address",
        type=parse_i2c_address,
        default=0x69,
        help="NH3 MCP3421 address (default: 0x69)",
    )
    i2c_group.add_argument(
        "--h2s-address",
        type=parse_i2c_address,
        default=0x6A,
        help="H2S MCP3421 address (default: 0x6A)",
    )
    i2c_group.add_argument(
        "--no-ec", action="store_true", help="disable NH3/H2S acquisition"
    )
    i2c_group.add_argument(
        "--ec-resolution",
        type=int,
        choices=(12, 14, 16, 18),
        default=18,
        help="MCP3421 resolution in bits (default: 18)",
    )
    i2c_group.add_argument(
        "--ec-gain",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help="MCP3421 gain (default: 1)",
    )
    i2c_group.add_argument(
        "--ec-one-shot",
        action="store_true",
        help="use MCP3421 one-shot mode instead of continuous mode",
    )
    i2c_group.add_argument(
        "--skip-probe",
        action="store_true",
        help="skip quick-write probing and attempt direct device reads",
    )

    svm_group = parser.add_argument_group("External SVM41 (USB-UART)")
    svm_group.add_argument(
        "--svm41-port",
        default="auto",
        help="SVM41 UART port, e.g. /dev/ttyUSB0; default: auto",
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
        help="seconds between frame deadlines (default: 1.0)",
    )
    output_group.add_argument(
        "--samples",
        type=nonnegative_int,
        default=0,
        help="number of frames; 0 means until Ctrl+C (default: 0)",
    )
    output_group.add_argument(
        "--csv",
        type=Path,
        help="optional CSV path, e.g. data/raw/all_sensor_test.csv",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="show additional startup diagnostics",
    )
    return parser


def serial_candidates() -> list[str]:
    candidates: list[str] = []
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ):
        candidates.extend(glob.glob(pattern))

    unique: list[str] = []
    resolved: set[str] = set()
    for candidate in sorted(candidates):
        real_path = str(Path(candidate).resolve())
        if real_path not in resolved:
            unique.append(candidate)
            resolved.add(real_path)
    return unique


def resolve_svm41_port(requested: str) -> str:
    if requested != "auto":
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(f"SVM41 serial port does not exist: {path}")
        return str(path)

    candidates = serial_candidates()
    if not candidates:
        raise FileNotFoundError(
            "no SVM41 serial port found; connect the Sensirion USB-UART "
            "cable or pass --svm41-port explicitly"
        )

    for candidate in candidates:
        if "sensirion" in candidate.lower():
            return candidate
    for candidate in candidates:
        if "ttyUSB" in str(Path(candidate).resolve()):
            return candidate
    return candidates[0]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


CSV_HEADER = [
    "timestamp_utc",
    "elapsed_s",
    "sequence",
    "frame_duration_ms",
    "deadline_miss_ms",
]
for _sensor_name, _channel in TGS_CHANNELS:
    CSV_HEADER.extend(
        [
            f"{_sensor_name}_raw",
            f"{_sensor_name}_voltage_v",
        ]
    )
CSV_HEADER.extend(
    [
        "ads7828_ok",
        "nh3_raw",
        "nh3_diff_voltage_v",
        "nh3_ok",
        "h2s_raw",
        "h2s_diff_voltage_v",
        "h2s_ok",
        "svm41_temperature_c",
        "svm41_relative_humidity_pct",
        "svm41_voc_index",
        "svm41_nox_index",
        "svm41_raw_temperature_c",
        "svm41_raw_humidity_pct",
        "svm41_raw_voc_ticks",
        "svm41_raw_nox_ticks",
        "svm41_ok",
        "error_codes",
    ]
)


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


def append_tgs_csv(
    row: list[Any], tgs: dict[str, Ads7828Sample] | None
) -> None:
    for sensor_name, _channel in TGS_CHANNELS:
        sample = None if tgs is None else tgs.get(sensor_name)
        if sample is None:
            row.extend(["", ""])
        else:
            row.extend([sample.raw, f"{sample.voltage_v:.9f}"])


def append_ec_csv(row: list[Any], sample: Any | None) -> None:
    if sample is None:
        row.extend(["", "", 0])
    else:
        row.extend([sample.raw, f"{sample.differential_voltage_v:.9f}", 1])


def append_svm41_csv(row: list[Any], sample: Svm41Sample | None) -> None:
    if sample is None:
        row.extend([""] * 8 + [0])
    else:
        row.extend(
            [
                f"{sample.temperature_c:.3f}",
                f"{sample.humidity_rh:.3f}",
                f"{sample.voc_index:.3f}",
                f"{sample.nox_index:.3f}",
                f"{sample.raw_temperature_c:.3f}",
                f"{sample.raw_humidity_rh:.3f}",
                sample.raw_voc_ticks,
                sample.raw_nox_ticks,
                1,
            ]
        )


def main() -> int:
    args = build_parser().parse_args()

    if not args.no_ec and args.nh3_address == args.h2s_address:
        print("ERROR: NH3 and H2S addresses must differ.", file=sys.stderr)
        return 2
    if args.no_tgs and args.no_ec and args.no_svm41:
        print("ERROR: all sensor groups are disabled.", file=sys.stderr)
        return 2

    print("ECE450 combined sensor acquisition")
    print(f"Frame interval: {args.interval:.3f} s")
    print(
        "Communication: TGS=ADS7828/I2C, EC Sense=MCP3421/I2C, "
        "SVM41=UART/SHDLC"
    )

    try:
        with ExitStack() as stack:
            csv_handle, csv_writer = open_csv(args.csv, stack)

            ads_reader: Ads7828 | None = None
            if not args.no_tgs:
                ads_reader = Ads7828(args.bus, args.ads_address)
                stack.callback(ads_reader.close)
                print(
                    f"TGS board: /dev/i2c-{args.bus}, "
                    f"ADS7828 at 0x{args.ads_address:02X}"
                )
                print(
                    "TGS mapping: "
                    + ", ".join(
                        f"CH{channel}={name.upper()}"
                        for name, channel in TGS_CHANNELS
                    )
                )

            ec_sensors: dict[str, MCP3421] = {}
            if not args.no_ec:
                ec_bus = stack.enter_context(I2CBus(args.bus))
                sensor_specs = (
                    ("NH3", args.nh3_address),
                    ("H2S", args.h2s_address),
                )
                print(f"EC Sense: /dev/i2c-{args.bus}")
                if not args.skip_probe:
                    probe_results = []
                    for name, address in sensor_specs:
                        result = "ACK" if ec_bus.probe(address) else "NACK"
                        probe_results.append(f"{name}=0x{address:02X}:{result}")
                    print("EC probe: " + ", ".join(probe_results))

                for name, address in sensor_specs:
                    sensor = MCP3421(
                        bus=ec_bus,
                        address=address,
                        resolution_bits=args.ec_resolution,
                        gain=args.ec_gain,
                        continuous=not args.ec_one_shot,
                    )
                    sensor.configure()
                    ec_sensors[name] = sensor
                    if args.verbose:
                        print(
                            f"Configured {name} at 0x{address:02X}; "
                            f"config=0x{sensor.config_byte:02X}"
                        )
                if args.ec_resolution == 18:
                    time.sleep(0.35)

            svm41_reader: Svm41UartReader | None = None
            if not args.no_svm41:
                svm41_port = resolve_svm41_port(args.svm41_port)
                svm41_reader = Svm41UartReader(
                    svm41_port,
                    args.svm41_baud,
                    stack,
                    reset=not args.no_svm41_reset,
                )
                print(
                    f"SVM41: {svm41_reader.product_name} "
                    f"SN={svm41_reader.serial_number} on {svm41_port}"
                )

            print("\nPress Ctrl+C to stop.\n")
            start_monotonic = time.monotonic()
            next_deadline = start_monotonic
            sequence = 0

            while args.samples == 0 or sequence < args.samples:
                frame_start = time.monotonic()
                deadline_miss_ms = max(
                    0.0, (frame_start - next_deadline) * 1000.0
                )
                timestamp = utc_timestamp()
                errors: list[str] = []

                tgs_values: dict[str, Ads7828Sample] | None = None
                if ads_reader is not None:
                    try:
                        tgs_values = ads_reader.read_tgs()
                        for sensor_name, sample in tgs_values.items():
                            if sample.near_rail:
                                errors.append(f"{sensor_name.upper()}_NEAR_RAIL")
                    except Exception as exc:
                        errors.append(f"ADS7828_{type(exc).__name__.upper()}")
                        if args.verbose:
                            print(f"ADS7828 read failed: {exc}", file=sys.stderr)

                nh3 = None
                if "NH3" in ec_sensors:
                    try:
                        nh3 = ec_sensors["NH3"].read()
                    except Exception as exc:
                        errors.append(f"NH3_{type(exc).__name__.upper()}")
                        if args.verbose:
                            print(f"NH3 read failed: {exc}", file=sys.stderr)

                h2s = None
                if "H2S" in ec_sensors:
                    try:
                        h2s = ec_sensors["H2S"].read()
                    except Exception as exc:
                        errors.append(f"H2S_{type(exc).__name__.upper()}")
                        if args.verbose:
                            print(f"H2S read failed: {exc}", file=sys.stderr)

                svm41_values = None
                if svm41_reader is not None:
                    try:
                        svm41_values = svm41_reader.read()
                    except Exception as exc:
                        errors.append(f"SVM41_{type(exc).__name__.upper()}")
                        if args.verbose:
                            print(f"SVM41 read failed: {exc}", file=sys.stderr)

                frame_end = time.monotonic()
                elapsed_s = frame_start - start_monotonic
                frame_duration_ms = (frame_end - frame_start) * 1000.0

                sections = [
                    timestamp,
                    f"seq={sequence}",
                    f"frame={frame_duration_ms:.1f}ms",
                ]
                if tgs_values is not None:
                    tgs_text = []
                    for sensor_name, _channel in TGS_CHANNELS:
                        sample = tgs_values[sensor_name]
                        flag = "!" if sample.near_rail else ""
                        tgs_text.append(
                            f"{sensor_name.upper()}={sample.raw}"
                            f"/{sample.voltage_v:.4f}V{flag}"
                        )
                    sections.append("TGS=[" + ", ".join(tgs_text) + "]")
                if nh3 is not None:
                    sections.append(
                        f"NH3={nh3.raw:+d}/"
                        f"{nh3.differential_voltage_v * 1000:+.3f}mV"
                    )
                if h2s is not None:
                    sections.append(
                        f"H2S={h2s.raw:+d}/"
                        f"{h2s.differential_voltage_v * 1000:+.3f}mV"
                    )
                if svm41_values is not None:
                    sections.append(
                        f"SVM41={svm41_values.temperature_c:.2f}C/"
                        f"{svm41_values.humidity_rh:.2f}%RH "
                        f"VOC={svm41_values.voc_index:.1f}"
                        f"({svm41_values.raw_voc_ticks}) "
                        f"NOx={svm41_values.nox_index:.1f}"
                        f"({svm41_values.raw_nox_ticks})"
                    )
                if errors:
                    sections.append("ERR=" + ";".join(errors))
                print("  ".join(sections), flush=True)

                if csv_writer is not None:
                    row: list[Any] = [
                        timestamp,
                        f"{elapsed_s:.6f}",
                        sequence,
                        f"{frame_duration_ms:.3f}",
                        f"{deadline_miss_ms:.3f}",
                    ]
                    append_tgs_csv(row, tgs_values)
                    row.append(1 if tgs_values is not None else 0)
                    append_ec_csv(row, nh3)
                    append_ec_csv(row, h2s)
                    append_svm41_csv(row, svm41_values)
                    row.append(";".join(errors))
                    csv_writer.writerow(row)
                    if csv_handle is not None and (sequence + 1) % 10 == 0:
                        csv_handle.flush()

                sequence += 1
                next_deadline += args.interval
                remaining_s = next_deadline - time.monotonic()
                if remaining_s > 0:
                    time.sleep(remaining_s)

            if csv_handle is not None:
                csv_handle.flush()

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
    except OSError as exc:
        print(
            f"ERROR: hardware communication failed: {exc}\n"
            "Check both board power supplies, common ground, SDA/SCL wiring, "
            "the SVM41 USB-UART cable, and I2C addresses.",
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

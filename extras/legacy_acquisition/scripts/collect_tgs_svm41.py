#!/usr/bin/env python3
"""Acquire six TGS channels and an external Sensirion SVM41.

Hardware communication follows AGENTS.md:

* TGS board: ADS7828 at I2C address 0x48 on /dev/i2c-1.
* External SVM41: USB-UART/SHDLC at 115200 baud.

The external SVM41 is read through its Sensirion USB-UART cable. TGS readings
are ADC codes and voltages, not ppm.

Place this file in ``ECE450_software/enose_tgs_svm41_v2/tools``.  CSV recording
starts only after both SVM41 gas indices are non-zero, so ``--samples`` counts
usable post-warm-up frames rather than the approximately 45-second zero period.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
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

# Dataset labels currently used by the food/freshness model.  The interactive
# menu also has a custom option, so future food groups do not require a code
# change.
FOOD_GROUPS = (
    "blank",
    "fresh_banana",
    "fermented_banana",
    "fresh_meat",
    "spoiled_meat",
)


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


def svm41_gas_indices_ready(sample: Svm41Sample | None) -> bool:
    """Return true only for post-warm-up processed gas-index readings."""
    return bool(
        sample is not None
        and math.isfinite(sample.voc_index)
        and math.isfinite(sample.nox_index)
        and sample.voc_index > 0.0
        and sample.nox_index > 0.0
    )


def wait_for_svm41_ready(
    reader: Svm41UartReader,
    timeout_s: float,
    poll_interval_s: float,
    verbose: bool,
) -> None:
    """Wait until VOC and NOx indices have both left their zero warm-up state."""
    started = time.monotonic()
    attempts = 0
    print(
        "Waiting for SVM41 VOC/NOx indices to become non-zero "
        f"(timeout {timeout_s:.0f} s)..."
    )
    while True:
        attempts += 1
        try:
            sample = reader.read()
            if svm41_gas_indices_ready(sample):
                elapsed = time.monotonic() - started
                print(
                    f"SVM41 ready after {elapsed:.1f} s: "
                    f"VOC={sample.voc_index:.1f}, NOx={sample.nox_index:.1f}"
                )
                return
            if verbose or attempts == 1 or attempts % 10 == 0:
                print(
                    f"  warm-up {time.monotonic() - started:.1f} s: "
                    f"VOC={sample.voc_index:.1f}, NOx={sample.nox_index:.1f}"
                )
        except Exception as exc:
            if verbose:
                print(f"  SVM41 warm-up read failed: {exc}", file=sys.stderr)

        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            raise TimeoutError(
                "SVM41 VOC/NOx indices remained zero or unreadable for "
                f"{timeout_s:.0f} seconds"
            )
        time.sleep(min(poll_interval_s, max(0.0, timeout_s - elapsed)))


def safe_name_component(value: str, field_name: str) -> str:
    """Convert a user label into a safe lowercase filename component."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def infer_identity_from_filename(filename: str) -> tuple[str, str]:
    """Infer food group and sample ID from enose_<group>_<id>.csv."""
    stem = Path(filename).stem
    if stem.lower().startswith("enose_"):
        stem = stem[6:]
    stem = safe_name_component(stem, "filename")
    # A timestamp identifies a recording, not its class.  In this case the
    # caller must provide --food-group or choose it from the interactive menu.
    if re.fullmatch(r"\d{8}t\d{6}(?:_\d+z?)?", stem, re.IGNORECASE):
        return "", stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return stem, ""


def next_sample_id(output_dir: Path, food_group: str) -> str:
    """Return the next unused numeric ID for a food group."""
    pattern = re.compile(
        rf"^enose_{re.escape(food_group)}_(\d+)\.csv$", re.IGNORECASE
    )
    used_ids: list[int] = []
    if output_dir.exists():
        for path in output_dir.glob(f"enose_{food_group}_*.csv"):
            match = pattern.match(path.name)
            if match:
                used_ids.append(int(match.group(1)))
    return str(max(used_ids, default=0) + 1)


def prompt_food_group() -> str:
    print("\nSelect food group:")
    for index, group in enumerate(FOOD_GROUPS, start=1):
        print(f"  {index}. {group}")
    print(f"  {len(FOOD_GROUPS) + 1}. custom")

    while True:
        answer = input("Food group number or name: ").strip()
        if answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= len(FOOD_GROUPS):
                return FOOD_GROUPS[choice - 1]
            if choice == len(FOOD_GROUPS) + 1:
                custom = input("Custom food group: ").strip()
                try:
                    return safe_name_component(custom, "food group")
                except ValueError as exc:
                    print(f"Invalid value: {exc}")
                    continue
        else:
            try:
                return safe_name_component(answer, "food group")
            except ValueError as exc:
                print(f"Invalid value: {exc}")
                continue
        print("Invalid selection; choose a listed number or enter a name.")


def resolve_output(args: argparse.Namespace) -> tuple[Path, str, str]:
    """Resolve CSV path and dataset labels from CLI or interactive input."""
    if args.csv is not None and args.filename is not None:
        raise ValueError("use either --csv or --filename, not both")

    explicit_path: Path | None = None
    if args.csv is not None:
        explicit_path = args.csv
    elif args.filename is not None:
        filename_path = Path(args.filename)
        if filename_path.name != args.filename:
            raise ValueError("--filename must be a filename, not a path")
        filename = filename_path.name
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        if not filename.lower().startswith("enose_"):
            filename = "enose_" + filename
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
            raise ValueError(
                "--filename may contain only letters, numbers, _, -, and ."
            )
        explicit_path = args.output_dir / filename

    inferred_group = ""
    inferred_id = ""
    if explicit_path is not None:
        inferred_group, inferred_id = infer_identity_from_filename(
            explicit_path.name
        )

    if args.food_group:
        food_group = safe_name_component(args.food_group, "food group")
    elif inferred_group:
        food_group = inferred_group
    elif sys.stdin.isatty():
        food_group = prompt_food_group()
    else:
        raise ValueError(
            "food group is required in non-interactive mode; use --food-group"
        )

    if args.sample_id:
        sample_id = safe_name_component(args.sample_id, "sample ID")
    elif inferred_id:
        sample_id = inferred_id
    elif explicit_path is None:
        if sys.stdin.isatty():
            entered_id = input(
                f"Recording ID [Enter = next for {food_group}]: "
            ).strip()
            sample_id = (
                safe_name_component(entered_id, "sample ID")
                if entered_id
                else next_sample_id(args.output_dir, food_group)
            )
        else:
            sample_id = next_sample_id(args.output_dir, food_group)
    else:
        sample_id = ""

    if explicit_path is None:
        explicit_path = args.output_dir / f"enose_{food_group}_{sample_id}.csv"

    if explicit_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output file already exists: {explicit_path}; choose another ID "
            "or pass --overwrite"
        )
    return explicit_path, food_group, sample_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read six ADS7828/TGS channels over I2C and SVM41 over USB-UART."
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
    svm_group.add_argument(
        "--svm41-warmup-timeout",
        type=positive_float,
        default=120.0,
        help=(
            "maximum seconds to wait for VOC and NOx indices to become "
            "non-zero before CSV recording starts (default: 120)"
        ),
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
        help="exact CSV path (legacy option)",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="directory for generated CSV files (default: data/raw)",
    )
    output_group.add_argument(
        "--filename",
        help="exact CSV filename, e.g. enose_fresh_meat_1.csv",
    )
    output_group.add_argument(
        "--food-group",
        help=(
            "dataset label; known groups: " + ", ".join(FOOD_GROUPS)
        ),
    )
    output_group.add_argument(
        "--sample-id",
        help="recording ID; default is the next unused number",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing CSV file",
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
    "food_group",
    "sample_id",
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

    if args.no_tgs and args.no_svm41:
        print("ERROR: all sensor groups are disabled.", file=sys.stderr)
        return 2

    try:
        output_path, food_group, sample_id = resolve_output(args)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("ECE450 combined sensor acquisition")
    print(f"Frame interval: {args.interval:.3f} s")
    print(f"Food group: {food_group}")
    print(f"Sample ID: {sample_id or '(not encoded in exact filename)'}")
    print(f"CSV output: {output_path}")
    print(
        "Communication: TGS=ADS7828/I2C, SVM41=UART/SHDLC"
    )

    try:
        with ExitStack() as stack:
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
                wait_for_svm41_ready(
                    svm41_reader,
                    args.svm41_warmup_timeout,
                    1.0,
                    args.verbose,
                )

            # Do not create or write the dataset until SVM41 warm-up succeeds.
            csv_handle, csv_writer = open_csv(output_path, stack)

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

                svm41_values = None
                if svm41_reader is not None:
                    try:
                        svm41_values = svm41_reader.read()
                    except Exception as exc:
                        errors.append(f"SVM41_{type(exc).__name__.upper()}")
                        if args.verbose:
                            print(f"SVM41 read failed: {exc}", file=sys.stderr)

                if svm41_reader is not None and not svm41_gas_indices_ready(
                    svm41_values
                ):
                    errors.append("SVM41_GAS_INDEX_NOT_READY")

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

                # VOC/NOx zero is an algorithm warm-up/unavailable marker, not
                # a gas measurement.  Do not write or count this frame.
                if "SVM41_GAS_INDEX_NOT_READY" in errors:
                    next_deadline += args.interval
                    remaining_s = next_deadline - time.monotonic()
                    if remaining_s > 0:
                        time.sleep(remaining_s)
                    continue

                if csv_writer is not None:
                    row: list[Any] = [
                        timestamp,
                        f"{elapsed_s:.6f}",
                        sequence,
                        food_group,
                        sample_id,
                        f"{frame_duration_ms:.3f}",
                        f"{deadline_miss_ms:.3f}",
                    ]
                    append_tgs_csv(row, tgs_values)
                    row.append(1 if tgs_values is not None else 0)
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

    print(f"CSV saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

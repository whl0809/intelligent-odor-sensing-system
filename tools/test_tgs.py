#!/usr/bin/env python3
"""Test only the six Figaro TGS sensors connected through an ADS7828.

Hardware mapping from AGENTS.md:

    ADS7828 I2C address: 0x48
    CH0: TGS2620
    CH1: TGS2610
    CH2: TGS2611
    CH3: TGS2600
    CH4: TGS2602
    CH5: TGS2603

The script reports 12-bit ADC codes and voltages. These readings are not ppm;
gas concentration conversion requires a validated calibration model.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


ADS7828_COMMANDS = {
    0: 0x8C,
    1: 0xCC,
    2: 0x9C,
    3: 0xDC,
    4: 0xAC,
    5: 0xEC,
}

TGS_CHANNELS = (
    ("tgs2620", 0),
    ("tgs2610", 1),
    ("tgs2611", 2),
    ("tgs2600", 3),
    ("tgs2602", 4),
    ("tgs2603", 5),
)

REFERENCE_V = 2.5
FULL_SCALE_CODES = 4096
NEAR_RAIL_LOW = 4
NEAR_RAIL_HIGH = 4091


@dataclass(frozen=True)
class TgsSample:
    raw: int
    voltage_v: float
    near_rail: bool


class Ads7828:
    """Minimal ADS7828 driver for six single-ended TGS channels."""

    def __init__(self, bus_number: int, address: int) -> None:
        try:
            from smbus2 import SMBus, i2c_msg
        except ImportError as exc:
            raise RuntimeError(
                "smbus2 is required. Install it with:\n"
                "  sudo apt update\n"
                "  sudo apt install -y python3-smbus2"
            ) from exc

        self.address = address
        self._i2c_msg = i2c_msg
        self._bus = SMBus(bus_number)

        # Enable the internal reference, wait for it to settle, then discard
        # the first conversion as required during ADS7828 startup.
        self._send_command(ADS7828_COMMANDS[0])
        time.sleep(0.002)
        self.read_channel(0)

    def close(self) -> None:
        self._bus.close()

    def __enter__(self) -> "Ads7828":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

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

    def read_channel(self, channel: int) -> TgsSample:
        if channel not in ADS7828_COMMANDS:
            raise ValueError(f"invalid TGS ADS7828 channel: {channel}")

        byte0, byte1 = self._transfer(ADS7828_COMMANDS[channel])
        raw = self.parse_raw(byte0, byte1)
        return TgsSample(
            raw=raw,
            voltage_v=raw * REFERENCE_V / FULL_SCALE_CODES,
            near_rail=raw <= NEAR_RAIL_LOW or raw >= NEAR_RAIL_HIGH,
        )


def parse_i2c_address(value: str) -> int:
    try:
        address = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid I2C address {value!r}; use a value such as 0x48"
        ) from exc
    if not 0x03 <= address <= 0x77:
        raise argparse.ArgumentTypeError("I2C address must be from 0x03 to 0x77")
    return address


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from exc
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
        description="Test only the six TGS sensors connected to the ADS7828."
    )
    parser.add_argument(
        "--bus", type=int, default=1, help="I2C bus number (default: 1)"
    )
    parser.add_argument(
        "--address",
        type=parse_i2c_address,
        default=0x48,
        help="ADS7828 7-bit I2C address (default: 0x48)",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=1.0,
        help="seconds between sampling deadlines (default: 1.0)",
    )
    parser.add_argument(
        "--samples",
        type=nonnegative_int,
        default=0,
        help="number of samples; 0 means run until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional CSV output path, e.g. ../data/raw/tgs_test.csv",
    )
    return parser


def csv_header() -> list[str]:
    header = [
        "timestamp_utc",
        "elapsed_s",
        "sequence",
        "frame_duration_ms",
        "deadline_miss_ms",
    ]
    for sensor_name, _ in TGS_CHANNELS:
        header.extend(
            (
                f"{sensor_name}_raw",
                f"{sensor_name}_voltage_v",
                f"{sensor_name}_near_rail",
            )
        )
    header.extend(("ads7828_ok", "error_codes"))
    return header


def open_csv(path: Path | None) -> tuple[TextIO | None, csv.writer | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(csv_header())
    handle.flush()
    return handle, writer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def format_terminal_row(
    sequence: int,
    samples: dict[str, TgsSample | None],
    error_codes: list[str],
) -> str:
    values = []
    for sensor_name, _ in TGS_CHANNELS:
        sample = samples[sensor_name]
        if sample is None:
            values.append(f"{sensor_name.upper()}=ERROR")
        else:
            rail = " NEAR_RAIL" if sample.near_rail else ""
            values.append(
                f"{sensor_name.upper()}={sample.raw:4d} "
                f"({sample.voltage_v:.4f} V){rail}"
            )
    status = "OK" if not error_codes else ";".join(error_codes)
    return f"[{sequence:06d}] " + " | ".join(values) + f" | status={status}"


def run(args: argparse.Namespace) -> int:
    print("ECE450 TGS-only sensor test")
    print(f"I2C: /dev/i2c-{args.bus}, ADS7828 address: 0x{args.address:02X}")
    print("Order: TGS2620, TGS2610, TGS2611, TGS2600, TGS2602, TGS2603")
    print("Output: ADC raw code and voltage only; values are not ppm.")

    csv_handle, writer = open_csv(args.csv)
    sequence = 0

    try:
        with Ads7828(args.bus, args.address) as adc:
            start_monotonic = time.monotonic()
            next_deadline = start_monotonic
            while args.samples == 0 or sequence < args.samples:
                if sequence > 0:
                    sleep_s = next_deadline - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                frame_start = time.monotonic()
                timestamp = utc_now()
                samples: dict[str, TgsSample | None] = {}
                error_codes: list[str] = []

                for sensor_name, channel in TGS_CHANNELS:
                    try:
                        sample = adc.read_channel(channel)
                        samples[sensor_name] = sample
                        if sample.near_rail:
                            error_codes.append(f"{sensor_name.upper()}_NEAR_RAIL")
                    except OSError:
                        samples[sensor_name] = None
                        error_codes.append(f"{sensor_name.upper()}_I2C_ERROR")

                frame_end = time.monotonic()
                elapsed_s = frame_start - start_monotonic
                frame_duration_ms = (frame_end - frame_start) * 1000.0
                deadline_miss_ms = max(0.0, (frame_start - next_deadline) * 1000.0)

                sequence += 1
                print(format_terminal_row(sequence, samples, error_codes))

                if writer is not None:
                    row: list[object] = [
                        timestamp,
                        f"{elapsed_s:.6f}",
                        sequence,
                        f"{frame_duration_ms:.3f}",
                        f"{deadline_miss_ms:.3f}",
                    ]
                    for sensor_name, _ in TGS_CHANNELS:
                        sample = samples[sensor_name]
                        if sample is None:
                            row.extend(("", "", ""))
                        else:
                            row.extend(
                                (
                                    sample.raw,
                                    f"{sample.voltage_v:.8f}",
                                    int(sample.near_rail),
                                )
                            )
                    all_channels_ok = not any(
                        samples[name] is None for name, _ in TGS_CHANNELS
                    )
                    row.extend((int(all_channels_ok), ";".join(error_codes)))
                    writer.writerow(row)
                    if sequence % 10 == 0 and csv_handle is not None:
                        csv_handle.flush()

                next_deadline = start_monotonic + sequence * args.interval

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (FileNotFoundError, PermissionError, OSError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Check that I2C is enabled, /dev/i2c-1 exists, the TGS board is "
            "powered, SDA/SCL/GND are connected correctly, and address 0x48 "
            "appears in: sudo i2cdetect -y 1",
            file=sys.stderr,
        )
        return 1
    finally:
        if csv_handle is not None:
            csv_handle.flush()
            csv_handle.close()

    if args.csv is not None:
        print(f"CSV saved to: {args.csv}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

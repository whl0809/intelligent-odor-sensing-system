#!/usr/bin/env python3
"""Standalone EC Sense NH3/H2S test for the ECE450 acquisition repository.

The EC Sense boards are read through MCP3421 ADCs:
  NH3: 0x6A
  H2S: 0x69

This script can be run directly from the repository without installing
the enose package because it adds the repository's src directory to sys.path.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

# Allow direct execution from ECE450_software/tools without pip install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enose.i2c_bus import I2CBus
from enose.mcp3421 import MCP3421


def parse_i2c_address(value: str) -> int:
    """Parse decimal or 0x-prefixed I2C address."""
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
        description="Read EC Sense NH3 and H2S MCP3421 ADC outputs."
    )
    parser.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    parser.add_argument(
        "--nh3-address",
        type=parse_i2c_address,
        default=0x6A,
        help="NH3 MCP3421 address (default: 0x6A)",
    )
    parser.add_argument(
        "--h2s-address",
        type=parse_i2c_address,
        default=0x69,
        help="H2S MCP3421 address (default: 0x69)",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=1.0,
        help="seconds between output rows (default: 1.0)",
    )
    parser.add_argument(
        "--samples",
        type=nonnegative_int,
        default=0,
        help="number of rows to read; 0 means run until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional CSV output path, for example data/raw/ec_sense_test.csv",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="skip SMBus quick-write address probing and attempt direct reads",
    )
    return parser


def open_csv(path: Path | None, stack: ExitStack) -> tuple[TextIO | None, csv.writer | None]:
    if path is None:
        return None, None

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
    writer = csv.writer(handle)
    writer.writerow(
        [
            "timestamp",
            "monotonic_s",
            "nh3_raw",
            "nh3_voltage_v",
            "nh3_voltage_mv",
            "h2s_raw",
            "h2s_voltage_v",
            "h2s_voltage_mv",
        ]
    )
    handle.flush()
    return handle, writer


def main() -> int:
    args = build_parser().parse_args()

    if args.nh3_address == args.h2s_address:
        print("ERROR: NH3 and H2S addresses must be different.", file=sys.stderr)
        return 2

    sensor_specs = (
        ("NH3", args.nh3_address),
        ("H2S", args.h2s_address),
    )

    print("EC Sense NH3/H2S test")
    print(f"I2C device: /dev/i2c-{args.bus}")
    print(f"NH3 address: 0x{args.nh3_address:02X}")
    print(f"H2S address: 0x{args.h2s_address:02X}")
    print("ADC mode: MCP3421, 18-bit, gain x1, continuous")
    print("The values below are ADC differential voltages, not ppm.\n")

    try:
        with ExitStack() as stack:
            bus = stack.enter_context(I2CBus(args.bus))
            csv_handle, csv_writer = open_csv(args.csv, stack)

            if not args.skip_probe:
                print("Address probe:")
                for name, address in sensor_specs:
                    acknowledged = bus.probe(address)
                    result = "ACK" if acknowledged else "NACK"
                    print(f"  {name}: 0x{address:02X} -> {result}")
                print(
                    "Note: a NACK from quick probing is not treated as fatal; "
                    "the script will still attempt a direct MCP3421 transfer.\n"
                )

            sensors: dict[str, MCP3421] = {}
            for name, address in sensor_specs:
                sensor = MCP3421(
                    bus=bus,
                    address=address,
                    resolution_bits=18,
                    gain=1,
                    continuous=True,
                )
                sensor.configure()
                sensors[name] = sensor
                print(
                    f"Configured {name} at 0x{address:02X} "
                    f"with config byte 0x{sensor.config_byte:02X}"
                )

            # One 18-bit conversion takes about 1 / 3.75 s.
            time.sleep(0.35)

            print("\nPress Ctrl+C to stop.\n")
            print(
                "timestamp                  "
                "NH3 raw       NH3 mV       H2S raw       H2S mV"
            )
            print("-" * 78)

            sample_index = 0
            while args.samples == 0 or sample_index < args.samples:
                cycle_start = time.monotonic()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                nh3 = sensors["NH3"].read()
                h2s = sensors["H2S"].read()

                nh3_mv = nh3.differential_voltage_v * 1000.0
                h2s_mv = h2s.differential_voltage_v * 1000.0

                print(
                    f"{timestamp}  "
                    f"{nh3.raw:+10d}  {nh3_mv:+11.4f}  "
                    f"{h2s.raw:+10d}  {h2s_mv:+11.4f}",
                    flush=True,
                )

                if csv_writer is not None and csv_handle is not None:
                    csv_writer.writerow(
                        [
                            timestamp,
                            f"{cycle_start:.6f}",
                            nh3.raw,
                            f"{nh3.differential_voltage_v:.9f}",
                            f"{nh3_mv:.6f}",
                            h2s.raw,
                            f"{h2s.differential_voltage_v:.9f}",
                            f"{h2s_mv:.6f}",
                        ]
                    )
                    csv_handle.flush()

                sample_index += 1
                elapsed = time.monotonic() - cycle_start
                time.sleep(max(0.0, args.interval - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except PermissionError:
        print(
            f"ERROR: permission denied opening /dev/i2c-{args.bus}.\n"
            "Try running with sudo, or add your user to the i2c group.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError:
        print(
            f"ERROR: /dev/i2c-{args.bus} does not exist. Enable I2C first.",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(
            f"ERROR: I2C communication failed: {exc}\n"
            "Check sensor power, common ground, SDA/SCL wiring, and addresses.",
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

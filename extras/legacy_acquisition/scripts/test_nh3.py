#!/usr/bin/env python3
"""Test only the EC Sense NH3 channel through its MCP3421 ADC.

Hardware configuration from AGENTS.md:

* I2C bus: /dev/i2c-1
* NH3 MCP3421 address: 0x69
* VIN+ = TIA_VOUT_1 and VIN- = VBIAS

The displayed value is the signed ADC code and differential voltage. It is
not an NH3 concentration in ppm because no validated calibration transfer
function is applied.

Place this file in ``ECE450_software/tools``. It adds the repository's ``src``
directory to ``sys.path``, so the local ``enose`` package need not be installed.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


# Allow direct execution from ECE450_software/tools without pip installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enose.i2c_bus import I2CBus
from enose.mcp3421 import MCP3421


CSV_HEADER = [
    "timestamp_utc",
    "elapsed_s",
    "sequence",
    "frame_duration_ms",
    "deadline_miss_ms",
    "nh3_raw",
    "nh3_diff_voltage_v",
    "nh3_ok",
    "error_code",
]


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
        description="Read only the EC Sense NH3 MCP3421 sensor."
    )
    parser.add_argument(
        "--bus", type=int, default=1, help="I2C bus number (default: 1)"
    )
    parser.add_argument(
        "--address",
        type=parse_i2c_address,
        default=0x69,
        help="NH3 MCP3421 I2C address (default: 0x69)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=(12, 14, 16, 18),
        default=18,
        help="ADC resolution in bits (default: 18)",
    )
    parser.add_argument(
        "--gain",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help="MCP3421 programmable gain (default: 1)",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="use one-shot conversion mode instead of continuous mode",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="skip the informational address probe and read directly",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=1.0,
        help="seconds between sample deadlines (default: 1.0)",
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
        help="optional CSV output path, e.g. ../data/raw/nh3_test.csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show configuration and read-error details",
    )
    return parser


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
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


def main() -> int:
    args = build_parser().parse_args()

    print("ECE450 EC Sense NH3 test")
    print(f"I2C device: /dev/i2c-{args.bus}")
    print(f"NH3 address: 0x{args.address:02X}")
    print(
        f"ADC mode: MCP3421, {args.resolution}-bit, gain x{args.gain}, "
        f"{'one-shot' if args.one_shot else 'continuous'}"
    )
    print("Output is differential voltage, not ppm.")

    try:
        with ExitStack() as stack:
            bus = stack.enter_context(I2CBus(args.bus))

            if not args.skip_probe:
                probe_result = "ACK" if bus.probe(args.address) else "NACK"
                print(f"Address probe: 0x{args.address:02X} -> {probe_result}")
                if probe_result == "NACK":
                    print(
                        "Note: probe NACK is informational; a direct MCP3421 "
                        "transfer will still be attempted."
                    )

            sensor = MCP3421(
                bus=bus,
                address=args.address,
                resolution_bits=args.resolution,
                gain=args.gain,
                continuous=not args.one_shot,
            )
            sensor.configure()
            if args.verbose:
                print(f"MCP3421 config byte: 0x{sensor.config_byte:02X}")

            # An 18-bit conversion takes about 267 ms; wait for the first
            # continuous-mode result rather than reporting a stale value.
            if args.resolution == 18 and not args.one_shot:
                time.sleep(0.35)

            csv_handle, csv_writer = open_csv(args.csv, stack)

            print("\nPress Ctrl+C to stop.\n")
            start_monotonic = time.monotonic()
            next_deadline = start_monotonic
            sequence = 0

            while args.samples == 0 or sequence < args.samples:
                frame_start = time.monotonic()
                timestamp = utc_timestamp()
                deadline_miss_ms = max(
                    0.0, (frame_start - next_deadline) * 1000.0
                )

                sample = None
                error_code = ""
                try:
                    sample = sensor.read()
                except Exception as exc:
                    error_code = f"NH3_{type(exc).__name__.upper()}"
                    if args.verbose:
                        print(f"NH3 read failed: {exc}", file=sys.stderr)

                frame_end = time.monotonic()
                elapsed_s = frame_start - start_monotonic
                frame_duration_ms = (frame_end - frame_start) * 1000.0

                if sample is None:
                    print(
                        f"{timestamp}  seq={sequence}  NH3=READ_ERROR  "
                        f"ERR={error_code}",
                        flush=True,
                    )
                else:
                    print(
                        f"{timestamp}  seq={sequence}  "
                        f"NH3 raw={sample.raw:+d}  "
                        f"voltage={sample.differential_voltage_v * 1000:+.3f} mV  "
                        f"frame={frame_duration_ms:.1f} ms",
                        flush=True,
                    )

                if csv_writer is not None:
                    if sample is None:
                        raw_value = ""
                        voltage_value = ""
                        ok = 0
                    else:
                        raw_value = sample.raw
                        voltage_value = f"{sample.differential_voltage_v:.9f}"
                        ok = 1

                    csv_writer.writerow(
                        [
                            timestamp,
                            f"{elapsed_s:.6f}",
                            sequence,
                            f"{frame_duration_ms:.3f}",
                            f"{deadline_miss_ms:.3f}",
                            raw_value,
                            voltage_value,
                            ok,
                            error_code,
                        ]
                    )
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
            "Add your user to the i2c group, then log out and in.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"ERROR: I2C communication failed: {exc}\n"
            "Check sensor-board power, common ground, SDA/SCL wiring, and "
            f"address 0x{args.address:02X}.",
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

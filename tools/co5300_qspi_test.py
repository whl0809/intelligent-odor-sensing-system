#!/usr/bin/env python3
"""CO5300 410x502 AMOLED bring-up tool for Raspberry Pi 5.

This program uses software-timed 4-data-line QSPI through the Linux lgpio
wave API.  It intentionally does not use /dev/spidev*, so it can coexist
with the ECE450 ADS114S06 wiring which owns the physical SPI0 lines.

Default independent BCM GPIO wiring:
    CLK  = 21
    SIO0 = 20
    SIO1 = 19
    SIO2 = 16
    SIO3 = 26
    CS   = 18
    RST  = 25
    TE   = 24

Repository location:
    ECE450_software/tools/co5300_qspi_test.py

Examples:
    sudo python3 tools/co5300_qspi_test.py --self-test
    sudo python3 tools/co5300_qspi_test.py \
        --init config/co5300_init.json --pattern bars --hold
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


WIDTH = 410
HEIGHT = 502
X_OFFSET = 22
Y_OFFSET = 0

OPCODE_COMMAND_WRITE = 0x02
OPCODE_QUAD_PIXEL_WRITE = 0x32

CMD_SLEEP_IN = 0x10
CMD_NORMAL_DISPLAY_ON = 0x13
CMD_DISPLAY_OFF = 0x28
CMD_DISPLAY_ON = 0x29
CMD_COLUMN_ADDRESS_SET = 0x2A
CMD_ROW_ADDRESS_SET = 0x2B
CMD_MEMORY_WRITE = 0x2C
CMD_MEMORY_WRITE_CONTINUE = 0x3C
CMD_BRIGHTNESS = 0x51

Pulse = namedtuple("Pulse", ("group_bits", "group_mask", "pulse_delay"))


class CO5300Error(RuntimeError):
    """Raised when a GPIO/QSPI operation fails."""


@dataclass(frozen=True)
class PinMap:
    clk: int = 21
    sio0: int = 20
    sio1: int = 19
    sio2: int = 16
    sio3: int = 26
    cs: int = 18
    rst: int = 25
    te: int = 24

    def outputs(self) -> list[int]:
        # The first line is the group leader used by lgpio.tx_wave().
        return [self.clk, self.sio0, self.sio1, self.sio2,
                self.sio3, self.cs, self.rst]

    def validate(self) -> None:
        values = self.outputs() + [self.te]
        if any(pin < 0 for pin in values):
            raise ValueError("GPIO numbers must be non-negative")
        if len(set(values)) != len(values):
            raise ValueError("Every CLK/SIO/CS/RST/TE GPIO must be unique")


@dataclass(frozen=True)
class InitCommand:
    command: int
    data: bytes
    delay_ms: int = 0


def parse_int(value: Any, name: str) -> int:
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[0-9A-Fa-f]{2}", text):
            result = int(text, 16)
        else:
            result = int(text, 0)
    else:
        raise ValueError(f"{name} must be an integer or numeric string")
    return result


def load_init_commands(path: Path) -> list[InitCommand]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CO5300Error(f"Cannot read initialization file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CO5300Error(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise CO5300Error("Initialization JSON must be a non-empty list")

    result: list[InitCommand] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CO5300Error(f"Initialization entry {index} is not an object")
        try:
            command = parse_int(item["cmd"], f"entry {index} cmd")
            data_values = item.get("data", [])
            delay_ms = int(item.get("delay_ms", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise CO5300Error(f"Invalid initialization entry {index}: {exc}") from exc

        if not 0 <= command <= 0xFF:
            raise CO5300Error(f"Entry {index}: command is outside 0..255")
        if not isinstance(data_values, list):
            raise CO5300Error(f"Entry {index}: data must be a list")
        data = bytearray()
        for data_index, value in enumerate(data_values):
            parsed = parse_int(value, f"entry {index} data[{data_index}]")
            if not 0 <= parsed <= 0xFF:
                raise CO5300Error(
                    f"Entry {index} data[{data_index}] is outside 0..255"
                )
            data.append(parsed)
        if delay_ms < 0:
            raise CO5300Error(f"Entry {index}: delay_ms must be non-negative")
        result.append(InitCommand(command, bytes(data), delay_ms))
    return result


def detect_gpiochip_number() -> int:
    """Return the /dev/gpiochipN index for the RP1 GPIO controller.

    Important: names under /sys/class/gpio such as gpiochip569 contain the
    global GPIO base number, not necessarily the character-device index.
    Therefore this function first parses `gpiodetect`, whose gpiochipN names
    match /dev/gpiochipN.
    """
    try:
        completed = subprocess.run(
            ["gpiodetect"],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in completed.stdout.splitlines():
            lower = line.lower()
            if "rp1" not in lower:
                continue
            match = re.match(r"\s*gpiochip(\d+)\b", line)
            if match:
                candidate = int(match.group(1))
                if Path(f"/dev/gpiochip{candidate}").exists():
                    return candidate
    except (FileNotFoundError, OSError):
        pass

    # Prefer common Raspberry Pi 5 device indices when present.
    for candidate in (0, 4):
        if Path(f"/dev/gpiochip{candidate}").exists():
            return candidate

    # Last resort: use the first actual character device, never a sysfs base.
    devices = []
    for path in Path("/dev").glob("gpiochip*"):
        match = re.fullmatch(r"gpiochip(\d+)", path.name)
        if match:
            devices.append(int(match.group(1)))
    if devices:
        return min(devices)

    raise CO5300Error("No suitable /dev/gpiochipN device was found")


def rgb565_bytes(red: int, green: int, blue: int, bgr: bool = False) -> bytes:
    for component in (red, green, blue):
        if not 0 <= component <= 255:
            raise ValueError("RGB components must be in the range 0..255")
    if bgr:
        red, blue = blue, red
    value = ((red & 0xF8) << 8) | ((green & 0xFC) << 3) | (blue >> 3)
    return bytes(((value >> 8) & 0xFF, value & 0xFF))


def solid_frame(red: int, green: int, blue: int, bgr: bool = False) -> bytes:
    return rgb565_bytes(red, green, blue, bgr) * (WIDTH * HEIGHT)


def color_bars_frame(bgr: bool = False) -> bytes:
    colors = (
        (255, 255, 255),
        (255, 255, 0),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 0),
    )
    row = bytearray()
    for x in range(WIDTH):
        bar = min((x * len(colors)) // WIDTH, len(colors) - 1)
        row.extend(rgb565_bytes(*colors[bar], bgr=bgr))
    return bytes(row) * HEIGHT


def checker_frame(bgr: bool = False, block: int = 25) -> bytes:
    white = rgb565_bytes(255, 255, 255, bgr)
    black = rgb565_bytes(0, 0, 0, bgr)
    frame = bytearray(WIDTH * HEIGHT * 2)
    offset = 0
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixel = white if ((x // block) + (y // block)) % 2 == 0 else black
            frame[offset:offset + 2] = pixel
            offset += 2
    return bytes(frame)


def pattern_frame(name: str, bgr: bool = False) -> bytes:
    if name == "bars":
        return color_bars_frame(bgr)
    if name == "checker":
        return checker_frame(bgr)
    colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }
    if name not in colors:
        raise ValueError(f"Unsupported pattern: {name}")
    return solid_frame(*colors[name], bgr=bgr)


class LgpioQuadBus:
    """Software-timed CO5300 single-command/quad-pixel bus."""

    CLK = 1 << 0
    D0 = 1 << 1
    D1 = 1 << 2
    D2 = 1 << 3
    D3 = 1 << 4
    CS = 1 << 5
    RST = 1 << 6
    ALL = (1 << 7) - 1

    def __init__(
        self,
        lgpio_module: Any,
        gpiochip: int,
        pins: PinMap,
        half_period_us: int,
    ) -> None:
        if half_period_us < 1:
            raise ValueError("half_period_us must be at least 1")
        pins.validate()
        self.lgpio = lgpio_module
        self.gpiochip = gpiochip
        self.pins = pins
        self.half_period_us = half_period_us
        self.handle: int | None = None
        self.group_leader = pins.clk
        self.te_claimed = False

    @property
    def idle_bits(self) -> int:
        return self.CS | self.RST

    @property
    def active_base(self) -> int:
        # CS low, RESET high, clock low.
        return self.RST

    def open(self) -> None:
        try:
            self.handle = self.lgpio.gpiochip_open(self.gpiochip)
            levels = [0, 0, 0, 0, 0, 1, 1]
            self.lgpio.group_claim_output(
                self.handle, self.pins.outputs(), levels
            )
            self.lgpio.gpio_claim_input(self.handle, self.pins.te)
            self.te_claimed = True
            self.lgpio.group_write(
                self.handle, self.group_leader, self.idle_bits, self.ALL
            )
        except Exception as exc:
            self.close()
            raise CO5300Error(
                f"Cannot claim GPIO lines on /dev/gpiochip{self.gpiochip}: {exc}"
            ) from exc

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            self.lgpio.group_write(
                self.handle, self.group_leader, self.idle_bits, self.ALL
            )
        except Exception:
            pass
        if self.te_claimed:
            try:
                self.lgpio.gpio_free(self.handle, self.pins.te)
            except Exception:
                pass
            self.te_claimed = False
        try:
            self.lgpio.group_free(self.handle, self.group_leader)
        except Exception:
            pass
        try:
            self.lgpio.gpiochip_close(self.handle)
        except Exception:
            pass
        self.handle = None

    def _write_group(self, bits: int) -> None:
        if self.handle is None:
            raise CO5300Error("GPIO bus is not open")
        self.lgpio.group_write(self.handle, self.group_leader, bits, self.ALL)

    def hardware_reset(self) -> None:
        self._write_group(self.idle_bits)
        time.sleep(0.010)
        self._write_group(self.CS)  # CS high, RESET low.
        time.sleep(0.020)
        self._write_group(self.idle_bits)
        time.sleep(0.150)

    def _send_wave(self, pulses: Sequence[Pulse]) -> None:
        if self.handle is None:
            raise CO5300Error("GPIO bus is not open")
        if not pulses:
            return
        try:
            result = self.lgpio.tx_wave(self.handle, self.group_leader, pulses)
            if isinstance(result, int) and result < 0:
                raise CO5300Error(f"lgpio.tx_wave returned {result}")
            tx_wave_kind = getattr(self.lgpio, "TX_WAVE", 1)
            while self.lgpio.tx_busy(
                self.handle, self.group_leader, tx_wave_kind
            ):
                time.sleep(0.0005)
        except Exception as exc:
            raise CO5300Error(f"GPIO waveform transmission failed: {exc}") from exc

    def _append_single_byte(self, pulses: list[Pulse], value: int) -> None:
        delay = self.half_period_us
        for shift in range(7, -1, -1):
            data = self.D0 if ((value >> shift) & 1) else 0
            low = self.active_base | data
            high = low | self.CLK
            pulses.append(Pulse(low, self.ALL, delay))
            pulses.append(Pulse(high, self.ALL, delay))

    def _append_quad_byte(self, pulses: list[Pulse], value: int) -> None:
        delay = self.half_period_us
        for nibble in ((value >> 4) & 0x0F, value & 0x0F):
            data = 0
            if nibble & 0x1:
                data |= self.D0
            if nibble & 0x2:
                data |= self.D1
            if nibble & 0x4:
                data |= self.D2
            if nibble & 0x8:
                data |= self.D3
            low = self.active_base | data
            high = low | self.CLK
            pulses.append(Pulse(low, self.ALL, delay))
            pulses.append(Pulse(high, self.ALL, delay))

    def single_transaction(self, payload: bytes) -> None:
        pulses: list[Pulse] = [
            Pulse(self.active_base, self.ALL, self.half_period_us)
        ]
        for value in payload:
            self._append_single_byte(pulses, value)
        pulses.append(Pulse(self.idle_bits, self.ALL, self.half_period_us))
        self._send_wave(pulses)

    def command(self, command: int, parameters: bytes = b"") -> None:
        if not 0 <= command <= 0xFF:
            raise ValueError("command must fit in one byte")
        packet = bytes((OPCODE_COMMAND_WRITE, 0x00, command, 0x00)) + parameters
        self.single_transaction(packet)

    def quad_pixel_transaction(self, ram_command: int, payload: bytes) -> None:
        if ram_command not in (CMD_MEMORY_WRITE, CMD_MEMORY_WRITE_CONTINUE):
            raise ValueError("ram_command must be 0x2C or 0x3C")
        pulses: list[Pulse] = [
            Pulse(self.active_base, self.ALL, self.half_period_us)
        ]
        # 0x32 instruction and 24-bit address are transmitted over SIO0.
        prefix = bytes((OPCODE_QUAD_PIXEL_WRITE, 0x00, ram_command, 0x00))
        for value in prefix:
            self._append_single_byte(pulses, value)
        # Pixel payload is transmitted two nibbles per QSPI clock using SIO0..3.
        for value in payload:
            self._append_quad_byte(pulses, value)
        pulses.append(Pulse(self.idle_bits, self.ALL, self.half_period_us))
        self._send_wave(pulses)

    def te_edges(self, seconds: float = 1.0) -> int:
        if self.handle is None or not self.te_claimed:
            return 0
        end = time.monotonic() + seconds
        last = int(self.lgpio.gpio_read(self.handle, self.pins.te))
        rising = 0
        while time.monotonic() < end:
            current = int(self.lgpio.gpio_read(self.handle, self.pins.te))
            if last == 0 and current == 1:
                rising += 1
            last = current
            time.sleep(0.0008)
        return rising


class CO5300Display:
    def __init__(
        self,
        bus: LgpioQuadBus,
        init_commands: Sequence[InitCommand],
        chunk_bytes: int,
    ) -> None:
        if chunk_bytes < 16 or chunk_bytes > 8192:
            raise ValueError("chunk_bytes must be between 16 and 8192")
        self.bus = bus
        self.init_commands = list(init_commands)
        self.chunk_bytes = chunk_bytes

    def initialize(self) -> None:
        print("[1/4] Hardware reset")
        self.bus.hardware_reset()

        # Eight 1 bits recover the controller from dual/quad command mode to
        # single-line command mode before 0x02 command transactions.
        self.bus.single_transaction(bytes((0xFF,)))

        print("[2/4] Sending initialization table")
        for entry in self.init_commands:
            suffix = " ".join(f"{value:02X}" for value in entry.data)
            print(f"  CMD {entry.command:02X}" + (f" {suffix}" if suffix else ""))
            self.bus.command(entry.command, entry.data)
            if entry.delay_ms:
                time.sleep(entry.delay_ms / 1000.0)

    def set_window(self, x: int, y: int, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if x < 0 or y < 0 or x + width > WIDTH or y + height > HEIGHT:
            raise ValueError("display window is outside the 410x502 image")
        x0 = x + X_OFFSET
        x1 = x0 + width - 1
        y0 = y + Y_OFFSET
        y1 = y0 + height - 1
        self.bus.command(
            CMD_COLUMN_ADDRESS_SET,
            bytes((x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF)),
        )
        self.bus.command(
            CMD_ROW_ADDRESS_SET,
            bytes((y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF)),
        )

    def write_frame(self, frame: bytes) -> None:
        expected = WIDTH * HEIGHT * 2
        if len(frame) != expected:
            raise ValueError(f"frame has {len(frame)} bytes; expected {expected}")
        self.set_window(0, 0, WIDTH, HEIGHT)
        print(
            f"[4/4] Sending {len(frame):,} RGB565 bytes through 4-line QSPI"
        )
        start = time.monotonic()
        offset = 0
        command = CMD_MEMORY_WRITE
        next_progress = 10
        while offset < len(frame):
            chunk = frame[offset:offset + self.chunk_bytes]
            self.bus.quad_pixel_transaction(command, chunk)
            command = CMD_MEMORY_WRITE_CONTINUE
            offset += len(chunk)
            percent = int(offset * 100 / len(frame))
            if percent >= next_progress:
                print(f"  {percent}%")
                next_progress += 10
        elapsed = time.monotonic() - start
        print(f"  Frame transfer completed in {elapsed:.2f} s")

    def shutdown(self) -> None:
        try:
            self.bus.command(CMD_BRIGHTNESS, bytes((0x00,)))
            time.sleep(0.010)
            self.bus.command(CMD_DISPLAY_OFF)
            time.sleep(0.020)
            self.bus.command(CMD_SLEEP_IN)
            time.sleep(0.120)
        except Exception as exc:
            print(f"Warning: display shutdown failed: {exc}", file=sys.stderr)


def run_self_test() -> None:
    assert rgb565_bytes(255, 0, 0) == b"\xF8\x00"
    assert rgb565_bytes(0, 255, 0) == b"\x07\xE0"
    assert rgb565_bytes(0, 0, 255) == b"\x00\x1F"
    assert len(color_bars_frame()) == WIDTH * HEIGHT * 2
    sample = InitCommand(0x3A, bytes((0x55,)), 0)
    assert sample.command == 0x3A and sample.data == b"\x55"
    print("[PASS] CO5300 offline self-test")
    print("       RGB565 generation and frame dimensions are valid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive a 410x502 CO5300 AMOLED using software 4-line QSPI "
            "through Raspberry Pi GPIO."
        )
    )
    parser.add_argument("--init", default="config/co5300_init.json")
    parser.add_argument("--gpiochip", default="auto",
                        help="gpiochip number, or auto (default)")
    parser.add_argument("--clk", type=int, default=21)
    parser.add_argument("--sio0", type=int, default=20)
    parser.add_argument("--sio1", type=int, default=19)
    parser.add_argument("--sio2", type=int, default=16)
    parser.add_argument("--sio3", type=int, default=26)
    parser.add_argument("--cs", type=int, default=18)
    parser.add_argument("--rst", type=int, default=25)
    parser.add_argument("--te", type=int, default=24)
    parser.add_argument("--half-period-us", type=int, default=2,
                        help="QSPI half clock period in microseconds (default 2)")
    parser.add_argument("--chunk-bytes", type=int, default=2048,
                        help="pixel bytes per QSPI waveform (default 2048)")
    parser.add_argument(
        "--pattern",
        choices=("bars", "checker", "black", "white", "red", "green", "blue"),
        default="bars",
    )
    parser.add_argument("--bgr", action="store_true",
                        help="swap red and blue in generated patterns")
    parser.add_argument("--skip-te-check", action="store_true")
    parser.add_argument("--hold", action="store_true",
                        help="keep process alive after drawing until Ctrl+C")
    parser.add_argument("--off-at-end", action="store_true",
                        help="turn the panel off before exiting")
    parser.add_argument("--self-test", action="store_true",
                        help="run an offline test without touching GPIO")
    return parser


def import_lgpio() -> Any:
    try:
        import lgpio  # type: ignore
        return lgpio
    except ImportError as exc:
        raise CO5300Error(
            "Python module 'lgpio' is missing. Install it with: "
            "sudo apt install python3-lgpio"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    pins = PinMap(
        clk=args.clk,
        sio0=args.sio0,
        sio1=args.sio1,
        sio2=args.sio2,
        sio3=args.sio3,
        cs=args.cs,
        rst=args.rst,
        te=args.te,
    )
    pins.validate()

    gpiochip = detect_gpiochip_number() if args.gpiochip == "auto" else int(args.gpiochip)
    init_path = Path(args.init)
    init_commands = load_init_commands(init_path)
    lgpio = import_lgpio()

    print("CO5300 Raspberry Pi GPIO-QSPI test")
    print(f"  gpiochip : /dev/gpiochip{gpiochip}")
    print(f"  CLK      : BCM {pins.clk}")
    print(f"  SIO0..3  : BCM {pins.sio0}, {pins.sio1}, {pins.sio2}, {pins.sio3}")
    print(f"  CS/RST/TE: BCM {pins.cs}, {pins.rst}, {pins.te}")
    print(f"  init     : {init_path}")
    print(f"  pattern  : {args.pattern}")

    bus = LgpioQuadBus(lgpio, gpiochip, pins, args.half_period_us)
    display = CO5300Display(bus, init_commands, args.chunk_bytes)
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        bus.open()
        display.initialize()

        if not args.skip_te_check:
            print("[3/4] Checking AMOLED_TE for 1 second")
            edges = bus.te_edges(1.0)
            print(f"  detected TE rising edges: {edges}")
            if edges < 10:
                print(
                    "  WARNING: fewer than 10 TE edges were detected; "
                    "check initialization, TE wiring, and power.",
                    file=sys.stderr,
                )

        frame = pattern_frame(args.pattern, args.bgr)
        display.write_frame(frame)
        print("[PASS] QSPI frame sent. Observe the physical display.")

        if args.hold:
            print("Holding the image. Press Ctrl+C to stop.")
            while not interrupted:
                time.sleep(0.2)

        if args.off_at_end:
            display.shutdown()
        return 0
    except (CO5300Error, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())

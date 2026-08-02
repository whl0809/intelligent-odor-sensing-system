#!/usr/bin/env python3
"""CO5300 dashboard for the ECE450 odor-sensing system.

Place this file at:
    ECE450_software/tools/co5300_dashboard.py

It reuses the verified GPIO-QSPI transport in:
    ECE450_software/tools/co5300_qspi_test.py

Examples:
    # Show example values once and keep the image on-screen
    sudo python3 tools/co5300_dashboard.py \
        --state-file config/display_state.example.json \
        --once --hold

    # Watch a JSON state file and refresh whenever it changes
    sudo python3 tools/co5300_dashboard.py \
        --state-file runtime/display_state.json \
        --refresh-seconds 2

    # Generate changing demo values
    sudo python3 tools/co5300_dashboard.py --demo --refresh-seconds 3
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit(
        "Pillow is missing. Install it with: sudo apt install python3-pil"
    ) from exc

# This file lives in the same tools/ directory as co5300_qspi_test.py.
from co5300_qspi_test import (  # type: ignore
    CMD_MEMORY_WRITE,
    CMD_MEMORY_WRITE_CONTINUE,
    CO5300Display,
    CO5300Error,
    LgpioQuadBus,
    PinMap,
    WIDTH,
    HEIGHT,
    detect_gpiochip_number,
    import_lgpio,
    load_init_commands,
)


@dataclass(frozen=True)
class DisplayState:
    food_type: str = "Unknown"
    freshness_level: str = "Unknown"
    confidence: float = 0.0
    temperature_c: float | None = None
    humidity_rh: float | None = None
    voc_raw: float | None = None
    nox_raw: float | None = None
    nh3_value: float | None = None
    nh3_unit: str = "mV"
    h2s_value: float | None = None
    h2s_unit: str = "mV"
    system_status: str = "STARTING"
    updated_at: str = ""


def _number_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric or null") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_confidence(value: Any) -> float:
    confidence = _number_or_none(value, "confidence")
    if confidence is None:
        return 0.0
    # Accept either 0..1 or 0..100.
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def load_state(path: Path) -> DisplayState:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("display state JSON must contain one object")

    return DisplayState(
        food_type=str(raw.get("food_type", "Unknown")),
        freshness_level=str(raw.get("freshness_level", "Unknown")),
        confidence=_bounded_confidence(raw.get("confidence", 0.0)),
        temperature_c=_number_or_none(raw.get("temperature_c"), "temperature_c"),
        humidity_rh=_number_or_none(raw.get("humidity_rh"), "humidity_rh"),
        voc_raw=_number_or_none(raw.get("voc_raw"), "voc_raw"),
        nox_raw=_number_or_none(raw.get("nox_raw"), "nox_raw"),
        nh3_value=_number_or_none(raw.get("nh3_value"), "nh3_value"),
        nh3_unit=str(raw.get("nh3_unit", "mV")),
        h2s_value=_number_or_none(raw.get("h2s_value"), "h2s_value"),
        h2s_unit=str(raw.get("h2s_unit", "mV")),
        system_status=str(raw.get("system_status", "UNKNOWN")),
        updated_at=str(raw.get("updated_at", "")),
    )


def find_font(bold: bool = False) -> str | None:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = find_font(bold)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(24, True)
FONT_SECTION = font(14, True)
FONT_LARGE = font(29, True)
FONT_MEDIUM = font(20, True)
FONT_VALUE = font(18, True)
FONT_SMALL = font(12, False)
FONT_SMALL_BOLD = font(12, True)


BACKGROUND = (8, 13, 22)
CARD = (18, 27, 42)
CARD_ALT = (22, 34, 52)
BORDER = (44, 61, 82)
TEXT = (240, 245, 250)
MUTED = (146, 163, 184)
ACCENT = (45, 212, 191)
GOOD = (50, 205, 120)
WARN = (245, 176, 65)
BAD = (240, 91, 91)
BLUE = (74, 144, 226)


def freshness_color(level: str) -> tuple[int, int, int]:
    normalized = level.strip().lower()
    if normalized in {"fresh", "good", "safe"}:
        return GOOD
    if normalized in {"moderate", "warning", "aging"}:
        return WARN
    if normalized in {"spoiled", "bad", "unsafe"}:
        return BAD
    return BLUE


def status_color(status: str) -> tuple[int, int, int]:
    normalized = status.strip().lower()
    if normalized in {"ok", "ready", "running", "healthy"}:
        return GOOD
    if normalized in {"warning", "degraded", "stale"}:
        return WARN
    if normalized in {"error", "failed", "offline"}:
        return BAD
    return BLUE


def text_size(draw: ImageDraw.ImageDraw, text: str, use_font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=use_font)
    return box[2] - box[0], box[3] - box[1]


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, minimum: int = 14) -> ImageFont.ImageFont:
    for size in range(start, minimum - 1, -1):
        candidate = font(size, True)
        if text_size(draw, text, candidate)[0] <= max_width:
            return candidate
    return font(minimum, True)


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def draw_round_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int] = CARD,
) -> None:
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=BORDER, width=1)


def draw_metric(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    unit: str,
    color: tuple[int, int, int] = ACCENT,
) -> None:
    draw_round_card(draw, box, CARD_ALT)
    x0, y0, x1, y1 = box
    draw.text((x0 + 12, y0 + 9), label, font=FONT_SMALL_BOLD, fill=MUTED)
    draw.text((x0 + 12, y0 + 30), value, font=FONT_VALUE, fill=TEXT)
    if unit:
        width, _ = text_size(draw, unit, FONT_SMALL)
        draw.text((x1 - 12 - width, y1 - 22), unit, font=FONT_SMALL, fill=color)


def render_dashboard(state: DisplayState) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    # Header
    draw.text((18, 14), "ODOR SENSING", font=FONT_TITLE, fill=TEXT)
    draw.text((18, 44), "Real-time food freshness monitor", font=FONT_SMALL, fill=MUTED)
    draw.line((14, 65, WIDTH - 14, 65), fill=BORDER, width=1)

    # Primary summary card
    summary = (14, 78, WIDTH - 14, 205)
    draw_round_card(draw, summary)
    draw.text((28, 91), "FOOD TYPE", font=FONT_SECTION, fill=MUTED)
    food_font = fit_font(draw, state.food_type, 210, 29)
    draw.text((28, 116), state.food_type, font=food_font, fill=TEXT)

    level_color = freshness_color(state.freshness_level)
    pill_text = state.freshness_level.upper()
    pill_font = fit_font(draw, pill_text, 130, 15, 11)
    pill_w, pill_h = text_size(draw, pill_text, pill_font)
    pill_box = (WIDTH - 34 - max(118, pill_w + 28), 92, WIDTH - 28, 126)
    draw.rounded_rectangle(pill_box, radius=17, fill=level_color)
    draw.text(
        (
            (pill_box[0] + pill_box[2] - pill_w) // 2,
            (pill_box[1] + pill_box[3] - pill_h) // 2 - 1,
        ),
        pill_text,
        font=pill_font,
        fill=(8, 13, 22),
    )

    confidence_percent = int(round(state.confidence * 100))
    draw.text((28, 163), "Confidence", font=FONT_SMALL_BOLD, fill=MUTED)
    conf_text = f"{confidence_percent}%"
    conf_w, _ = text_size(draw, conf_text, FONT_MEDIUM)
    draw.text((WIDTH - 28 - conf_w, 151), conf_text, font=FONT_MEDIUM, fill=TEXT)

    bar = (28, 185, WIDTH - 28, 193)
    draw.rounded_rectangle(bar, radius=4, fill=(34, 47, 63))
    fill_right = bar[0] + int((bar[2] - bar[0]) * state.confidence)
    if fill_right > bar[0]:
        draw.rounded_rectangle((bar[0], bar[1], fill_right, bar[3]), radius=4, fill=level_color)

    # Sensor values: 2 columns x 3 rows
    left_x = 14
    gap = 10
    card_w = (WIDTH - 28 - gap) // 2
    right_x = left_x + card_w + gap
    y_positions = (218, 292, 366)
    card_h = 64

    draw_metric(
        draw, (left_x, y_positions[0], left_x + card_w, y_positions[0] + card_h),
        "TEMPERATURE", fmt(state.temperature_c), "°C", WARN
    )
    draw_metric(
        draw, (right_x, y_positions[0], right_x + card_w, y_positions[0] + card_h),
        "HUMIDITY", fmt(state.humidity_rh), "%RH", BLUE
    )
    draw_metric(
        draw, (left_x, y_positions[1], left_x + card_w, y_positions[1] + card_h),
        "VOC", fmt(state.voc_raw, 0), "SRAW", ACCENT
    )
    draw_metric(
        draw, (right_x, y_positions[1], right_x + card_w, y_positions[1] + card_h),
        "NOx", fmt(state.nox_raw, 0), "SRAW", ACCENT
    )
    draw_metric(
        draw, (left_x, y_positions[2], left_x + card_w, y_positions[2] + card_h),
        "NH₃", fmt(state.nh3_value, 2), state.nh3_unit, WARN
    )
    draw_metric(
        draw, (right_x, y_positions[2], right_x + card_w, y_positions[2] + card_h),
        "H₂S", fmt(state.h2s_value, 2), state.h2s_unit, BAD
    )

    # Status strip
    status_box = (14, 443, WIDTH - 14, 488)
    draw_round_card(draw, status_box)
    color = status_color(state.system_status)
    draw.ellipse((28, 457, 40, 469), fill=color)
    draw.text((50, 452), "SYSTEM", font=FONT_SMALL_BOLD, fill=MUTED)
    draw.text((50, 466), state.system_status.upper(), font=FONT_SMALL_BOLD, fill=TEXT)

    updated = state.updated_at.strip()
    if updated:
        timestamp = updated
    else:
        timestamp = datetime.now().strftime("%H:%M:%S")
    timestamp = timestamp[-19:]
    t_w, _ = text_size(draw, timestamp, FONT_SMALL)
    draw.text((WIDTH - 28 - t_w, 460), timestamp, font=FONT_SMALL, fill=MUTED)

    return image


def image_to_rgb565(image: Image.Image, bgr: bool = False) -> bytes:
    image = image.convert("RGB")
    if image.size != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

    source = image.tobytes()
    output = bytearray(WIDTH * HEIGHT * 2)
    out = 0
    for index in range(0, len(source), 3):
        red, green, blue = source[index], source[index + 1], source[index + 2]
        if bgr:
            red, blue = blue, red
        value = ((red & 0xF8) << 8) | ((green & 0xFC) << 3) | (blue >> 3)
        output[out] = (value >> 8) & 0xFF
        output[out + 1] = value & 0xFF
        out += 2
    return bytes(output)


def demo_state(step: int) -> DisplayState:
    levels = ("Fresh", "Fresh", "Moderate", "Spoiled")
    level = levels[(step // 4) % len(levels)]
    confidence = (0.94, 0.91, 0.78, 0.88)[(step // 4) % 4]
    return DisplayState(
        food_type=("Chicken", "Beef", "Salmon", "Apple")[(step // 8) % 4],
        freshness_level=level,
        confidence=confidence,
        temperature_c=24.0 + 0.6 * math.sin(step / 2),
        humidity_rh=51.0 + 2.4 * math.sin(step / 3),
        voc_raw=12100 + 180 * math.sin(step / 2),
        nox_raw=2450 + 60 * math.cos(step / 3),
        nh3_value=12.4 + 0.8 * math.sin(step / 4),
        nh3_unit="mV",
        h2s_value=3.2 + 0.3 * math.cos(step / 5),
        h2s_unit="mV",
        system_status="OK",
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show odor-sensing data on a CO5300 AMOLED.")
    parser.add_argument("--state-file", default="config/display_state.example.json")
    parser.add_argument("--init", default="config/co5300_init.json")
    parser.add_argument("--gpiochip", default="auto")
    parser.add_argument("--clk", type=int, default=21)
    parser.add_argument("--sio0", type=int, default=20)
    parser.add_argument("--sio1", type=int, default=19)
    parser.add_argument("--sio2", type=int, default=16)
    parser.add_argument("--sio3", type=int, default=26)
    parser.add_argument("--cs", type=int, default=18)
    parser.add_argument("--rst", type=int, default=25)
    parser.add_argument("--te", type=int, default=24)
    parser.add_argument("--half-period-us", type=int, default=5)
    parser.add_argument("--chunk-bytes", type=int, default=1024)
    parser.add_argument("--refresh-seconds", type=float, default=3.0)
    parser.add_argument("--bgr", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--skip-te-check", action="store_true")
    parser.add_argument("--off-at-end", action="store_true")
    parser.add_argument("--preview", metavar="PNG", help="render to PNG without using GPIO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.refresh_seconds <= 0:
        raise SystemExit("--refresh-seconds must be positive")

    state_path = Path(args.state_file)

    # Preview mode is useful on a normal computer and does not touch GPIO.
    if args.preview:
        state = demo_state(0) if args.demo else load_state(state_path)
        render_dashboard(state).save(args.preview)
        print(f"Saved dashboard preview to {args.preview}")
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
    init_commands = load_init_commands(Path(args.init))
    lgpio = import_lgpio()

    bus = LgpioQuadBus(lgpio, gpiochip, pins, args.half_period_us)
    display = CO5300Display(bus, init_commands, args.chunk_bytes)
    stop_requested = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print("CO5300 odor-sensing dashboard")
    print(f"  gpiochip : /dev/gpiochip{gpiochip}")
    print(f"  state    : {'demo generator' if args.demo else state_path}")
    print(f"  refresh  : {args.refresh_seconds:.1f} s")

    last_signature: tuple[Any, ...] | None = None
    step = 0

    try:
        bus.open()
        display.initialize()

        if not args.skip_te_check:
            edges = bus.te_edges(1.0)
            print(f"  TE edges : {edges}")
            if edges < 10:
                print("WARNING: low TE edge count", file=sys.stderr)

        while not stop_requested:
            try:
                state = demo_state(step) if args.demo else load_state(state_path)
            except ValueError as exc:
                print(f"State read warning: {exc}", file=sys.stderr)
                state = DisplayState(
                    food_type="Unknown",
                    freshness_level="Unknown",
                    confidence=0.0,
                    system_status="ERROR",
                    updated_at=datetime.now().isoformat(timespec="seconds"),
                )

            signature = tuple(state.__dict__.values())
            if signature != last_signature:
                print(
                    f"Drawing: food={state.food_type}, "
                    f"freshness={state.freshness_level}, "
                    f"confidence={state.confidence:.0%}, "
                    f"status={state.system_status}"
                )
                frame = image_to_rgb565(render_dashboard(state), args.bgr)
                display.write_frame(frame)
                last_signature = signature

            if args.once:
                break

            step += 1
            end = time.monotonic() + args.refresh_seconds
            while not stop_requested and time.monotonic() < end:
                time.sleep(0.1)

        if args.hold and not stop_requested:
            print("Dashboard is active. Press Ctrl+C to stop.")
            while not stop_requested:
                time.sleep(0.2)

        if args.off_at_end:
            display.shutdown()
        return 0

    except (CO5300Error, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())

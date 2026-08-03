from __future__ import annotations

from pathlib import Path
import sys

import pytest


pytest.importorskip("PIL")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extras" / "food_freshness" / "tools"))

from co5300_dashboard import (  # noqa: E402
    HEIGHT,
    WIDTH,
    image_to_rgb565,
    load_state,
    render_dashboard,
)


def test_classification_display_state_renders_to_panel_dimensions() -> None:
    state = load_state(
        ROOT
        / "extras"
        / "food_freshness"
        / "config"
        / "display_state.example.json"
    )

    assert state.food_type == "Banana"
    assert state.combined_class == "Fresh Banana"
    assert state.input_rows == 60
    assert state.valid_rows == 60
    assert state.model_windows == 9

    image = render_dashboard(state)
    frame = image_to_rgb565(image)

    assert image.size == (WIDTH, HEIGHT)
    assert len(frame) == WIDTH * HEIGHT * 2

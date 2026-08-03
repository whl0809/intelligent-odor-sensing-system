# Live Classification on the CO5300 Display

The live display workflow has two processes:

1. `python -m enose acquire-classify` samples the reduced sensor set, writes
   the acquisition CSV, classifies the latest input window at the selected
   update interval, and atomically publishes the newest result to
   `runtime/display_state.json`.
2. `extras/food_freshness/tools/co5300_dashboard.py` watches that JSON file and sends a rendered
   RGB565 frame to the CO5300 over software QSPI.

Keeping display transfer outside the acquisition process prevents a slow
screen refresh from delaying the 1 Hz sensor schedule. A display-state write
or rendering failure does not stop CSV acquisition.

## Install on Raspberry Pi

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[classification]'

sudo apt update
sudo apt install -y python3-lgpio python3-pil gpiod
```

The classifier runs with the virtual-environment Python. The display process
uses the Raspberry Pi OS Python so it can use the `lgpio` and Pillow packages
installed by `apt`.

## Run acquisition, classification, and display

With the virtual environment active:

```bash
bash extras/food_freshness/tools/run_acquire_classify_display.sh
```

Press `Ctrl+C` to stop both processes and close the CSV cleanly. For a bounded
hardware test:

```bash
bash extras/food_freshness/tools/run_acquire_classify_display.sh --frames 70
```

The display initially reports that it is collecting samples. The first
classification appears after row 60, then updates at rows 70, 80, and so on.
Each update is also printed in the terminal.

The launcher forwards classifier options. For example, to classify the latest
30 rows every 5 new rows:

```bash
bash extras/food_freshness/tools/run_acquire_classify_display.sh \
  --classification-window-rows 30 \
  --classification-update-rows 5
```

The input window minimum is 20 rows. Defaults are 60 input rows and 10 update
rows.

The screen shows:

- food-type prediction and confidence;
- freshness prediction and confidence;
- combined-class prediction and confidence;
- valid rows in the selected model input and completed model-window count;
- current NH3 and H2S differential voltages in mV;
- state status and the UTC update time.

These voltage readings are not ppm values. Classification labels and
confidence values are model outputs, not calibrated gas concentrations.

## Preview without display hardware

Pillow can render the example state to a PNG:

```bash
python3 extras/food_freshness/tools/co5300_dashboard.py \
  --state-file extras/food_freshness/config/display_state.example.json \
  --preview dashboard_preview.png
```

The preview path does not access GPIO.

## Verify the display before live acquisition

Follow the wiring and bring-up procedure in
`extras/food_freshness/docs/CO5300_RPI_QSPI.md`, then run:

```bash
python3 extras/food_freshness/tools/co5300_qspi_test.py --self-test
sudo ./extras/food_freshness/tools/run_co5300_qspi.sh
```

Confirm the color-bar pattern and TE activity before starting the live
classifier.

## Published state format

`extras/food_freshness/config/display_state.example.json` documents the JSON contract. The live
command writes the following fields atomically:

```text
food_type
freshness_level
combined_class
food_confidence
freshness_confidence
combined_confidence
input_rows
valid_rows
model_windows
nh3_value
nh3_unit
h2s_value
h2s_unit
system_status
updated_at
```

`runtime/` is intentionally ignored by Git.

## Troubleshooting

- If the classifier starts but the display stays unchanged, inspect
  `runtime/display_state.json` and the dashboard messages in the terminal.
- If GPIO is busy, stop any previous CO5300 process before retrying.
- If `lgpio` or Pillow cannot be imported by `python3`, confirm the `apt`
  packages above are installed for the system Python.
- If the screen is black, return to the standalone color-bar test and verify
  3.3 V power, common ground, H1 orientation, QSPI wiring, and TE activity.

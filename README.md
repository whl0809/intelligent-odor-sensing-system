# Bosch E-nose acquisition

Small, synchronous Python 3.11 application for acquiring the two E-nose
prototype boards at 1 Hz on a Raspberry Pi 5.

## Safety and wiring

Power each board from its own USB-C connector and connect a common ground.
Do **not** connect Raspberry Pi 3.3 V or 5 V to either H1 header. Wire the bus
by net name because the two H1 connectors swap SDA and SCL:

- Raspberry Pi GPIO2 / pin 3 -> both SDA nets
- Raspberry Pi GPIO3 / pin 5 -> both SCL nets
- Raspberry Pi ground -> both DGND nets

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m enose probe --config config/rpi5.toml
python -m enose diagnose --config config/rpi5.toml
python -m enose acquire --config config/rpi5.toml
```

Use `--verbose` after the subcommand for byte-level diagnostics. A bounded
acquisition can be run with `--frames 60`.

`acquire` is the only raw data-acquisition mode. Select any combination with a
comma-separated `--sensors` value. Valid names are `tgs`, `nh3`, `h2s`,
`bme690`, `sgp41`, `sht45`, and `svm41`:

```bash
python -m enose acquire \
  --config config/rpi5.toml \
  --sensors tgs,nh3,h2s,svm41 \
  --uart /dev/ttyUSB0
```

Omitting `--sensors` selects the I2C sensors enabled in `config/rpi5.toml` and
does not add SVM41. Use `--sensors all` for all seven choices. For example,
TGS and SVM41 only is:

```bash
python -m enose acquire \
  --config config/rpi5.toml \
  --sensors tgs,svm41 \
  --uart /dev/ttyUSB0
```

Only selected sensors are initialized and only their columns appear in the
terminal and CSV. Each file uses a descriptive name such as
`enose_raw_tgs-nh3-h2s-svm41_20260802T120000_000000Z.csv`, with a matching
metadata JSON. Use `--frames 60` for a bounded run; otherwise acquisition
continues until Ctrl+C.

The BME690 driver uses Bosch Sensortec's official BME690 SensorAPI v1.1.0
through a small native extension built during installation. It runs the sensor
in forced mode with the heater settings from `config/rpi5.toml`. BME690 remains
optional in the default configuration, so its initialization or read failures
do not stop the other sensors.

Sensirion's official `sensirion-gas-index-algorithm` package processes each
new 1 Hz SGP41 raw sample into VOC Index and NOx Index while preserving both
raw signals. These indices are not ppm or concentration measurements.

## SVM41 UART wiring

SVM41 temperature, humidity, VOC Index, and NOx Index can be included in any
acquisition selection. The indices are dimensionless, not ppm or concentration.

Keep NH3 (`0x69`) and H2S (`0x6A`) on `/dev/i2c-1`. The SVM41 module also uses
`0x6A` in I2C mode, so it must not be connected to that I2C bus. Connect the
SVM41 through its Sensirion USB-UART cable instead:

- SVM41 pin 1/red: VDD, 3.3 V or 5 V
- SVM41 pin 2/black: GND
- SVM41 pin 3/green: module RX
- SVM41 pin 4/yellow: module TX
- SVM41 pin 5/blue: SEL, leave floating or pull to VDD for UART
- SVM41 pin 6/purple: do not connect

With the provided USB-UART cable, plug the cable into the Raspberry Pi and
confirm its device name:

```bash
ls -l /dev/ttyUSB*
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. If Linux assigns a
different serial path, pass that path with `--uart`. Install the new official
driver dependency after pulling this change:

```bash
source .venv/bin/activate
python -m pip install -e .
```

If the Raspberry Pi's configured piwheels mirror times out, install from PyPI
without the global pip configuration:

```bash
PIP_CONFIG_FILE=/dev/null \
PIP_INDEX_URL=https://pypi.org/simple \
PIP_DEFAULT_TIMEOUT=120 \
python -m pip install -e .
```

## Food and freshness classification

The shared classifier supports both live inference and offline processing of a
CSV containing TGS, NH3, and H2S fields. Classification does not change the
acquisition CSV or the sensor-reading schedule.

Install the optional ML dependencies:

```bash
python -m pip install -e '.[classification]'
```

For automatic acquisition and live classification, run:

```bash
python -m enose acquire-classify --config config/rpi5.toml
```

This command acquires ADS7828/TGS, NH3, and H2S data at the configured rate,
prints and saves every frame, and loads the three model pipelines once. The
default first classification uses acquisition rows 1–60. Classification then
runs every 10 new rows using a 60-row sliding input, so consecutive inputs
retain 50 rows. Each classification and confidence are printed in the
terminal. Press Ctrl+C to close the CSV cleanly.

Set a different input size and update interval when needed:

```bash
python -m enose acquire-classify \
  --config config/rpi5.toml \
  --classification-window-rows 30 \
  --classification-update-rows 5
```

The classification window must contain at least 20 rows. The update interval
must be between 1 and the selected window size. Defaults remain 60 and 10.

To show the same live results on the CO5300 display, first install its
system-level GPIO and rendering dependencies:

```bash
sudo apt update
sudo apt install -y python3-lgpio python3-pil gpiod
```

With the project virtual environment active, start the display, acquisition,
and classifier together:

```bash
bash extras/food_freshness/tools/run_acquire_classify_display.sh
```

The display shows food type, freshness, combined class, all three model
confidences, valid rows in the current classification input, model-window
count, and the current NH3/H2S differential voltages. The dashboard runs
separately from the 1 Hz acquisition loop, so a slow software-QSPI refresh
does not delay sensor sampling. See
`extras/food_freshness/docs/CO5300_DASHBOARD.md` for wiring and preview
commands.

To classify a previously completed CSV instead, run:

```bash
python extras/food_freshness/tools/classify_csv.py \
  data/raw/enose_raw_tgs-nh3-h2s_20260802T120000_000000Z.csv
```

The utility writes per-window predictions and an overall JSON summary under
`data/classification/<csv filename>/`. Its fixed training baseline and three
model artifacts and historical training inputs are grouped under
`extras/food_freshness/`. Joblib model files use Python pickle internally, so
load only repository artifacts whose provenance is trusted.

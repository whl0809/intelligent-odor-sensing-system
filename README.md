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
python -m enose acquire-no-sgp41-bme690-sht45 --config config/rpi5.toml
```

Use `--verbose` after the subcommand for byte-level diagnostics. A bounded
acquisition can be run with `--frames 60`.

Both standard acquisition commands print every CSV field for each frame while
writing the same frame to disk. `acquire` uses all enabled sensors on both
PCBs. `acquire-no-sgp41-bme690-sht45` records only all six TGS/ADS7828
channels, NH3, and H2S. It disables SGP41, BME690, and SHT45; those devices
do not appear in that mode's terminal rows or CSV columns, and no SVM41 is
used.

## Four-state hard-code demonstration

For the current eight labeled food recordings, use the transparent rule-based
command instead of the generalization model:

```bash
python -m enose acquire-classify --config config/rpi5.toml --frames 120 \
  --display-state runtime/display_state.json
```

It waits 60 seconds, averages the latest 10 raw readings from TGS2603,
TGS2620, and TGS2602, and requires five matching rule outputs before showing a
final label. TGS2603 separates fresh meat, fresh banana, and non-fresh states;
TGS2620 plus TGS2602 confirm spoiled meat. The 2150--2250 TGS2603 boundary and
any non-demonstrated combination are reported as `uncertain`, rather than
being forced into a food class. This is a demo rule for the current device and
eight recordings only; it is not a calibrated gas measurement, food-safety
result, or cross-day validation.

The BME690 driver uses Bosch Sensortec's official BME690 SensorAPI v1.1.0
through a small native extension built during installation. It runs the sensor
in forced mode with the heater settings from `config/rpi5.toml`. BME690 remains
optional in the default configuration, so its initialization or read failures
do not stop the other sensors.

Sensirion's official `sensirion-gas-index-algorithm` package processes each
new 1 Hz SGP41 raw sample into VOC Index and NOx Index while preserving both
raw signals. These indices are not ppm or concentration measurements.

## NH3, H2S, and SVM41 UART mode

The isolated `acquire-svm41` mode records only the two existing MCP3421
channels and a Sensirion SVM41 module:

```bash
python -m enose acquire-svm41 \
  --config config/rpi5.toml \
  --uart /dev/ttyUSB0
```

It runs continuously at 1 Hz until Ctrl+C, prints every frame, and writes a
timestamped `enose_nh3_h2s_svm41_*.csv` plus metadata in `data/raw/`.
VOC Index and NOx Index are dimensionless indices, not ppm or concentration.

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

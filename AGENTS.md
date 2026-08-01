# AGENTS.md

## Mission

Build a compact Python data-acquisition application for the Bosch E-nose prototype on a Raspberry Pi 5.

Two independently powered PCBs share one I2C bus:

- **TGS board:** six Figaro TGS sensors, ADS7828 ADC, SHT45.
- **Electrochemical/environment board:** NH3 and H2S analog front ends with two MCP3421 ADCs, SGP41, BME690.

The current scope is trustworthy 1 Hz time-series acquisition and CSV logging. Do not add ML inference, dashboards, networking, databases, actuators, or ppm conversion unless explicitly requested.

## Keep the implementation small

- Use Python 3.11+ and `smbus2`.
- Prefer a synchronous loop. Do not add threads or `asyncio` without measured need.
- Do not introduce C++, CMake, bindings, plugin systems, dependency-injection frameworks, or a generic sensor hierarchy.
- Each device driver should be one focused module with a small API.
- Use dataclasses for returned samples.
- Use `tomllib` for configuration.
- Delete obsolete paths when behavior changes.
- Keep byte-level diagnostics behind `--verbose`.
- Never preserve old code merely because it may be useful later.
- Before adding an abstraction, explain why a direct implementation is insufficient.

## Hardware source of truth

Store exports under:

```text
hardware/
  tgs_board/
    schematic.pdf
    netlist.tel
    bom.xlsx
    pcb.pdf
    interactive_bom.html
  electrochemical_board/
    schematic.pdf
    netlist.tel
    bom.xlsx
    pcb.pdf
    interactive_bom.html
```

Resolve hardware facts in this order:

1. Schematic and netlist
2. Exact BOM part number
3. Official manufacturer datasheet
4. PCB layout / interactive BOM
5. Existing software only as historical reference

Do not copy assumptions from the previous ADS114S06/C++ repository. The redesigned TGS board uses **ADS7828 over I2C**, not ADS114S06 over SPI.

Stop and report any conflict between the schematic, netlist, BOM, and datasheet. Do not guess.

Official references:

- ADS7828: https://www.ti.com/lit/ds/symlink/ads7828.pdf
- MCP3421: https://ww1.microchip.com/downloads/en/devicedoc/22003e.pdf
- SHT4x: https://sensirion.com/resource/datasheet/sht4x
- SGP41: https://sensirion.com/resource/datasheet/sgp41
- BME690: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme690-ds001.pdf

## Raspberry Pi wiring

Target:

- Raspberry Pi 5 / Raspberry Pi OS Bookworm
- I2C device: `/dev/i2c-1`
- SDA: GPIO2, physical pin 3
- SCL: GPIO3, physical pin 5
- Common ground required

Both boards are powered through their own USB-C ports. Their H1 headers do not supply power to the Pi. Do not connect Pi 3.3 V or 5 V to H1.

### Connector warning

The two boards swap the SDA and SCL header positions.

**Electrochemical board H1**

| Pin | Net |
|---:|---|
| 1 | SENSOR_SDA |
| 2 | DGND |
| 3 | SENSOR_SCL |
| 4 | DGND |
| 5 | NC |
| 6 | DGND |

**TGS board H1**

| Pin | Net |
|---:|---|
| 1 | SCL |
| 2 | DGND |
| 3 | SDA |
| 4 | DGND |
| 5 | NC |
| 6 | DGND |

Wire by net name, not by matching pin number. Both boards already contain I2C pull-ups; do not add more by default.

## I2C device map

All addresses are 7-bit.

| Address | Device | Board |
|---:|---|---|
| `0x44` | SHT45-AD1F-R2 | TGS |
| `0x48` | ADS7828E/2K5 | TGS |
| `0x59` | SGP41-D-R4 | Electrochemical |
| `0x69` | MCP3421A1T-E/CH, NH3 | Electrochemical |
| `0x6A` | MCP3421A2T-E/CH, H2S | Electrochemical |
| `0x76` | BME690 | Electrochemical |

When all devices are powered, `probe` should expect:

```text
0x44 0x48 0x59 0x69 0x6A 0x76
```

An ACK proves only that an address responds. Use identity/self-test checks where available.

An external SVM41 module also uses I2C address `0x6A`. Never connect it to the
same I2C bus as the H2S MCP3421. For every SVM41 acquisition mode, keep any
enabled MCP3421 and ADS7828 devices on I2C and connect SVM41 over UART/SHDLC at
115200 baud.

## TGS board

### ADS7828 channel mapping

| ADS channel | Sensor signal |
|---:|---|
| CH0 | TGS2620 |
| CH1 | TGS2610 |
| CH2 | TGS2611 |
| CH3 | TGS2600 |
| CH4 | TGS2602 |
| CH5 | TGS2603 |
| CH6 | AGND, unused |
| CH7 | AGND, unused |

Do not use the old ADS114S06 channel order.

### ADS7828 protocol

Hardware configuration:

- Address `0x48`
- VDD 3.3 V
- COM connected to AGND
- A0 and A1 grounded
- Internal 2.5 V reference
- Single-ended 12-bit readings

Command byte:

```text
bit 7      SD       1 = single-ended
bits 6:4   C2:C0    channel selection code
bits 3:2   PD1:PD0  11 = internal reference on, ADC on
bits 1:0            00
```

Physical channel and command mapping:

| Channel | C2:C0 | Command |
|---:|---:|---:|
| CH0 | `000` | `0x8C` |
| CH1 | `100` | `0xCC` |
| CH2 | `001` | `0x9C` |
| CH3 | `101` | `0xDC` |
| CH4 | `010` | `0xAC` |
| CH5 | `110` | `0xEC` |
| CH6 | `011` | `0xBC` |
| CH7 | `111` | `0xFC` |

Startup:

1. Send a command with the internal reference enabled.
2. Wait at least 2 ms.
3. Discard the first conversion.
4. Keep `PD1 = 1` in every later command.

Read two bytes and parse:

```python
raw = ((byte0 & 0x0F) << 8) | byte1
voltage_v = raw * 2.5 / 4096.0
```

Flag values near 0 or 4095 as possible short/open/saturation conditions. Do not convert TGS voltage to ppm without an explicit calibration model.

### SHT45

- Address `0x44`.
- Validate Sensirion CRC for every returned word.
- Return temperature in °C and RH in %.
- Do not hide smoothing inside the driver.

## Electrochemical/environment board

### MCP3421 mapping

| Channel | Device/address | VIN+ | VIN- |
|---|---|---|---|
| NH3 | A1 option, `0x69` | TIA_VOUT_1 | VBIAS |
| H2S | A2 option, `0x6A` | TIA_VOUT | VBIAS |

Expose resolution, gain, and mode in TOML. Default configuration may use 18-bit, gain 1, continuous mode, but it must remain configurable.

Decode signed 12/14/16/18-bit data correctly. Use:

```python
voltage_v = raw * 2.048 / ((2 ** (resolution_bits - 1)) * gain)
```

Check the returned RDY bit. Log raw code and differential voltage. Do not report NH3 or H2S ppm without validated calibration coefficients and the complete analog transfer function.

### SGP41

- Address `0x59`.
- Validate all CRC bytes.
- After power-up or heater-off, condition for 10 seconds; never exceed 10 seconds.
- Then measure once per second.
- Each raw measurement may take up to 50 ms.
- Use current SHT45 RH/T values for humidity compensation.
- If current SHT45 data are unavailable, mark compensation unavailable; do not use stale or invented values.
- Log `sraw_voc` and `sraw_nox`.
- Add VOC/NOx indices only through Sensirion's official Gas Index Algorithm at 1 Hz.
- Turn the heater off on graceful shutdown.

### BME690

- Address `0x76`.
- Verify chip identity during probing.
- Use Bosch's official BME68x/BME690 API or a Python driver explicitly verified for BME690.
- Do not assume a BME680/BME688 library fully supports BME690.
- Keep the adapter small.
- Log compensated temperature, humidity, pressure, gas resistance, gas-valid, and heater-stable fields when available.
- Keep heater/profile settings in TOML.
- Do not add BSEC/BME AI outputs unless the required Bosch software and configuration are explicitly introduced.
- BME690 failure must not stop otherwise valid acquisition.

## Repository structure

```text
AGENTS.md
README.md
pyproject.toml
config/rpi5.toml
hardware/
models/
  food_freshness/
    dataset_and_preprocessing_metadata.json
    combined_class_best_model.joblib
    food_type_best_model.joblib
    freshness_best_model.joblib
src/enose/
  __init__.py
  cli.py
  config.py
  i2c_bus.py
  acquisition.py
  csv_logger.py
  records.py
  ads7828.py
  mcp3421.py
  sht45.py
  sgp41.py
  bme690.py
  classification.py
  svm41_uart.py
  svm41_acquisition.py
tests/
tools/
  classify_food_freshness.py
  co5300_dashboard.py
  co5300_qspi_test.py
  run_acquire_classify_display.sh
docs/
  CO5300_DASHBOARD.md
  CO5300_RPI_QSPI.md
data/raw/
```

Do not add more layers or directories without a concrete need.

Keep offline food/freshness classification isolated from acquisition. Store
its fixed metadata and model artifacts under `models/food_freshness/`, and run
completed CSV files through `tools/classify_food_freshness.py`. Do not load
untrusted joblib/pickle artifacts.

The live classification path uses the reduced ADS7828/TGS, NH3, and H2S
acquisition configuration. Load model artifacts once. Use a configurable
sliding input of at least 20 acquisition rows and a configurable update
interval no larger than that input; default to 60 and 10 rows respectively.
Keep classification failures from stopping CSV acquisition.

For CO5300 visualization, publish classification state atomically and keep the
software-QSPI dashboard in a separate process so display transfer time cannot
delay 1 Hz sensor acquisition. `tools/run_acquire_classify_display.sh` is the
one-command launcher for this workflow.

## CLI

Implement:

```bash
python -m enose probe --config config/rpi5.toml
python -m enose diagnose --config config/rpi5.toml
python -m enose acquire --config config/rpi5.toml
python -m enose acquire-no-sgp41-bme690-sht45 --config config/rpi5.toml
python -m enose acquire-no-sgp41-bme690-sht45-with-svm41 --config config/rpi5.toml --uart /dev/ttyUSB0
python -m enose acquire-tgs-svm41 --config config/rpi5.toml --uart /dev/ttyUSB0
python -m enose acquire-classify --config config/rpi5.toml [--classification-window-rows N] [--classification-update-rows N] [--display-state PATH]
python -m enose acquire-svm41 --config config/rpi5.toml --uart /dev/ttyUSB0
```

- `probe`: check expected addresses and identities; concise table.
- `diagnose`: one complete read from enabled devices; detailed bytes only with `--verbose`.
- `acquire`: continuous time-series recording and terminal output; default one frame per second.
- `acquire-no-sgp41-bme690-sht45`: records ADS7828/TGS, NH3, and H2S only.
  SGP41, BME690, and SHT45 are disabled and omitted from terminal and CSV
  output; no SVM41.
- `acquire-no-sgp41-bme690-sht45-with-svm41`: records the same ADS7828/TGS,
  NH3, and H2S fields plus SVM41 temperature, humidity, VOC Index, and NOx
  Index over UART. SGP41, BME690, and SHT45 remain omitted.
- `acquire-tgs-svm41`: records ADS7828/TGS plus SVM41 temperature, humidity,
  VOC Index, and NOx Index over UART. It does not initialize or expose NH3,
  H2S, SGP41, BME690, or SHT45.
- `acquire-classify`: the same reduced acquisition plus persistent live
  food/freshness classification with configurable input/update row counts,
  defaulting to a 60-row input updated every 10 frames.
- `acquire-svm41`: isolated 1 Hz NH3/H2S I2C plus SVM41 UART recording until Ctrl+C.

Required devices may fail the command. Optional device failures must be reported without blocking the remaining sensors.

## Acquisition loop

Prefer this synchronous order:

1. Start UTC and monotonic frame timestamps.
2. Read SHT45.
3. Trigger/read SGP41 with current RH/T compensation.
4. Read all six ADS7828 channels.
5. Read NH3 and H2S MCP3421 values.
6. Read BME690.
7. Assemble one immutable frame.
8. Write one CSV row.
9. Sleep until the next absolute monotonic deadline.

Do not use `sleep(1)` after each completed frame because that accumulates drift. Advance from absolute deadlines.

Never reuse a previous sensor value as if it were current.

## CSV contract

Create a timestamped file in `data/raw/`, one row per logical frame, UTC ISO-8601 timestamps with `Z`.

Minimum fields:

```text
timestamp_utc, elapsed_s, sequence, frame_duration_ms, deadline_miss_ms

sht45_temperature_c, sht45_relative_humidity_pct, sht45_ok

tgs2620_raw, tgs2620_voltage_v
tgs2610_raw, tgs2610_voltage_v
tgs2611_raw, tgs2611_voltage_v
tgs2600_raw, tgs2600_voltage_v
tgs2602_raw, tgs2602_voltage_v
tgs2603_raw, tgs2603_voltage_v
ads7828_ok

nh3_raw, nh3_diff_voltage_v, nh3_ok
h2s_raw, h2s_diff_voltage_v, h2s_ok

sgp41_sraw_voc, sgp41_sraw_nox
sgp41_voc_index, sgp41_nox_index
sgp41_compensated, sgp41_ok

bme690_temperature_c
bme690_relative_humidity_pct
bme690_pressure_pa
bme690_gas_resistance_ohm
bme690_gas_valid
bme690_heater_stable
bme690_ok

error_codes
```

Rules:

- Missing values are empty/NaN, not zero.
- `*_ok` means the frame contains a new valid sample.
- Never substitute a stale value after a failed read.
- `error_codes` uses concise machine-readable codes separated by semicolons.
- Write a sidecar metadata JSON/TOML containing effective configuration, software commit, hostname, start time, and enabled devices.
- Flush at least every 10 rows and always on shutdown.
- Keep column order stable; schema changes require a version.

## Configuration

Use `config/rpi5.toml` with explicit addresses, enable/required flags, acquisition interval, output directory, MCP3421 settings, and BME690 heater/profile settings.

Protocol constants belong in drivers, not in configuration.

## Errors and logging

- Use clear driver exceptions.
- Convert exceptions to frame status at the acquisition boundary.
- Limit retries to 0 or 1 by default.
- Distinguish NACK, CRC failure, timeout/not-ready, invalid identity, saturation, and parse errors.
- Put stack traces in the application log, not the CSV.
- Never fabricate measurements.
- Unit tests prove protocol logic; only Raspberry Pi tests prove hardware operation.

## Tests

Use `pytest` with small fake I2C responses.

Required coverage:

- ADS7828 channel-command mapping, parsing, 2.5 V conversion, sensor order, saturation.
- MCP3421 signed parsing for all resolutions, RDY, gain, voltage conversion, byte count.
- SHT45/SGP41 CRC validation.
- SGP41 RH/T tick conversion, conditioning limit, 1 Hz behavior.
- One failed sensor does not corrupt other fields.
- Failed values remain missing, not stale.
- Stable CSV header/order.
- Deadline scheduler does not accumulate drift.
- Graceful shutdown flushes/closes and turns off SGP41 heater.

Do not build a large hardware simulator.

## Raspberry Pi validation

Before claiming completion:

1. Confirm `/dev/i2c-1` access.
2. Run `i2cdetect -y 1`.
3. Run `probe`.
4. Run `diagnose`.
5. Run `acquire` for at least 60 seconds.
6. Verify one row per frame, correct TGS mapping, SGP41 1 Hz operation, visible CRC/NACK failures, and clean Ctrl+C shutdown.
7. Compare representative analog voltages with a multimeter when practical.
8. Do not claim ppm accuracy or classification performance.

## Rules for Codex changes

For every change:

1. Read this file and the relevant hardware exports.
2. State the smallest planned change.
3. Preserve validated behavior.
4. Remove superseded code and tests in the same change.
5. Do not add temporary debug logs or investigation history to this file.
6. Update this file only for stable hardware facts or permanent workflows.
7. Report:
   - files changed
   - obsolete files/paths removed
   - tests run
   - Raspberry Pi commands
   - remaining hardware assumptions

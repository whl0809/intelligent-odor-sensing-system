# ECE450 Odor-Sensing Raspberry Pi 5 Application — outdated C++ version

Linux user-space scaffold for the Raspberry Pi 5 version of the odor-sensing
project. The software platform is Raspberry Pi OS, CMake, and C++17.

The executable can run either with safe mock interfaces or with a validated
machine-specific Raspberry Pi runtime configuration. It handles `SIGINT` and
`SIGTERM` and prints a non-blocking heartbeat during normal execution.

## Hardware Architecture

```text
TGS2610 / TGS2620 / TGS2603 / TGS2602 / TGS2600 / TGS2611
    -> analog sensing circuits
    -> ADS114S06
    -> Linux spidev
    -> Raspberry Pi 5

ES1-NH3-100
    -> dedicated NH3 analog front end
    -> MCP3421A1T-E/CH at 0x69
    -> Linux I2C
    -> Raspberry Pi 5

ES1-H2S-100
    -> dedicated H2S analog front end
    -> MCP3421A2T-E/CH at 0x6A
    -> Linux I2C
    -> Raspberry Pi 5

SGP41-D-R4 at 0x59
    -> Linux I2C
    -> Raspberry Pi 5

SHT45-AD1F-R2 at 0x44
    -> Linux I2C
    -> Raspberry Pi 5

BME690 at 0x76
    -> Linux I2C
    -> Raspberry Pi 5
```

## Confirmed Board Contract

Board 1 `CN1` exposes `SENSOR_SCL`, `SENSOR_SDA`, and `DGND`; pins 5 and 6 are
not yet confirmed and must not be used.

Board 1 I2C devices:

```text
SGP41          0x59
NH3 MCP3421    0x69
H2S MCP3421    0x6A
BME690         0x76
```

Board 2 ADS114S06 connector:

```text
CN1 pin 1 -> START
CN1 pin 2 -> DIN   (Raspberry Pi MOSI)
CN1 pin 3 -> SCLK
CN1 pin 4 -> DOUT  (Raspberry Pi MISO)
CN1 pin 5 -> DRDY#
CN1 pin 6 -> AGND
```

ADS114S06 board wiring:

```text
CS#     -> DGND, permanently selected
RESET#  -> IOVDD, not Raspberry Pi controlled
CLK     -> DGND, internal clock configuration
REFP0   -> external 4.096 V REF5040AIDR reference
REFN0   -> AGND
```

The current software profile selects the ADS114S06 internal 2.5 V reference
with REFP/REFN input buffers disabled.

Do not add an ADS114S06 chip-select GPIO. The ADC must be the only active
device on its physical SPI data lines.

TGS channel order:

```text
AIN0 -> TGS2610_VOUT
AIN1 -> TGS2620_VOUT
AIN2 -> TGS2603_VOUT
AIN3 -> TGS2602_VOUT
AIN4 -> TGS2600_VOUT
AIN5 -> TGS2611_VOUT
AINCOM -> AGND
```

The SHT45 pads on Board 2 are:

```text
Pad 1 -> SHT45_SDA
Pad 2 -> SHT45_SCL
Pad 3 -> VDD3V3
Pad 4 -> AGND
```

Under the current independent-power plan, connect Raspberry Pi SDA, SCL, and
GND only. Do not connect Raspberry Pi 3.3 V to SHT45 Pad 3 unless the power
architecture is intentionally changed and revalidated.

## Electrochemical Front Ends

Current provisional assembly assumption:

```text
NH3 JP2 = OPEN -> three-electrode sensor
H2S JP2 = OPEN -> three-electrode sensor
```

Both signal chains use OPA2192 analog front ends and MCP3421 differential
measurement with `VIN+` at TIA output and `VIN-` at `VBIAS`. Nominal metadata:

```text
VBIAS ~= 1.65 V from 10 kOhm / 10 kOhm divider
TIA feedback resistor = 1 MOhm
TIA feedback capacitor = 10 uF
TIA input series resistor = 100 Ohm
Output low-pass resistor = 49.9 kOhm
Output low-pass capacitor = 10 uF
```

Signal polarity, zero offset, sensitivity, and concentration calibration remain
unvalidated. Concentration values must not be reported as valid yet.

## Power Rules

```text
Raspberry Pi 5 -> independently powered
Board 1         -> independently powered through its own USB-C input
Board 2         -> independently powered through its own USB-C input
All three       -> common ground
```

Do not parallel the Raspberry Pi 3.3 V or 5 V rails with either board's
regulated supply. Evaluate combined I2C pull-up resistance and back-powering
before joining independently powered I2C domains.

## Software Structure

```text
include/
  config.h
  error_flags.h
  sensor_types.h
  drivers/
  hardware/
  services/

src/
  main.cpp
  drivers/
  hardware/linux/
  hardware/mock/
  services/

tests/unit/
config/odor-sensing.example.toml
config/odor-sensing.rpi5.toml
```

## Raspberry Pi OS Dependencies

```bash
sudo apt install build-essential cmake libgpiod-dev
```

Do not edit boot configuration, device-tree overlays, udev rules, user groups,
or systemd services until the wiring and device-access plan are reviewed.

## Build And Test

Default Raspberry Pi build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On non-Linux development hosts, build the portable/mock scaffold without Linux
hardware wrappers:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DODOR_BUILD_LINUX_HARDWARE=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Run

Validate the selected machine-specific configuration without starting
acquisition:

```bash
./build/odor_sensing_app --config config/odor-sensing.rpi5.toml --validate-only
```

Run independent hardware diagnostics for the configured devices:

```bash
./build/odor_sensing_app --config config/odor-sensing.rpi5.toml --diagnostic
```

When only the ADS114S06/TGS PCB is connected, use the TGS-only profile:

```bash
./build/odor_sensing_app --config config/odor-sensing.rpi5-tgs-only.toml --diagnostic
```

Run the normal application loop with hardware enabled after validation passes:

```bash
./build/odor_sensing_app --config config/odor-sensing.rpi5.toml
```

With ADS114S06 enabled, normal mode records continuous six-channel TGS scans
to a timestamped CSV file under the configured raw CSV output directory
(`data/` in the Raspberry Pi profiles). Stop with `Ctrl+C`; the application
flushes and closes the CSV, sends ADS serial STOP, and prints the file path and
scan count.

Run without hardware access:

```bash
./build/odor_sensing_app --mock
```

Expected output:

```text
Odor Sensing Raspberry Pi 5 Application
Version: scaffold-0.1
Build: ...
Target: Raspberry Pi 5 / Raspberry Pi OS
Runtime configuration validated: config/odor-sensing.rpi5.toml
spi_actual,mode=1,bits_per_word=8,max_speed_hz=1000000
ADS114S06 initialization: OK,error_flags=0
TGS CSV: data/tgs_timeseries_YYYYMMDD_HHMMSSZ.csv
status,tgs_scans=1,last_ok=true,error_flags=0
```

Stop with `Ctrl+C`.

## Provisional Runtime Profile

The example TOML currently selects these provisional software defaults:

- SHT45 at 1 Hz, high precision, heater off, CRC required.
- SGP41 at 1 Hz using recent valid SHT45 compensation when available.
- BME690 at `0x76` with Bosch SensorAPI-derived forced-measurement compensation and heater setup.
- MCP3421 one-shot, 16-bit, gain x1, retaining signed raw code and differential voltage only.
- ADS114S06 internal 2.5 V reference, fixed TGS channel mapping, and raw-code/voltage output only.
- ADS114S06 TGS acquisition uses START/SYNC held low, serial START after
  configuration, 80 ms after each INPMUX change, and RDATA `[0x12, 0x00,
  0x00]` parsed from RX bytes 1 and 2.

The machine-specific `config/odor-sensing.rpi5.toml` currently selects:

- shared I2C adapter `/dev/i2c-1`
- ADS114S06 SPI device `/dev/spidev0.0`
- SPI mode `1`, 8 bits per word, provisional `1 MHz` clock
- GPIO controller `/dev/gpiochip4`, expected label `pinctrl-rp1`
- ADS START GPIO line `17`, active high
- ADS DRDY# GPIO line `27`, active low
- ADS START/SYNC held low; conversions use serial START commands.
- TGS CSV output directory `data/` with a 1000 ms scan interval.

`config/odor-sensing.rpi5-tgs-only.toml` uses the same ADS/SPI/GPIO settings
but disables SGP41, BME690, both MCP3421 devices, and SHT45 for the current
single-PCB ADS/TGS diagnostic setup.

## Raw CSV Schema

`RawCsvLogger` writes schema version `1` with stable columns for monotonic and
wall-clock timestamps, system state, validity flags, error flags, TGS raw
codes/voltages, NH3/H2S raw codes and differential voltages, SGP41
`SRAW_VOC`/`SRAW_NOX`, BME690 fields/status, and SHT45 temperature/humidity.

Normal ADS/TGS recording writes a smaller time-series CSV with one complete
six-channel scan per row:

```text
timestamp_utc,elapsed_ms,
tgs0_raw,tgs0_voltage_v,...,tgs5_raw,tgs5_voltage_v,error_flags
```

The schema intentionally does not report NH3/H2S ppm, electrochemical current,
TGS resistance, TGS `Rs/R0`, TGS gas concentration, or odor classification as
valid outputs.

## Still Unresolved

- Connected-board validation of `/dev/i2c-1`, `/dev/spidev0.0`, `/dev/gpiochip4`, GPIO17, and GPIO27
- Confirmation that `/dev/gpiochip4` still reports label `pinctrl-rp1` after boot
- Confirmation of the five expected I2C addresses on `/dev/i2c-1`
- ADS114S06 long-duration stability, noise, saturation, disconnect behavior, and calibrated TGS interpretation
- MCP3421 gain, resolution, and conversion mode
- Electrochemical polarity, zero offset, sensitivity, and calibration constants
- TGS heater control requirements and per-model calibration constants
- Raspberry Pi service user, device permissions, and deployment model
- BME690 compensated outputs are implemented against Bosch's BME690 SensorAPI
  logic, but still require Raspberry Pi and physical-board validation before
  they are treated as production-ready.

## Next Step

Run the native Raspberry Pi build, then run `--validate-only`. After wiring and
power checks are complete, run `--diagnostic` to initialize and read each known
sensor path independently. Do not treat the output as calibrated gas data.

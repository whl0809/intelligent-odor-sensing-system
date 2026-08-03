# CO5300 410x502 AMOLED Raspberry Pi 5 bring-up

This optional display path uses ordinary GPIO plus the `lgpio` waveform
engine. It does not use `/dev/spidev*` and does not conflict with the I2C
sensor bus.

The implementation, launcher, and initialization table are grouped under
`extras/food_freshness/`.

## Power-off wiring

Turn the Raspberry Pi off before changing wiring. Software uses BCM numbers.

| Display H1 | Signal | Raspberry Pi BCM | Physical pin |
|---:|---|---:|---:|
| 14 | GND | GND | 6 |
| 13 | VCC | 3.3 V | 1 or 17 |
| 12 | QSPI_CLK | GPIO21 | 40 |
| 11 | QSPI_SIO0 | GPIO20 | 38 |
| 10 | QSPI_SIO1 | GPIO19 | 35 |
| 9 | QSPI_SIO2 | GPIO16 | 36 |
| 8 | QSPI_SIO3 | GPIO26 | 37 |
| 7 | QSPI_CS | GPIO18 | 12 |
| 6 | QSPI_RST | GPIO25 | 22 |
| 5 | AMOLED_TE | GPIO24 | 18 |
| 4 | TP_SCL | leave disconnected | - |
| 3 | TP_SDA | leave disconnected | - |
| 2 | TP_INT | leave disconnected | - |
| 1 | TP_RST | leave disconnected | - |

Never connect display VCC to 5 V. Verify the H1 pin-1 marker before applying
power.

## Install and offline test

```bash
sudo apt update
sudo apt install -y python3-lgpio

python3 extras/food_freshness/tools/co5300_qspi_test.py --self-test
```

Expected output:

```text
[PASS] CO5300 offline self-test
```

## GPIO availability

Stop any previous display process, then check the GPIO controller:

```bash
ls -l /dev/gpiochip*
for f in /sys/class/gpio/gpiochip*/label; do echo "$f: $(cat "$f")"; done
```

## Physical display test

```bash
sudo ./extras/food_freshness/tools/run_co5300_qspi.sh
```

The display should show a color-bar pattern and report TE activity. Other
patterns can be selected directly:

```bash
sudo python3 extras/food_freshness/tools/co5300_qspi_test.py \
  --pattern checker \
  --hold
```

If red and blue are reversed, add `--bgr`. If wiring is long or unstable, add
`--half-period-us 5`.

The initialization table is stored in
`extras/food_freshness/config/co5300_init.json`. Replace or extend that file
when a complete panel-specific register table is available; the QSPI transport
does not need to change.

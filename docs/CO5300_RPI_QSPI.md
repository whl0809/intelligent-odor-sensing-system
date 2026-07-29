# CO5300 410×502 AMOLED — Raspberry Pi 5 QSPI bring-up

This bring-up path uses ordinary GPIO plus the `lgpio` waveform engine. It
does **not** use `/dev/spidev*` and does not conflict with the I2C sensor bus.

## Repository files

Copy the files to these exact locations:

```text
ECE450_software/
├── config/
│   └── co5300_init.json
├── docs/
│   └── CO5300_RPI_QSPI.md
└── tools/
    ├── co5300_qspi_test.py
    └── run_co5300_qspi.sh
```

No `CMakeLists.txt` change is needed for this first hardware bring-up.

## Power-off wiring

Turn the Raspberry Pi off before changing wiring. Use BCM numbering only in software.

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
| 4 | TP_SCL | leave disconnected | — |
| 3 | TP_SDA | leave disconnected | — |
| 2 | TP_INT | leave disconnected | — |
| 1 | TP_RST | leave disconnected | — |

**Never connect display VCC to 5 V.** Verify the H1 pin-1 marker before applying power.

## Install dependency

```bash
sudo apt update
sudo apt install -y python3-lgpio
```

## Copy into the repository

From a folder containing the release package:

```bash
cp tools/co5300_qspi_test.py ~/ECE450_software/tools/
cp tools/run_co5300_qspi.sh ~/ECE450_software/tools/
cp config/co5300_init.json ~/ECE450_software/config/
cp docs/CO5300_RPI_QSPI.md ~/ECE450_software/docs/
chmod +x ~/ECE450_software/tools/co5300_qspi_test.py
chmod +x ~/ECE450_software/tools/run_co5300_qspi.sh
```

Adjust `~/ECE450_software` if your repository folder is named `ECE450_software-main`.

## Offline code test

```bash
cd ~/ECE450_software
python3 tools/co5300_qspi_test.py --self-test
```

Expected:

```text
[PASS] CO5300 offline self-test
```

## GPIO availability check

Stop any previous Python/C++ display test first. Check the GPIO controller:

```bash
ls -l /dev/gpiochip*
for f in /sys/class/gpio/gpiochip*/label; do echo "$f: $(cat "$f")"; done
```

The script automatically searches for the RP1 GPIO controller.

## Run the physical display test

```bash
cd ~/ECE450_software
sudo ./tools/run_co5300_qspi.sh
```

Equivalent explicit command:

```bash
sudo python3 tools/co5300_qspi_test.py \
  --init config/co5300_init.json \
  --gpiochip auto \
  --clk 21 --sio0 20 --sio1 19 --sio2 16 --sio3 26 \
  --cs 18 --rst 25 --te 24 \
  --half-period-us 2 \
  --chunk-bytes 2048 \
  --pattern bars \
  --hold
```

The program should:

1. reset the panel;
2. send the JSON initialization table;
3. report roughly 50–70 TE rising edges per second;
4. transfer a 410×502 RGB565 color-bar frame using opcode `0x32` and SIO0–SIO3;
5. keep the image visible until `Ctrl+C`.

## Other patterns

```bash
sudo python3 tools/co5300_qspi_test.py --pattern red --hold
sudo python3 tools/co5300_qspi_test.py --pattern green --hold
sudo python3 tools/co5300_qspi_test.py --pattern blue --hold
sudo python3 tools/co5300_qspi_test.py --pattern white --hold
sudo python3 tools/co5300_qspi_test.py --pattern checker --hold
```

If red and blue are reversed:

```bash
sudo python3 tools/co5300_qspi_test.py --pattern bars --bgr --hold
```

If the wiring is long or unstable, slow down the software clock:

```bash
sudo python3 tools/co5300_qspi_test.py \
  --half-period-us 5 --pattern bars --hold
```

## Interpreting results

- **TE ≈ 60 edges/s and color bars visible:** command and quad-pixel paths both work.
- **TE ≈ 60 edges/s but black screen:** GPIO command path works; check the complete supplier initialization table, panel power/current, FPC seating, and actual controller variant.
- **No TE:** check VCC/GND, RESET, CS, CLK, SIO0, H1 orientation, and the selected gpiochip.
- **`GPIO busy` or `GPIO not allocated`:** another process owns one of the selected GPIO lines. Stop it and rerun.

The initialization table is deliberately stored in `config/co5300_init.json`. When the supplier provides a complete panel-specific register table, replace or extend this JSON without rewriting the QSPI transport code.

# ECE450 TGS + SVM41 Pipeline v2

This isolated pipeline uses only the six ADS7828/TGS channels and the external
SVM41 over UART/SHDLC.  It does not initialize or use EC Sense NH3/H2S.

## Folder layout

```text
ECE450_software/
└── enose_tgs_svm41_v2/
    ├── tools/
    │   ├── collect_all_sensors.py
    │   └── enose_auto_classify_display.py
    ├── model/
    │   ├── food_freshness_multitask.py
    │   └── test_model.py
    ├── data/
    │   ├── raw/          # original enose_*.csv files
    │   ├── processed/    # cleaned rows and temporal windows
    │   └── session/      # baseline/test CSVs created by auto mode
    ├── models/
    │   └── food_freshness/  # joblib models, metadata, metrics, figures
    └── runtime/          # display state and runtime cache
```

Keep this directory inside the existing `ECE450_software` repository.  The
automatic script uses the parent repository's `tools/co5300_dashboard.py` and
`config/co5300_init.json` when display output is enabled.

## Cleaning rules

- Collection begins only after both `svm41_voc_index` and
  `svm41_nox_index` are greater than zero.  The default warm-up timeout is
  120 seconds.  Thus `--samples 90` records 90 post-warm-up rows.
- Training and testing apply the same rule again, so older CSV files containing
  the approximately 45-second zero period remain compatible.
- A TGS reading is saturated when its ADS7828 code is `<= 4` or `>= 4091`.
  Only that value is changed to missing; other normal values from the same TGS
  channel remain available.
- A TGS channel is removed from the model only when it has no unsaturated
  training values, or no unsaturated blank/baseline values.  The exact selected
  and excluded channels are stored in `training_metadata.json`.
- If a selected TGS is fully saturated in one later test session, that session
  treats the channel as unavailable while keeping the model input schema and
  all other sensors unchanged.

## Install dependencies on Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-smbus2 python3-serial
python3 -m pip install --user --break-system-packages \
  sensirion-uart-svm4x pandas numpy scipy scikit-learn joblib matplotlib
```

## Collect data

From `~/Documents/ECE450_software`:

```bash
python3 enose_tgs_svm41_v2/tools/collect_all_sensors.py \
  --food-group fermented_banana \
  --samples 90
```

The default output is:

```text
enose_tgs_svm41_v2/data/raw/enose_fermented_banana_1.csv
```

The collector supports `blank`, `fresh_banana`, `fermented_banana`,
`fresh_meat`, `spoiled_meat`, and custom labels.  It automatically chooses the
next unused numeric ID.  A filename can also be supplied explicitly:

```bash
python3 enose_tgs_svm41_v2/tools/collect_all_sensors.py \
  --filename enose_fresh_meat_2.csv \
  --food-group fresh_meat \
  --samples 90
```

## Train

```bash
python3 enose_tgs_svm41_v2/model/food_freshness_multitask.py
```

Training automatically scans
`enose_tgs_svm41_v2/data/raw/enose_*.csv`.  Labels are read from the CSV
`food_group` column first and from the filename second, so both
`enose_fresh_banana_1.csv` and timestamp filenames written by the collector are
supported.

## Test

```bash
python3 enose_tgs_svm41_v2/model/test_model.py \
  --baseline-csv enose_tgs_svm41_v2/data/raw/enose_blank_1.csv \
  --input-csv enose_tgs_svm41_v2/data/raw/enose_fresh_banana_1.csv
```

## Automatic baseline, sample, classification, and display

```bash
sudo python3 enose_tgs_svm41_v2/tools/enose_auto_classify_display.py
```

Run the full classification path without accessing the display:

```bash
python3 enose_tgs_svm41_v2/tools/enose_auto_classify_display.py \
  --baseline-csv enose_tgs_svm41_v2/data/raw/enose_blank_1.csv \
  --input-csv enose_tgs_svm41_v2/data/raw/enose_fresh_meat_1.csv \
  --no-display
```

Old schema-2 model files are intentionally rejected.  Retrain once with the
v2 training script before running test or auto mode.

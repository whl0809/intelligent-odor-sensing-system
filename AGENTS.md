# AGENTS.md

## Mission

Maintain the ECE450 food-type and freshness model training and evaluation
pipeline. This branch is model-first. The maintained Raspberry Pi sensor
collector lives on the separate acquisition branch.

## Supported model pipeline

The authoritative pair is:

```text
model/enose_multitask.py
model/test_model.py
```

`enose_multitask.py` trains the format-version 2 fused TGS + SVM41 model.
`test_model.py` validates that bundle contract and performs evaluation or
inference. Do not silently substitute the incompatible legacy trainers under
`extras/legacy_models/`.

The main training command is:

```bash
python model/enose_multitask.py train \
  --data-dir data/training \
  --output-dir model/artifacts
```

The main test command is:

```bash
python model/test_model.py \
  --input-csv data/test/test1.csv \
  --output-json results/test1_prediction.json
```

## Data contract

The active dataset uses ADS7828/TGS plus SVM41 fields:

```text
tgs2620_raw
tgs2610_raw
tgs2611_raw
tgs2600_raw
tgs2602_raw
tgs2603_raw
svm41_temperature_c
svm41_relative_humidity_pct
svm41_voc_index
svm41_nox_index
```

Use status and timing fields when present. Validate numeric data, remove the
SVM41 warm-up interval, and treat rail-saturated TGS values as unavailable.
Never call VOC Index or NOx Index ppm or concentration.

Training labels are inferred from filenames containing:

```text
blank
fresh_banana
fermented_banana
fresh_meat
spoiled_meat
```

Blank is baseline/reference data, not an output class.

## Validation rules

- Split and validate by complete recording, never by randomly mixing windows
  from the same CSV across train and validation sets.
- Keep preprocessing, selected channels, feature timing, scikit-learn version,
  and model format in the saved bundle.
- Report recording-level results and dataset limitations plainly.
- Do not describe confidence as calibrated probability unless calibration is
  explicitly demonstrated.
- The dataset is small; do not claim production classification performance.

## Repository layout

```text
AGENTS.md
README.md
pyproject.toml
model/
  enose_multitask.py
  test_model.py
  artifacts/
data/
  training/
  test/
pipeline/
  run_model_pipeline.py
  co5300_dashboard.py
  co5300_qspi_test.py
  config/
tests/
extras/
  legacy_models/
  legacy_acquisition/
```

Keep the main model path compact. Do not introduce a framework, service layer,
database, dashboard framework, or generic model hierarchy without a concrete
need.

## Acquisition boundary

Do not develop or validate new sensor acquisition behavior in this branch.
The old local acquisition package and scripts are archived under
`extras/legacy_acquisition/` only for reference.

`pipeline/run_model_pipeline.py` may classify an existing CSV directly. For
live collection it must use `--acquisition-root` to call a separate checkout
of the maintained acquisition branch with:

```bash
python -m enose acquire \
  --config config/rpi5.toml \
  --sensors tgs,svm41 \
  --uart /dev/ttyUSB0
```

Do not add the archived acquisition package back to the root Python path or
root dependency set.

## Dependencies and artifacts

- Use Python 3.11+.
- Keep the model runtime dependencies in root `pyproject.toml`.
- Pin scikit-learn to the version recorded by trusted model bundles when
  joblib compatibility requires it.
- Generated files below `model/artifacts/`, `results/`, and `runtime/` are not
  committed.
- Joblib uses pickle deserialization. Never load model artifacts from an
  untrusted source.

## Tests

Use focused tests for:

- expected training/test dataset inventory;
- one complete train-to-test run in a temporary directory;
- saved bundle format and prediction labels;
- delegation of live collection to the acquisition branch interface.

Tests must not write generated models or reports into the repository.

## Rules for changes

For every change:

1. Read this file and the relevant model/data documentation.
2. Preserve the format-version 2 trainer/test-runner compatibility.
3. Keep legacy acquisition and incompatible model experiments isolated.
4. Remove superseded paths and tests in the same change.
5. Avoid unrelated model changes during repository cleanup.
6. Report changed and removed files, tests run, and remaining model or hardware
   validation.

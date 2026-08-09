# ECE450 food-classification model

This branch is organized around training and testing the TGS + SVM41 food and
freshness classifier. The maintained sensor-acquisition application lives on
the separate acquisition branch; the old acquisition snapshot in this branch
is retained only under `extras/legacy_acquisition/`.

## What the model does

The active model uses recordings containing six ADS7828/TGS channels and the
SVM41 temperature, humidity, VOC Index, and NOx Index fields. Training:

1. removes invalid rows and the SVM41 warm-up period;
2. treats rail-saturated TGS readings as unavailable;
3. builds overlapping temporal windows;
4. validates by holding out complete recording files, not random rows;
5. trains banana-versus-meat, freshness-condition, and direct four-state
   classifiers; and
6. fuses the hierarchical and direct state probabilities.

Blank recordings are reference data, not an output class. Predictions are
model labels and confidence scores—not calibrated gas concentrations.

## Layout

```text
model/
  enose_multitask.py    active trainer and low-level predictor
  test_model.py         validated model-test wrapper
  artifacts/            generated model bundle and reports
data/
  training/             15 labeled training recordings
  test/                 three example test recordings
pipeline/               optional automated test and CO5300 display workflow
tests/                  focused train-to-test pipeline checks
extras/
  legacy_models/        incompatible earlier experiments
  legacy_acquisition/   outdated local acquisition snapshot
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The model bundle records scikit-learn 1.9.0, so the environment pins that
version to avoid incompatible joblib deserialization.

## Train

From the repository root:

```bash
python model/enose_multitask.py train \
  --data-dir data/training \
  --output-dir model/artifacts
```

This writes `model/artifacts/enose_multitask_model.joblib` plus validation
metadata and reports. Existing generated artifacts are ignored by Git.

## Test a recording

```bash
python model/test_model.py \
  --model model/artifacts/enose_multitask_model.joblib \
  --input-csv data/test/test1.csv \
  --output-json results/test1_prediction.json \
  --no-figures
```

Optional expected labels turn the command into a known-label check:

```bash
python model/test_model.py \
  --input-csv data/test/test1.csv \
  --expected-food meat \
  --expected-freshness fresh
```

See [model/README.md](model/README.md) for the model contract and inference
options.

## Automated pipeline and display

Classify an existing CSV without touching acquisition hardware or the display:

```bash
python pipeline/run_model_pipeline.py \
  --input-csv data/test/test1.csv \
  --no-display
```

For live acquisition, provide a separate checkout of the acquisition branch:

```bash
python pipeline/run_model_pipeline.py \
  --acquisition-root ../ECE450_acquisition \
  --uart /dev/ttyUSB0 \
  --frames 120 \
  --no-display
```

The pipeline calls that checkout's unified acquisition command with
`--sensors tgs,svm41`; it does not use the archived acquisition code in this
branch. Display setup is documented in [pipeline/README.md](pipeline/README.md).

## Validate the branch

```bash
python -m pytest
```

The focused integration test trains into a temporary directory and then runs
the matching test wrapper on an included recording. Current validation is
recording-level but still based on a small dataset, so reported accuracy and
confidence must not be treated as production performance.

# Active TGS + SVM41 model

`enose_multitask.py` and `test_model.py` are the authoritative, compatible
pair in this branch. The trainer writes a format-version 2 fused model bundle;
the test wrapper rejects bundles missing that contract.

## Inputs

Required sensor fields:

- `tgs2620_raw`, `tgs2610_raw`, `tgs2611_raw`, `tgs2600_raw`,
  `tgs2602_raw`, and `tgs2603_raw`
- `svm41_temperature_c` and `svm41_relative_humidity_pct`
- `svm41_voc_index` and `svm41_nox_index`

Status and timing columns are used when present. Training labels are inferred
from filenames containing `blank`, `fresh_banana`, `fermented_banana`,
`fresh_meat`, or `spoiled_meat`.

Default feature timing is a 45-frame minimum warm-up, 60-frame window, 20-frame
step, and 35-frame minimum partial window. The saved bundle records the exact
configuration and selected channels.

## Commands

```bash
python model/enose_multitask.py train \
  --data-dir data/training \
  --output-dir model/artifacts

python model/test_model.py \
  --input-csv data/test/test1.csv \
  --output-json results/test1_prediction.json
```

`test_model.py` defaults to `model/artifacts/enose_multitask_model.joblib`.
Use `--streaming-prefix` to aggregate all currently available windows during
live collection, or `--latest-window` to use only the newest window. A
separate baseline CSV is optional because the selected deployment model uses
absolute recording-level features.

## Limitations

Validation holds out complete CSV recordings, but the dataset is small and
contains few independent sessions for some classes. Collect more recordings
on different days before treating model metrics or confidence as reliable.


# E-nose food type and freshness model

This package trains a hierarchical classifier from labeled e-nose CSV recordings.

## Model design

1. Remove invalid acquisition rows.
2. Remove at least 45 SVM41 warm-up frames, extending the cutoff until VOC has positive readings in at least 3 of 5 consecutive frames.
3. Drop a TGS sensor only when it is rail-saturated in every training CSV. A sensor saturated in only some recordings remains usable; its window-level saturation fraction is included as a feature.
4. Convert each recording into overlapping 60-frame windows. Each channel contributes robust level, spread, slope, change, and relative-change features. Voltage columns are omitted because they duplicate the raw ADC values.
5. Use a soft-voting ensemble of class-balanced logistic regression and shrinkage LDA, both suited to small, correlated sensor datasets.
6. Predict `blank / banana / meat`, then predict `fresh / not_fresh`. Map `banana + not_fresh` to `fermented` and `meat + not_fresh` to `spoiled`.

Validation is leave-one-recording-out. Windows from the same CSV never appear on both sides of a validation fold, and predictions are scored once per held-out CSV.

The training command also generates `training_validation_learning_curves.png` and `learning_curve_points.csv`. These curves show recording-level training and validation accuracy as the number of independent training CSV files increases. Each plotted point is averaged over 30 random grouped splits, with one complete CSV per class reserved for validation in every repeat. The shaded bands show one standard deviation.

## Train

```bash
python enose_multitask.py train \
  --data-dir ./training_csv \
  --output-dir ./artifacts
```

Filenames must contain one of: `blank`, `fresh_banana`, `fermented_banana`, `fresh_meat`, or `spoiled_meat`.

## Predict

Without a separate blank/baseline recording:

```bash
python enose_multitask.py predict \
  --model ./artifacts/enose_multitask_model.joblib \
  --csv ./new_sample.csv
```

With a baseline recording collected immediately before the sample:

```bash
python enose_multitask.py predict \
  --model ./artifacts/enose_multitask_model.joblib \
  --baseline-csv ./current_blank.csv \
  --csv ./new_sample.csv
```

At least about 80 frames are required: approximately 45 warm-up frames plus 35 usable frames. Around 105 frames gives one full 60-frame post-warm-up window.

## Important limitation

There is only one spoiled-meat recording and two fermented-banana recordings. The food-type result is reasonably promising, but freshness performance is preliminary. Collect at least 5–10 independent recordings per food/freshness state, on different days and after complete chamber cleaning, before treating the model as reliable.

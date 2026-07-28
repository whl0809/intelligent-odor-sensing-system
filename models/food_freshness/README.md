# Food and freshness model artifacts

These artifacts were supplied for offline inference on CSV files produced by
`acquire-no-sgp41-bme690-sht45`.

The metadata selects seven voltage channels:

- TGS2620, TGS2610, TGS2611, TGS2602, and TGS2603
- NH3 and H2S differential voltage

TGS2600 was excluded during training because its recorded saturation rate was
100%. Each selected channel is normalized with the fixed baseline mean and
scale in `dataset_and_preprocessing_metadata.json`. The runner uses 20-row
windows with a 5-row stride and extracts 102 features: 11 statistics for each
channel, 21 pairwise correlations, three TGS-array statistics, and one
NH3-minus-H2S statistic.

Static inspection of the model packages identifies scikit-learn 1.9.0
pipelines containing median imputation, variance filtering, standard scaling,
and balanced logistic regression. Their output classes are:

- `food_type`: `banana`, `meat`
- `freshness`: `fermented`, `fresh`
- `combined_class`: `fermented_banana`, `fresh_banana`, `fresh_meat`

SHA-256 checksums of the supplied artifacts:

```text
freshness_best_model.joblib
EEC2EDF311A853994B8B47E56C4CFE47E4556B8D2D61FDC2156AD816FA1BB6CF

food_type_best_model.joblib
5F9C42BAF247D4BD4451EBFBC17805FB0B335B63BD88AE6DEC7900EB00CF1D6B

combined_class_best_model.joblib
C81B92D89224D1E3C77BAA110EFD5576953475CABC1646B03DF2CF99A6586886
```

Joblib artifacts use pickle deserialization. Do not replace or load these files
from an untrusted source.

For live use, `acquire-classify` loads these three artifacts once and predicts
from the latest 60 acquisition rows every 10 frames.

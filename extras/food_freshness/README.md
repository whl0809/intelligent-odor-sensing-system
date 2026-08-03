# Optional food/freshness pipeline

This directory keeps the experimental model pipeline separate from the core
sensor-acquisition code. It is preserved for possible future use but is not
required for `probe`, `diagnose`, or normal `acquire` operation.

- `artifacts/`: fixed preprocessing metadata and trusted joblib models
- `training_data/`: the supplied historical training CSV files
- `tools/`: offline classification and optional CO5300 display utilities
- `config/`: CO5300 initialization and example display state
- `docs/`: optional display wiring and validation notes

The live `acquire-classify` command remains available and loads the artifacts
from this directory. Install its optional dependencies with:

```bash
python -m pip install -e '.[classification]'
```

Offline classification from the repository root:

```bash
python extras/food_freshness/tools/classify_csv.py data/raw/<file>.csv
```

Only load the repository-provided joblib files; joblib uses Python pickle and
must not be used with untrusted artifacts.

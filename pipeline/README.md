# Optional model-test and display pipeline

The pipeline can test an existing CSV, or request a new TGS + SVM41 recording
from a separate checkout of the maintained acquisition branch.

Existing CSV, no GPIO access:

```bash
python pipeline/run_model_pipeline.py \
  --input-csv data/test/test1.csv \
  --no-display
```

Live collection through another checkout:

```bash
python pipeline/run_model_pipeline.py \
  --acquisition-root ../ECE450_acquisition \
  --uart /dev/ttyUSB0 \
  --frames 120
```

The external checkout must support:

```bash
python -m enose acquire \
  --config config/rpi5.toml \
  --sensors tgs,svm41 \
  --uart /dev/ttyUSB0
```

Preview or test the CO5300 software without model acquisition:

```bash
python -m pip install -e '.[display]'
python pipeline/co5300_dashboard.py --demo
python pipeline/co5300_qspi_test.py --self-test
```

Actual GPIO output also requires the Raspberry Pi `lgpio` package and the
wiring/init settings in `pipeline/config/`.


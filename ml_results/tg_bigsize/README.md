# Large-Data Tg Benchmark

This folder contains the final large-data glass-transition-temperature (`Tg`)
benchmark for Morgan fingerprint, p-ECFP, and BigFingerprint representations
using support vector regression (SVR), random forest regression (RF), and exact
Gaussian process regression (GPR).

## Data and evaluation protocol

- 7,170 valid, standard two-connection-point p-SMILES records were retained;
  the other four source rows were excluded.
- The stored `Tg` target and all plots use degrees Celsius.
- The observed target range is -139 to 495 degrees Celsius.
- The data are split into five fixed folders of 1,434 samples each using seed 42.
- Each reported test fold uses the other four folds (5,736 samples) for training.
- Hyperparameters were optimized without using Fold 1. The selected parameters
  were then held fixed while Fold 1 through Fold 5 were used as the test fold in
  turn.
- Reported standard deviations use the population definition (`ddof=0`).
- Fingerprint inputs were used exactly as generated, with no scaling, feature
  selection, variance filtering, PCA, or other preprocessing.

The fingerprint matrices have shapes `(7170, 2048)` for Morgan fingerprint and
p-ECFP and `(7170, 2088)` for BigFingerprint. The final 40 p-TFP dimensions of
BigFingerprint are constant across this linear-homopolymer data set, with the
constant vector `[1, 0, ..., 0]`.

## Five-fold test performance

| Model | Fingerprint | R2 mean +/- std | RMSE mean +/- std (degrees Celsius) |
|---|---|---:|---:|
| SVR | Morgan | 0.8994 +/- 0.0044 | 35.54 +/- 0.64 |
| SVR | p-ECFP | 0.8978 +/- 0.0055 | 35.81 +/- 0.85 |
| SVR | BigFingerprint | 0.8979 +/- 0.0056 | 35.80 +/- 0.85 |
| RF | Morgan | 0.8835 +/- 0.0084 | 38.23 +/- 1.52 |
| RF | p-ECFP | 0.8824 +/- 0.0076 | 38.42 +/- 1.43 |
| RF | BigFingerprint | 0.8835 +/- 0.0075 | 38.23 +/- 1.46 |
| GPR | Morgan | 0.8996 +/- 0.0053 | 35.50 +/- 0.92 |
| GPR | p-ECFP | 0.8981 +/- 0.0057 | 35.75 +/- 0.93 |
| GPR | BigFingerprint | 0.8981 +/- 0.0057 | 35.75 +/- 0.93 |

The full-precision values are in
[`results/fivefold_summary_all_models.csv`](results/fivefold_summary_all_models.csv).

## Contents

- `data/`: cleaned data, fixed split files, and the three generated fingerprint
  matrices.
- `models/`: six serialized Fold-1-held-out RF/SVR models. They were serialized
  with scikit-learn 1.6.1 and expect the unprocessed fingerprint vectors.
- `results/rf_svr/`: per-fold metrics and predictions for RF and SVR.
- `results/gpr/`: per-fold metrics, learned kernels, and predictions from the
  five exact-GPR SCC jobs.
- `results/fold1_predictions/`: saved train/test values used to create the nine
  manuscript-style Fold-1 panels.
- `figures/`: final PNG and editable SVG figures, including the shared axis
  panel with the degree-Celsius labels.
- `metadata/`: split, optimization, model, and verification settings.
- `scripts/`: data preparation, model optimization/evaluation, SCC GPR, and
  plotting scripts.

The exact GPR jobs did not serialize fitted GPR objects because each estimator
contains the full exact-kernel training state and is comparatively large. The
saved GPR predictions, fold metrics, learned kernels, and complete training
script are included instead. The six final RF/SVR estimators are provided as
trained model files.

## Regenerate the processed figures

From this folder, install the listed Python dependencies and run:

```bash
python scripts/plot_fold1.py
```

The script reads the verified prediction tables and rewrites the PNG and SVG
files in `figures/`. Times New Roman and a LaTeX installation are recommended
for an exact manuscript-style match.

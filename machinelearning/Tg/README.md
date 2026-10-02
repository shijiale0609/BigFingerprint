# Large-Data Tg Benchmark

This folder contains the final large-data glass-transition-temperature (`Tg`)
benchmark for Morgan fingerprint, p-ECFP, and BigFingerprint representations
using support vector regression (SVR), random forest regression (RF), and exact
Gaussian process regression (GPR).

## Data and evaluation protocol

- The analyzed data contain 7,157 standard two-connection-point linear
  homopolymers. Thirteen structures containing explicit deuterium (`[2H]`)
  were excluded from the 7,170 eligible source records before fingerprint
  generation and splitting.
- The stored target values and all plots use degrees Celsius. The observed
  target range is -139 to 495 degrees Celsius.
- The data were divided with seed 42 into five fixed folders containing 1,432,
  1,432, 1,431, 1,431, and 1,431 samples.
- Fold 1 was kept out of hyperparameter selection. Hyperparameters were
  optimized on Folds 2-5, then held fixed while each of Folds 1-5 served once
  as the held-out test fold.
- Reported standard deviations use the population definition (`ddof=0`).
- Fingerprint inputs were used exactly as generated, without scaling, feature
  selection, variance filtering, PCA, or other preprocessing.

## Fingerprints

| Representation | Structural input | Matrix shape |
|---|---|---:|
| Morgan fingerprint | canonical p-SMILES | `(7157, 2048)` |
| p-ECFP | BigSMILES | `(7157, 2048)` |
| p-TFP | BigSMILES | `(7157, 40)` |
| BigFingerprint | concatenated p-ECFP and p-TFP | `(7157, 2088)` |

The p-TFP contains four unique vectors. Dimension 1 is active for all samples.
Dimensions 20 and 21, which encode E/trans and Z/cis double-bond
stereochemistry, are active in 784 and 75 samples, respectively, with 20
samples containing both bits. The other 37 dimensions are constant at zero.
Thus, 839 samples (11.72%) contain at least one sample-dependent p-TFP bit.

## Five-fold test performance

| Model | Fingerprint | R2 mean +/- std | RMSE mean +/- std (degrees Celsius) |
|---|---|---:|---:|
| SVR | Morgan | 0.8971 +/- 0.0105 | 35.91 +/- 1.92 |
| SVR | p-ECFP | 0.8959 +/- 0.0103 | 36.12 +/- 1.88 |
| SVR | BigFingerprint | 0.8958 +/- 0.0101 | 36.15 +/- 1.85 |
| RF | Morgan | 0.8832 +/- 0.0130 | 38.25 +/- 2.15 |
| RF | p-ECFP | 0.8820 +/- 0.0112 | 38.47 +/- 1.89 |
| RF | BigFingerprint | 0.8814 +/- 0.0116 | 38.56 +/- 1.95 |
| GPR | Morgan | 0.8984 +/- 0.0118 | 35.68 +/- 2.15 |
| GPR | p-ECFP | 0.8963 +/- 0.0111 | 36.04 +/- 1.98 |
| GPR | BigFingerprint | 0.8964 +/- 0.0112 | 36.03 +/- 1.99 |

The full-precision values are stored in the model-specific summary CSV files
and in
[`results/fivefold_summary_all_models.csv`](results/fivefold_summary_all_models.csv).

## Contents

- `data/`: final processed data, fixed split files, and all four fingerprint
  matrices, including the separately stored 40-dimensional p-TFP.
- `results/models/`: six serialized Fold-1-held-out RF/SVR models. They were
  serialized with scikit-learn 1.9.0 and expect the unprocessed fingerprint
  vectors.
- `results/rf_svr/`: per-fold metrics and predictions for RF and SVR.
- `results/gpr/`: per-fold metrics, learned kernels, and predictions from the
  five exact-GPR SCC jobs.
- `results/fold1_predictions/`: train/test values used to create the nine
  manuscript-style Fold-1 parity plots.
- `results/figures/`: final PNG and editable SVG parity plots, the shared
  degree-Celsius axis panel, and p-TFP visualizations.
- `results/metadata/`: fingerprint provenance, split, optimization, model,
  p-TFP, and verification settings.
- `code/`: fingerprint generation, RF/SVR evaluation, SCC GPR, and plotting
  scripts used for this benchmark.

The exact GPR jobs did not serialize fitted estimator objects because each
estimator contains the full exact-kernel training state. The saved GPR
predictions, per-fold metrics, learned kernels, and complete training script
are included. The six final RF/SVR estimators are provided as trained model
files.

## Regenerate the processed figures

From the repository root, run:

```bash
python machinelearning/Tg/code/plot_fold1.py
python machinelearning/Tg/code/visualize_ptfp.py
```

Times New Roman and a LaTeX installation are recommended for an exact
manuscript-style match.

# Machine-learning results

This directory contains the code, processed data, trained models, numerical
results, and figures used for the manuscript property-prediction studies. The
installable BigFingerprint package does not include these research artifacts.

## Directory layout

```text
ml_results/
├── README.md
├── code/
│   ├── shared/             # parameterized EA/IP workflows
│   └── tg_bigsize/         # large-data Tg workflows
├── data/
│   └── tg_bigsize/         # final Tg table, embeddings, and splits
└── results/
    ├── ev/                 # EA/EV manuscript figures
    ├── ip/                 # IP manuscript figures
    └── tg_bigsize/         # Tg metrics, predictions, models, and figures
```

Only `tg_bigsize` is used for the manuscript Tg analysis. The earlier
small-data `tg` directory is not part of this repository.

## Large-data Tg benchmark

The Tg benchmark contains 7,157 standard two-connection-point linear
homopolymers. Thirteen structures containing explicit deuterium (`[2H]`) were
excluded before fingerprint generation and splitting. The target values and
figures use degrees Celsius.

The three input representations are:

| Representation | Structural input | Matrix shape |
|---|---|---:|
| Morgan fingerprint | canonical p-SMILES | `(7157, 2048)` |
| p-ECFP | BigSMILES | `(7157, 2048)` |
| p-TFP | BigSMILES | `(7157, 40)` |
| BigFingerprint | p-ECFP concatenated with p-TFP | `(7157, 2088)` |

The data were split with seed 42 into five fixed folds. Fold 1 was excluded
from hyperparameter selection; parameters were optimized on Folds 2–5 and then
held fixed while each fold served once as the test set. Reported standard
deviations use `ddof=0`, and the generated embeddings were used without
preprocessing.

Full-precision metrics are in
[`results/tg_bigsize/fivefold_summary_all_models.csv`](results/tg_bigsize/fivefold_summary_all_models.csv).
Detailed contents and performance values are documented in
[`results/tg_bigsize/README.md`](results/tg_bigsize/README.md).

## Code

The shared EA/IP scripts replace the earlier collection of nearly identical
files:

- `code/shared/ev_ip_svr_rf.py`: hyperparameter optimization and fixed-parameter
  five-fold RF/SVR evaluation.
- `code/shared/prepare_ev_ip_gpr_data.py`: generation of the corrected,
  fold-aligned 2,700-row GPR input bundle.
- `code/shared/ev_ip_gpr.py`: validation, one-fold exact-GPR execution, and
  five-fold summarization.

The Tg scripts reproduce fingerprint generation, five-fold RF/SVR evaluation,
exact GPR execution, and the saved figures. For example:

```bash
python ml_results/code/tg_bigsize/run_gpr_fold.py validate
python ml_results/code/tg_bigsize/plot_fold1.py
python ml_results/code/tg_bigsize/visualize_ptfp.py
```

The EA/IP scripts take explicit data and output paths. Example:

```bash
python ml_results/code/shared/ev_ip_svr_rf.py evaluate \
  --target ip --data-dir /path/to/vipea --output-dir /path/to/results
```

EA/IP source arrays are not duplicated in this repository. Their required
filenames are `dataset.csv`, `psmiles_mfp_arr.npy`, `EVIP_pecfp_arr.npy`, and
`EVIP_bigfp.npy`.

## Results

For Tg, `results/tg_bigsize/` includes five-fold metrics and predictions,
Fold-1 plotting data, six serialized Fold-1-held-out RF/SVR estimators,
manuscript figures, and provenance metadata. Exact-GPR estimator objects are
not serialized because they contain the full exact-kernel training state; the
saved predictions, metrics, learned kernels, and training code are retained.

The separate EV and IP SVG panels are retained as manuscript figure outputs,
not as duplicated executable code.

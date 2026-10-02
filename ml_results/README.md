# Machine-learning results

This directory contains the property-prediction artifacts used to compare the
Morgan fingerprint, p-ECFP, and BigFingerprint representations. Research data
and models are kept outside the installable Python wheel.

## Tg benchmark

- `tg_bigsize/` is the Tg analysis used in the manuscript. It contains the final
  7,157-polymer data set, generated embeddings, fixed five-fold splits,
  trained Fold-1 RF/SVR models, all RF/SVR/GPR predictions and metrics, and the
  processed figures.

The complete protocol and numerical results for this analysis are in
[`tg_bigsize/README.md`](tg_bigsize/README.md).

## Directory layout

```text
ml_results/
├── README.md
├── scripts/              # shared, parameterized EA/IP workflows
├── ev/                    # EA/EV manuscript prediction panels
├── ip/                    # IP manuscript prediction panels
└── tg_bigsize/            # current large-data Tg benchmark
```

Each prediction-plot directory contains separate panels for the three
fingerprints and three regressors because these are distinct manuscript panel
outputs, not duplicate source code.

## Consolidated EA/IP scripts

The earlier repository stored one nearly identical Python file for every
property, fingerprint, and model. Those copies have been replaced by three
shared scripts:

- `scripts/ev_ip_svr_rf.py` performs held-out-fold hyperparameter optimization
  or fixed-parameter five-fold SVR/RF evaluation for either EA or IP.
- `scripts/prepare_ev_ip_gpr_data.py` creates the corrected, fold-aligned
  2,700-row data bundle used by the SCC jobs.
- `scripts/ev_ip_gpr.py` validates the prepared bundle, runs one exact-GPR
  property/fold job, and summarizes the five completed folds.

The random 2,700-row subset uses Python random seed 0. It contains all nine
monomer-A identities, 100 randomly selected monomer-B identities, and all
three topologies. KFold uses five splits, shuffling, and random state 42. Fold
1 is excluded from hyperparameter selection; fixed parameters are then
evaluated with each of the five folds held out once. Reported standard
deviations use `ddof=0`.

The raw EA/IP input directory must contain:

```text
dataset.csv
psmiles_mfp_arr.npy
EVIP_pecfp_arr.npy
EVIP_bigfp.npy
```

These research scripts additionally require NumPy, pandas, scikit-learn, and
Matplotlib; those packages are intentionally not runtime dependencies of the
installable fingerprint library.

Example commands:

```bash
python ml_results/scripts/ev_ip_svr_rf.py optimize \
  --target ea --model rf --fingerprint bigfp \
  --data-dir /path/to/vipea --output-dir /path/to/results

python ml_results/scripts/ev_ip_svr_rf.py evaluate \
  --target ip --data-dir /path/to/vipea --output-dir /path/to/results

python ml_results/scripts/prepare_ev_ip_gpr_data.py \
  --raw-data-dir /path/to/vipea --output-dir /path/to/scc_bundle/data

python ml_results/scripts/ev_ip_gpr.py run \
  --target ea --fold 1 --data-dir /path/to/scc_bundle/data \
  --output-dir /path/to/scc_bundle/results
```

For a broader hyperparameter search, pass a JSON parameter grid with
`--grid`. For fixed-parameter evaluation, pass a nested JSON configuration
with `--parameters`; otherwise the manuscript RF/SVR settings embedded in the
script are used.

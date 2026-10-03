# IP benchmark

This folder contains the ionization-potential (`IP`; target key `ip`) benchmark
for Morgan fingerprint, p-ECFP, and BigFingerprint using SVR, RF, and exact
GPR.

## Contents

- `code/`: parameterized RF/SVR optimization and evaluation, selected-data
  preparation, and exact-GPR execution/summarization.
- `data/`: the 2,700 selected samples, three embedding matrices, fixed fold
  indices, and checksum manifest. `targets.csv` contains both EA and IP so
  that the common sample mapping can be checked directly.
- `results/rf_svr/`: five-fold metrics, summary, out-of-fold predictions,
  per-fold prediction archives, and run metadata.
- `results/gpr/ip/`: complete raw outputs, fold metrics, summary metrics, and
  out-of-fold predictions from all five SCC jobs.
- `results/figures/`: final manuscript SVG panels.

The 2,700 global sample indices are unique. Sampling used Python seed 0 and
five-fold splitting used shuffle=True with random state 42. Every fold contains
2,160 training and 540 test samples; reported standard deviations use
`ddof=0`.

Validate the included selected-data bundle from the repository root:

```bash
python machinelearning/IP/code/ea_ip_gpr.py validate \
  --data-dir machinelearning/IP/data
```

The RF/SVR script reproduces sampling from the original 42,966-row source and
therefore requires external `dataset.csv`, `psmiles_mfp_arr.npy`,
`EAIP_pecfp_arr.npy`, and `EAIP_bigfp.npy` files. The saved RF/SVR predictions
and metrics in this folder are complete and do not require those source arrays
for inspection.

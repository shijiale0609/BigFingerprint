# Machine-learning benchmarks

This directory is organized by predicted property. Each property folder keeps
its code, processed data, numerical results, and manuscript figures together.
These research artifacts are not included in the installable BigFingerprint
Python package.

```text
machinelearning/
├── EV/
│   ├── code/
│   ├── data/
│   └── results/
├── IP/
│   ├── code/
│   ├── data/
│   └── results/
└── Tg/
    ├── code/
    ├── data/
    └── results/
```

## EV and IP

EV/EA and IP use the same randomly selected 2,700 polymers and the same five
folds. Python random seed 0 selected 100 monomer-B identities across all nine
monomer-A identities and three topologies. Five-fold splitting used shuffling
and random state 42, giving 540 held-out samples per fold.

Each property folder contains:

- the three selected embedding matrices (Morgan, p-ECFP, and BigFingerprint),
  both targets, split indices, and a checksum manifest;
- complete five-fold RF/SVR metrics, out-of-fold predictions, and per-fold
  prediction archives;
- complete five-fold exact-GPR metrics, out-of-fold predictions, and raw
  per-fold outputs for all three fingerprints;
- the final manuscript SVG panels.

The EV and IP data directories intentionally contain the same embedding and
split files so that each property folder is self-contained. The parameterized
source files are also identical; the target is selected with `--target ea` or
`--target ip`.

## Tg

The Tg benchmark contains 7,157 linear homopolymers and compares Morgan
fingerprint, p-ECFP, and BigFingerprint representations using SVR, RF, and
exact GPR. See [`Tg/README.md`](Tg/README.md) for the full protocol and reported
performance.

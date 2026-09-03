# BigFingerprint

BigFingerprint generates polymer fingerprints from BigSMILES strings by combining:

- p-ECFP features from repeat-unit chemistry
- p-TFP features from polymer topology
- BigSMILES preprocessing helpers for explicit or missing `[H]` end groups

All source code is in the [`script/`](script/) folder.

## Installation

RDKit is required. The recommended setup is a conda environment because RDKit wheels can be platform- and Python-version-sensitive.

```bash
git clone git@github.com:shijiale0609/BigFingerprint.git
cd BigFingerprint

conda create -n bigfingerprint python=3.10 -y
conda activate bigfingerprint
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

If you prefer pip-only installation, install RDKit with the package/version that supports your Python version, then run:

```bash
pip install -r requirements.txt
```

## Run The Built-In Example

From the repository root:

```bash
cd script
python bigfingerprint.py
```

This prints example BigSMILES strings, their BigFingerprint vectors, and the vector shapes.

## Use BigFingerprint In Your Own Script

From the repository root, set `PYTHONPATH` to the `script/` folder:

```bash
PYTHONPATH=script python - <<'PY'
from bigfingerprint import generate_bigfingerprint_vector

bigsmiles = "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}"
fingerprint = generate_bigfingerprint_vector(bigsmiles)

print(fingerprint)
print(fingerprint.shape)
PY
```

## Use The BigSMILES Preprocessor

The preprocessor expands explicit or missing `[H]` end groups using compatible repeat-unit fragments.

```bash
PYTHONPATH=script python - <<'PY'
from bigfingerprint_preprocess import preprocess_bigsmiles

bigsmiles = "{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"
candidates = preprocess_bigsmiles(bigsmiles)

for candidate in candidates:
    print(candidate)
PY
```

Example output:

```text
CCO{[>][<]CCO[>], [<]CC(C)O[>][<]}CC
CC(C)O{[>][<]CCO[>], [<]CC(C)O[>][<]}CC
```

## Run Tests

```bash
cd script
python -m unittest test_bigfingerprint_preprocess.py
```

## ML Results

ML plotting scripts and prediction figures are collected in [`ml_results/`](ml_results/).

The available property folders are:

- [`ml_results/tg/`](ml_results/tg/): glass transition temperature prediction plots.
- [`ml_results/tg_bigsize/`](ml_results/tg_bigsize/): large-data Tg benchmark with 7,170 polymers, fixed five-fold results, trained RF/SVR models, exact-GPR outputs, and manuscript figures.
- [`ml_results/ev/`](ml_results/ev/): electron affinity / EV prediction plots using `EA vs SHE (eV)`.
- [`ml_results/ip/`](ml_results/ip/): ionization potential prediction plots using `IP vs SHE (eV)`.

Each property folder contains:

- `scripts/`: scripts used to generate the ML predictions and figures.
- `prediction_plots/`: SVG prediction plots for MorganFP, p-ECFP, and BigFP with SVR, Gaussian process, and random forest models.

## Repository Layout

```text
BigFingerprint/
├── README.md
├── requirements.txt
├── ml_results/
│   ├── tg/
│   ├── ev/
│   └── ip/
└── script/
    ├── bigfingerprint.py
    ├── bigfingerprint_preprocess.py
    ├── rdkit_repeat_orientation.py
    ├── polymer_ecfp.py
    ├── polymer_topfp.py
    └── ...
```

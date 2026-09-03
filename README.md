# BigFingerprint

BigFingerprint is an installable Python package for polymer informatics with
BigSMILES. It combines a 2,048-bit polymer extended-connectivity fingerprint
(p-ECFP) with a 40-element polymer topology fingerprint (p-TFP), and includes
BigSMARTS matching plus explicit or missing `[H]` end-group preprocessing.

## Installation

Install the package and its RDKit, NumPy, and NetworkX dependencies directly
from the repository:

```bash
python -m pip install .
```

For an editable development installation:

```bash
python -m pip install -e ".[test]"
pytest
```

The package can also be installed directly from GitHub:

```bash
python -m pip install "git+https://github.com/shijiale0609/BigFingerprint.git"
```

Python 3.9 or newer is required. If pip cannot find a compatible RDKit wheel
for a platform, create a conda environment with RDKit first and then install
BigFingerprint into that environment.

## Python API

### Generate one fingerprint

Use the low-level function when the BigSMILES string is already resolved and
does not need `[H]` expansion:

```python
import bigfingerprint as bfp

bigsmiles = "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}"  # closed terminals
fp = bfp.generate_bigfingerprint_vector(bigsmiles)

print(fp.shape)  # (2088,)
```

The default p-ECFP radius is 2 and the default p-ECFP bit-vector size is 2,048.
All examples in this README use these standard defaults, producing a
2,088-element BigFingerprint after concatenating the 40-element p-TFP vector.

### Preprocess `[H]` end groups and fingerprint all candidates

An open polymer end can expand to multiple chemically valid repeat-fragment
candidates. The high-level API preserves each candidate next to its vector
instead of silently selecting or averaging one:

```python
import bigfingerprint as bfp

result = bfp.generate_bigfingerprint_vectors(
    "[H]{[$][$]CC(C)[$][$]}[H]"
)

print(result.candidates)
print(result.vectors.shape)  # (4, 2088)

for candidate, vector in result:
    print(candidate, vector.shape)
```

If preprocessing produces exactly one candidate, `result.vector` returns its
one-dimensional vector. If several candidates exist, that property raises a
clear error so ambiguity cannot pass unnoticed.

The preprocessing stage can also be used independently:

```python
from bigfingerprint import (
    add_missing_h_endgroups,
    expand_h_endgroups,
    preprocess_bigsmiles,
)

candidates = preprocess_bigsmiles(
    "{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"
)
```

`preprocess_bigsmiles` first adds missing `[H]` markers at open stochastic
object ends, then expands every marker with compatible repeat-unit fragments.
Multiple sites form a Cartesian product. Closed `[]` terminals are left alone,
and nested BigSMILES text is preserved during directional expansion.

### Lower-level modules

The original algorithms remain available as package submodules:

```python
from bigfingerprint import polymer_ecfp, polymer_topfp, search_tools

graphs = search_tools.generate_NetworkX_graphs(bigsmiles)
_, clusters, _ = search_tools.identify_cycles(graphs)
pecfp = polymer_ecfp.ECFP(
    graphs,
    clusters,
    print_all=False,
    save_graph=False,
)
ptfp = polymer_topfp.bigsmiles_topology_vector(bigsmiles)
```

## Command line

Installation provides a `bigfingerprint` command:

```bash
bigfingerprint preprocess "[H]{[$][$]CC(C)[$][$]}[H]"
bigfingerprint fingerprint "{[][<]CC[>][]}" --output fingerprint.json
bigfingerprint --version
```

The `fingerprint` command preprocesses `[H]` by default and writes every
candidate, its on-bit indices, and its full vector as JSON. Pass
`--no-preprocess-h` to fingerprint the input verbatim.

## Optional diagnostics

Normal API calls do not print legacy debug traces or create files. To export
the historical atom-invariant spreadsheet or network graph, install the
optional dependencies and opt in explicitly:

```bash
python -m pip install ".[debug]"
```

```python
fp = bfp.generate_bigfingerprint_vector(
    bigsmiles,
    print_identifiers=True,
    save_graph=True,
    graph_name="_example",
    verbose=True,
)
```

This writes `Atom_Invariants.xls` and `network_graph_example.png` in the current
directory.

## Research results

Research artifacts remain available in [`ml_results/`](ml_results/) for Tg,
large-data Tg, EV, and IP prediction studies. Vector-similarity datasets,
figures, and reproduction scripts are in
[`vector_similarity_research/`](vector_similarity_research/). These research
artifacts stay in the repository but are excluded from the installable wheel.

## Repository layout

```text
BigFingerprint/
├── pyproject.toml       # package metadata and dependencies
├── README.md
├── LICENSE
├── script/              # installed as the `bigfingerprint` package
│   ├── __init__.py      # public API
│   ├── api.py           # [H]-aware high-level workflow
│   ├── bigfingerprint.py
│   ├── bigfingerprint_preprocess.py
│   ├── polymer_ecfp.py
│   ├── polymer_topfp.py
│   └── ...
├── tests/
└── ml_results/          # research outputs; excluded from wheels
```

## License

BigFingerprint is distributed under the GNU General Public License v3.0.

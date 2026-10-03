# BigFingerprint

BigFingerprint is an installable Python package for polymer informatics with
BigSMILES. It combines a 2,048-bit polymer extended-connectivity fingerprint
(p-ECFP) with a 40-element polymer topology fingerprint (p-TFP), and includes
BigSMARTS matching plus automatic explicit or missing `[H]` end-group
preprocessing. Its unified fingerprint API always returns one binary vector,
including when preprocessing produces multiple candidate structures.

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

### Generate one unified fingerprint

Use `generate_bigfingerprint_vector` for every BigSMILES input. The function
always runs `[H]` preprocessing internally, even when no expansion is needed:

```python
import bigfingerprint as bfp

bigsmiles = "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}"  # closed terminals
fp = bfp.generate_bigfingerprint_vector(
    bigsmiles,
    pecfp_iterations=2,
    fp_size=2048,
)

print(fp.shape)  # (2088,)
print(fp.dtype)  # uint8
```

The p-ECFP implementation accepts any non-negative integer radius; radius 2 is
the standard default, not a limitation of the algorithm. The default p-ECFP
bit-vector size is 2,048. All examples in this README use these defaults,
producing a 2,088-element binary BigFingerprint after concatenating the
40-element p-TFP vector.

### Automatic `[H]` preprocessing and candidate aggregation

An open polymer end can expand to multiple chemically valid repeat-fragment
candidates. The same function processes every candidate internally and returns
one unified vector:

```python
import bigfingerprint as bfp

bigsmiles = "{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"
fp = bfp.generate_bigfingerprint_vector(
    bigsmiles,
    pecfp_iterations=2,
    fp_size=2048,
)

print(fp.shape)  # (2088,)
print(fp.dtype)  # uint8
```

For one candidate, its p-ECFP is used directly. For multiple candidates, their
p-ECFP vectors are combined with a bitwise OR. The p-TFP is calculated once
from the original BigSMILES and appended to the aggregated p-ECFP. Therefore,
inputs with one or many candidates have the same output format: one binary
`numpy.ndarray` with shape `(fp_size + 40,)`.

The preprocessing stage remains available independently when candidate strings
need to be inspected:

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
    iteration_input=2,
    fp_size=2048,
    print_all=False,
    save_graph=False,
)
ptfp = polymer_topfp.bigsmiles_topology_vector(bigsmiles)
```

## Command line

Installation provides a `bigfingerprint` command:

```bash
bigfingerprint preprocess "[H]{[$][$]CC(C)[$][$]}[H]"
bigfingerprint fingerprint "{[][<]CC[>][]}" \
  --radius 2 \
  --fp-size 2048 \
  --output fingerprint.json
bigfingerprint --version
```

The `fingerprint` command always preprocesses `[H]` internally. Its JSON output
contains the one aggregated vector and its on-bit indices, regardless of how
many candidate structures preprocessing generates.

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
    pecfp_iterations=2,
    fp_size=2048,
    print_identifiers=True,
    save_graph=True,
    graph_name="_example",
    verbose=True,
)
```

This writes `Atom_Invariants.xls` and `network_graph_example.png` in the current
directory.

## Research results

Research artifacts remain available in [`machinelearning/`](machinelearning/)
for the manuscript's EA, IP, and large-data Tg prediction studies.
Vector-similarity datasets, figures, and reproduction scripts are in
[`vector_similarity_search/`](vector_similarity_search/). These research
artifacts stay in the repository but are excluded from the installable wheel.

## Repository layout

```text
BigFingerprint/
├── pyproject.toml       # package metadata and dependencies
├── README.md
├── LICENSE
├── script/              # installed as the `bigfingerprint` package
│   ├── __init__.py      # public API
│   ├── bigfingerprint.py
│   ├── bigfingerprint_preprocess.py
│   ├── polymer_ecfp.py
│   ├── polymer_topfp.py
│   └── ...
├── tests/
├── machinelearning/     # EA, IP, and Tg code, data, and results
└── vector_similarity_search/
                         # vector-search data, figures, and scripts
```

## License

BigFingerprint is distributed under the GNU General Public License v3.0.

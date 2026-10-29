# BigFingerprint: A Topology-aware Extended-connectivity Fingerprint for Polymers

[Jiale Shi](https://www.utoledo.edu/engineering/chemical-engineering/facultystaff/jiale-shi/),
[Nathan J. Rebello](https://www.linkedin.com/in/nathanrebello/),
[Jiarui Lu](https://www.linkedin.com/in/sabrinaluljr/),
[Aaron A. Burkey](https://www.nist.gov/people/aaron-burkey),
[Debra J. Audus](https://www.nist.gov/people/debra-audus),
[Bradley D. Olsen](https://cheme.mit.edu/profile/bradley-d-olsen/)

This repository accompanies the manuscript *BigFingerprint: A Topology-aware
Extended-connectivity Fingerprint for Polymers* and contains the BigFingerprint
implementation, processed datasets, machine-learning workflows and results,
and vector similarity search case studies.

BigFingerprint is a topology-aware extended-connectivity fingerprint that uses
BigSMILES, an expressive line notation developed explicitly for polymers by
the [Olsen Lab at MIT](https://olsenlab.mit.edu/)
([BigSMILES documentation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html#examples);
[original publication](https://doi.org/10.1021/acscentsci.9b00476)).
BigFingerprint consists of two components: a polymer extended-connectivity
fingerprint (p-ECFP), which encodes atom-centered substructures within and
across molecular fragments such as repeat units, and a polymer topology
fingerprint (p-TFP), which encodes global architectural features such as block,
statistical, graft, star, branched, and network topologies.

The utility of BigFingerprint is demonstrated through three use cases:
topology-aware vector similarity search that produces intuitive results;
predictive performance comparable to pSMILES-based Morgan fingerprints for
the glass transition temperatures of linear homopolymers; and improved
prediction of the electron affinities and ionization potentials of copolymers.

As a critical component of the BigSMILES-centered ecosystem, BigFingerprint
provides a general and chemically interpretable representation for
substructure analysis, similarity assessment, machine-learning-based property
prediction, and inverse design of structurally diverse polymers.

## Development and updates

For ongoing development and the latest updates, please visit the
[BigFingerprint development repository](https://github.com/shijiale0609/BigFingerprint),
maintained by [Jiale Shi](https://github.com/shijiale0609).

Bug reports and feature requests are welcome in the development repository's
[issue tracker](https://github.com/shijiale0609/BigFingerprint/issues).

## BigFingerprint Python package

The repository provides BigFingerprint as an installable Python package for
polymer informatics. By default, the package combines a 2,048-bit p-ECFP with a
40-element p-TFP; the p-ECFP bit-vector size is configurable. The package also
includes BigSMARTS matching plus automatic preprocessing of explicit or
missing `[H]` end groups. Its unified fingerprint API always returns one binary
vector, including when preprocessing produces multiple candidate structures.

### Installation

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

### Python API

#### Generate one unified fingerprint

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

The p-ECFP implementation supports radius values of 0, 1, or 2, with 2 as the
default. Here, `radius` is the number of fingerprint-expansion
(neighborhood-update) iterations after atom initialization, following the
same parameter convention as
[RDKit's Morgan fingerprints](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator).
Radius 0 uses only the initial atom identifiers; radius 2 performs two update
iterations. Initialization is not counted as an update iteration.

The Python parameters `pecfp_iterations` and `iteration_input` specify this
same radius value. For the Morgan baseline, `MORGAN_RADIUS` and the metadata
field `morgan_radius` retain RDKit's radius terminology and likewise specify
the number of update iterations.

Based on the previous work on polymer fingerprints (e.g., Morgan Fingerprint
of pSMILES), radius 2 provides sufficient substructural information.
The current implementation is also limited to radius 2 to avoid incompatible
across-descriptor connection histories;
future work will develop algorithms that maintain connection-history
consistency at higher radii.

The default p-ECFP bit-vector size is 2,048. All examples in this README use
these defaults, producing a 2,088-element binary BigFingerprint after
concatenating the 40-element p-TFP vector.

#### Automatic `[H]` preprocessing and candidate aggregation

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

#### Lower-level modules

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

### Command line

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
many candidate structures preprocessing generates. The `--radius` option maps
to the Python API's `pecfp_iterations` parameter and accepts only `0`, `1`, or
`2` (default: `2`). It sets the number of update iterations after atom
initialization, using the radius convention described above.

### Optional diagnostics

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
Vector similarity search datasets, figures, and reproduction scripts are in
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
                         # vector similarity search data, figures, and scripts
```

## Contact

[Jiale Shi, PhD](https://www.utoledo.edu/engineering/chemical-engineering/facultystaff/jiale-shi/)

Assistant Professor

Department of Chemical Engineering

The University of Toledo (UToledo)

Email: [jiale.shi@utoledo.edu](mailto:jiale.shi@utoledo.edu)

GitHub: [shijiale0609](https://github.com/shijiale0609)

## License

BigFingerprint is distributed under the GNU General Public License v3.0.

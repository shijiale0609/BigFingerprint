# Vector similarity search

This folder contains the data, similarity matrices, and figures used for the
BigFingerprint vector similarity search case studies.

## Contents

### `eight_polymer_case/`

The controlled comparison contains eight polymers (P1-P8) spanning linear,
random, block, three-arm star, and four-arm star representations. The folder
includes:

- `morgan_tanimoto.csv`: 8 x 8 Morgan-fingerprint similarity matrix. Star
  polymers that cannot be represented by the pSMILES-based Morgan baseline are
  recorded as unavailable.
- `pecfp_tanimoto.csv`: 8 x 8 p-ECFP similarity matrix.
- `ptfp_tanimoto.csv`: 8 x 8 p-TFP similarity matrix.
- `bigfingerprint_product_similarity.csv`: 8 x 8 BigFingerprint similarity
  matrix, calculated elementwise as p-ECFP similarity multiplied by p-TFP
  similarity.
- `polymer_definitions_and_bit_counts.csv` and
  `blue_individual_heatmaps/p1_p8_mapping.csv`: polymer definitions and P1-P8
  mapping.
- `blue_individual_heatmaps/`: publication-ready individual heatmaps in PDF,
  PNG, and SVG formats.

CSV files are the human-readable primary data. Matching NumPy files are
included for direct computational reuse.

### `large_database_402/`

This curated database contains 402 deduplicated polymer BigSMILES extracted
from the query-target dataset reported in the BigSMARTS study. It spans
multiple repeat-unit chemistries and polymer topologies. Polymer IDs are
preserved from the source mapping for traceability and therefore are not
necessarily contiguous.

The complete pairwise matrices are:

- `pecfp_tanimoto_similarity_402.csv`
- `ptfp_tanimoto_similarity_402.csv`
- `bigfingerprint_product_similarity_402.csv`

Each matrix is 402 x 402, symmetric, bounded between 0 and 1, and has a unit
diagonal. BigFingerprint similarity is the exact elementwise product of the
p-ECFP and p-TFP Tanimoto similarities. Matching NumPy arrays are also
provided.

Additional files include:

- `polymer_id_mapping_402.csv`: polymer IDs, BigSMILES, and source metadata.
- `pecfp_fingerprints.*` and `ptfp_fingerprints.*`: fingerprint arrays.
- `query_selection_candidates_402.json` and
  `query_selection_summary_402.csv`: BigFingerprint-ranked search results.
- `ranked_query_figure/`: ranked-query figures and manuscript figure.
- `metadata.json`, `validation_summary.json`, and removal/audit tables: data
  lineage and validation records.

The topology labels in `query_selection_candidates_402.json` contain 318
single, 63 block, 3 segmented, 6 graft, 5 star, and 7 network entries.

### `scripts/`

Analysis scripts used to generate, correct, deduplicate, filter, rank, and plot
the similarity results are retained for reproducibility. Some filenames
contain the size of an intermediate dataset because the final 402-entry data
set was produced through a sequence of audited filtering steps.

## Similarity definition

For polymers *i* and *j*:

```text
S_BigFingerprint(i,j) = S_p-ECFP(i,j) * S_p-TFP(i,j)
```

This construction requires agreement in both local chemical environments and
global polymer topology.

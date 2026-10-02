#!/usr/bin/env python3
"""Remove complete Advanced/Extras case members from the corrected 475 dataset.

This expands the earlier header-only filter in two ways:

1. Advanced membership includes the contiguous Advanced target block in the
   original workbook (Excel rows 448--489) plus later Advanced-topology targets
   at rows 492 and 496.
2. Extras membership includes both exact query-header targets and target rows
   explicitly referenced in the Extras row-4 case labels (for example,
   ``432, 20`` identifies the P424 negative-control target).

Original polymer IDs are preserved. Similarity matrices are exact row/column
subsets of the corrected, verified-deduplicated 475-entry matrices.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "results_475_endgroup_corrected_deduplicated"
HEADER_CSV = (
    HERE
    / "results_436_endgroup_corrected_deduplicated_no_advanced_extras"
    / "advanced_extras_query_headers.csv"
)
SOURCE_WORKBOOK = HERE.parent / "BigSMARTS" / "query-target-dataset.xlsx"

# The original target catalog's Advanced cases form one contiguous block. Two
# later polyelectrolyte targets extend that block after newer literature entries
# were appended to the workbook.
ADVANCED_SOURCE_ROWS = set(range(448, 490)) | {492, 496}


def write_similarity_csv(matrix: np.ndarray, ids: list[str], path: Path) -> None:
    pd.DataFrame(matrix, index=ids, columns=ids).to_csv(
        path, index_label="polymer_id", float_format="%.8f"
    )


def write_fingerprint_csv(
    array: np.ndarray, ids: list[str], prefix: str, path: Path
) -> None:
    columns = [f"{prefix}_{index}" for index in range(1, array.shape[1] + 1)]
    frame = pd.DataFrame(array.astype(np.uint8), columns=columns)
    frame.insert(0, "polymer_id", ids)
    frame.to_csv(path, index=False)


def validate_similarity(matrix: np.ndarray, name: str) -> dict:
    symmetric = bool(np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12))
    diagonal_one = bool(
        np.allclose(np.diag(matrix), np.ones(matrix.shape[0]), rtol=0.0, atol=1e-12)
    )
    bounded = bool(np.all((matrix >= -1e-12) & (matrix <= 1.0 + 1e-12)))
    if not (symmetric and diagonal_one and bounded):
        raise ValueError(
            f"{name} failed validation: symmetric={symmetric}, "
            f"diagonal_one={diagonal_one}, bounded={bounded}"
        )
    return {
        "shape": list(matrix.shape),
        "symmetric": symmetric,
        "diagonal_one": diagonal_one,
        "bounded_0_1": bounded,
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
    }


def join_unique(values: list[str]) -> str:
    return " | ".join(dict.fromkeys(value for value in values if value))


def main() -> None:
    mapping = pd.read_csv(SOURCE_DIR / "polymer_id_mapping_475.csv").fillna("")
    if len(mapping) != 475 or not mapping["BigSMILES"].is_unique:
        raise ValueError("Expected a 475-row mapping with unique BigSMILES")

    headers = pd.read_csv(HEADER_CSV).fillna("")
    extras_headers = headers.loc[headers["source_sheet"] == "Extras"].copy()
    if len(extras_headers) != 200:
        raise ValueError("Expected 200 Extras query headers")

    extras_exact_evidence: dict[str, list[str]] = defaultdict(list)
    extras_reference_evidence: dict[int, list[str]] = defaultdict(list)
    bigsmiles_to_id = mapping.set_index("BigSMILES")["polymer_id"].to_dict()

    for row in extras_headers.itertuples(index=False):
        exact_id = bigsmiles_to_id.get(row.query_bigsmarts)
        if exact_id:
            extras_exact_evidence[exact_id].append(f"{row.source_cell}: {row.rule}")

        reference_match = re.match(r"^(\d+)\s*,", str(row.case_topology).strip())
        if reference_match:
            source_excel_row = int(reference_match.group(1))
            extras_reference_evidence[source_excel_row].append(
                f"{row.source_cell}: {row.rule}; case={row.case_topology}"
            )

    audit_rows = []
    for record in mapping.to_dict("records"):
        polymer_id = record["polymer_id"]
        source_excel_row = int(record["source_excel_row"])
        advanced_member = source_excel_row in ADVANCED_SOURCE_ROWS
        extras_exact = polymer_id in extras_exact_evidence
        extras_reference = source_excel_row in extras_reference_evidence
        if not (advanced_member or extras_exact or extras_reference):
            continue

        membership = []
        if advanced_member:
            membership.append("Advanced")
        if extras_exact or extras_reference:
            membership.append("Extras")
        audit_rows.append(
            {
                "polymer_id": polymer_id,
                "case_membership": "+".join(membership),
                "advanced_target_member": advanced_member,
                "extras_exact_query_target": extras_exact,
                "extras_explicit_target_reference": extras_reference,
                "extras_exact_evidence": join_unique(
                    extras_exact_evidence.get(polymer_id, [])
                ),
                "extras_reference_evidence": join_unique(
                    extras_reference_evidence.get(source_excel_row, [])
                ),
                **record,
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values("deduplicated_position")
    removed_ids = set(audit["polymer_id"])
    keep_mask = ~mapping["polymer_id"].isin(removed_ids)
    keep_indices = np.flatnonzero(keep_mask.to_numpy())
    filtered_count = int(keep_mask.sum())
    if len(removed_ids) != 73 or filtered_count != 402:
        raise ValueError(
            f"Expected 73 complete case members and 402 survivors; got "
            f"{len(removed_ids)} and {filtered_count}"
        )

    output_dir = (
        HERE / f"results_{filtered_count}_endgroup_corrected_deduplicated_"
        "no_complete_advanced_extras_cases"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_mapping = mapping.loc[keep_mask].copy().reset_index(drop=True)
    filtered_mapping.insert(0, "filtered_position", np.arange(1, filtered_count + 1))
    filtered_ids = filtered_mapping["polymer_id"].tolist()
    filtered_mapping.to_csv(
        output_dir / f"polymer_id_mapping_{filtered_count}.csv", index=False
    )

    pecfp = np.load(SOURCE_DIR / "pecfp_fingerprints.npy")
    ptfp = np.load(SOURCE_DIR / "ptfp_fingerprints.npy")
    filtered_pecfp = pecfp[keep_indices]
    filtered_ptfp = ptfp[keep_indices]
    np.save(output_dir / "pecfp_fingerprints.npy", filtered_pecfp)
    np.save(output_dir / "ptfp_fingerprints.npy", filtered_ptfp)
    write_fingerprint_csv(
        filtered_pecfp,
        filtered_ids,
        "pECFP",
        output_dir / "pecfp_fingerprints.csv",
    )
    write_fingerprint_csv(
        filtered_ptfp,
        filtered_ids,
        "pTFP",
        output_dir / "ptfp_fingerprints.csv",
    )

    matrix_specs = {
        "p-ECFP": "pecfp_tanimoto_similarity",
        "p-TFP": "ptfp_tanimoto_similarity",
        "BigFingerprint": "bigfingerprint_product_similarity",
    }
    filtered_matrices = {}
    validation = {}
    for label, stem in matrix_specs.items():
        source_matrix = np.load(SOURCE_DIR / f"{stem}_475.npy")
        filtered_matrix = source_matrix[np.ix_(keep_indices, keep_indices)]
        filtered_matrices[label] = filtered_matrix
        validation[label] = validate_similarity(filtered_matrix, label)
        np.save(output_dir / f"{stem}_{filtered_count}.npy", filtered_matrix)
        write_similarity_csv(
            filtered_matrix,
            filtered_ids,
            output_dir / f"{stem}_{filtered_count}.csv",
        )

    product_matches = bool(
        np.allclose(
            filtered_matrices["BigFingerprint"],
            filtered_matrices["p-ECFP"] * filtered_matrices["p-TFP"],
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not product_matches:
        raise ValueError("Filtered BigFingerprint matrix is not p-ECFP × p-TFP")

    audit.to_csv(
        output_dir / "removed_complete_advanced_extras_case_members.csv",
        index=False,
    )
    extras_headers.to_csv(output_dir / "extras_query_headers.csv", index=False)

    advanced_ids = set(audit.loc[audit["advanced_target_member"], "polymer_id"])
    extras_ids = set(
        audit.loc[
            audit["extras_exact_query_target"]
            | audit["extras_explicit_target_reference"],
            "polymer_id",
        ]
    )
    metadata = {
        "dataset": (
            "End-group-corrected, verified-deduplicated polymer dataset after "
            "excluding complete Advanced and Extras case membership"
        ),
        "source_directory": str(SOURCE_DIR),
        "source_workbook": str(SOURCE_WORKBOOK),
        "source_polymer_count": int(len(mapping)),
        "filtered_polymer_count": filtered_count,
        "advanced_membership_rule": {
            "contiguous_source_excel_rows": "448--489 inclusive",
            "later_advanced_source_excel_rows": [492, 496],
        },
        "extras_membership_rule": (
            "Union of exact Extras query-header BigSMILES targets and target rows "
            "explicitly referenced by '<row>, <case>' labels in Extras row 4"
        ),
        "advanced_member_count": len(advanced_ids),
        "extras_member_count": len(extras_ids),
        "advanced_extras_overlap_count": len(advanced_ids & extras_ids),
        "removed_polymer_count": len(removed_ids),
        "removed_ids": sorted(removed_ids, key=lambda value: int(value[1:])),
        "preserved_polymer_ids": True,
        "matrix_operation": "Exact row/column subset of corrected 475 matrices",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    validation_summary = {
        "source_count": int(len(mapping)),
        "advanced_member_count": len(advanced_ids),
        "extras_member_count": len(extras_ids),
        "advanced_extras_overlap_count": len(advanced_ids & extras_ids),
        "removed_count": len(removed_ids),
        "filtered_count": filtered_count,
        "p424_removed": "P424" in removed_ids,
        "mapping_bigsmiles_unique": bool(filtered_mapping["BigSMILES"].is_unique),
        "mapping_polymer_ids_unique": bool(filtered_mapping["polymer_id"].is_unique),
        "no_removed_ids_remain": not bool(set(filtered_ids) & removed_ids),
        "fingerprint_shapes": {
            "p-ECFP": list(filtered_pecfp.shape),
            "p-TFP": list(filtered_ptfp.shape),
        },
        "similarity_matrices": validation,
        "bigfingerprint_exact_product": product_matches,
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(validation_summary, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "source_count": len(mapping),
                "advanced_member_count": len(advanced_ids),
                "extras_member_count": len(extras_ids),
                "advanced_extras_overlap_count": len(advanced_ids & extras_ids),
                "removed_count": len(removed_ids),
                "filtered_count": filtered_count,
                "p424_removed": "P424" in removed_ids,
                "validation": validation_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

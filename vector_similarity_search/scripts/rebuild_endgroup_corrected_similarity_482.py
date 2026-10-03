#!/usr/bin/env python3
"""Rebuild the 482-polymer matrices with H-end-group-aware p-ECFP.

For every BigSMILES, missing or explicit [H] end groups are expanded into all
compatible repeat-unit attachments.  p-ECFP is calculated for every expanded
case and the binary vectors are combined by bitwise OR.  p-TFP remains defined
on the original BigSMILES, and BigFingerprint similarity is the product of the
two binary Tanimoto similarities.

The original results_482 directory is read-only input.  Corrected outputs are
written to results_482_endgroup_corrected.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from bigfingerprint import add_missing_h_endgroups, preprocess_bigsmiles

from compute_similarity_matrices_489 import (
    PECFP_FAST_CLUSTER_TARGET_INDICES,
    compute_one_pecfp,
)

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "results_482"
OUTPUT_DIR = HERE / "results_482_endgroup_corrected"

EXPECTED_POLYMERS = 482
PECFP_SIZE = 2048
PTFP_SIZE = 40


def tanimoto_matrix(fingerprints: np.ndarray) -> np.ndarray:
    binary = np.asarray(fingerprints, dtype=bool)
    intersections = binary.astype(np.uint16) @ binary.T.astype(np.uint16)
    counts = binary.sum(axis=1, dtype=np.uint16)
    unions = counts[:, None] + counts[None, :] - intersections
    return np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=np.float64),
        where=unions != 0,
    )


def save_matrix(matrix: np.ndarray, stem: str, ids: list[str]) -> None:
    frame = pd.DataFrame(matrix, index=ids, columns=ids)
    frame.index.name = "polymer_id"
    frame.to_csv(OUTPUT_DIR / f"{stem}.csv", float_format="%.8f")
    np.save(OUTPUT_DIR / f"{stem}.npy", matrix)


def validate_similarity_matrix(matrix: np.ndarray, label: str) -> dict:
    expected_shape = (EXPECTED_POLYMERS, EXPECTED_POLYMERS)
    if matrix.shape != expected_shape:
        raise ValueError(f"{label} has shape {matrix.shape}; expected {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    diagonal_error = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if symmetry_error > 1e-12 or diagonal_error > 1e-12:
        raise ValueError(
            f"{label} validation failed: symmetry={symmetry_error}, "
            f"diagonal={diagonal_error}"
        )
    if matrix.min() < 0 or matrix.max() > 1:
        raise ValueError(f"{label} contains values outside [0, 1]")
    return {
        "shape": list(matrix.shape),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "symmetry_max_abs_error": symmetry_error,
        "diagonal_max_abs_error": diagonal_error,
    }


def main() -> None:
    started = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(SOURCE_DIR / "polymer_id_mapping_482.csv")
    source_pecfp = np.load(SOURCE_DIR / "pecfp_fingerprints.npy")
    ptfp = np.load(SOURCE_DIR / "ptfp_fingerprints.npy")

    if len(mapping) != EXPECTED_POLYMERS:
        raise ValueError(
            f"Expected {EXPECTED_POLYMERS} mapping rows; found {len(mapping)}"
        )
    if source_pecfp.shape != (EXPECTED_POLYMERS, PECFP_SIZE):
        raise ValueError(f"Unexpected source p-ECFP shape: {source_pecfp.shape}")
    if ptfp.shape != (EXPECTED_POLYMERS, PTFP_SIZE):
        raise ValueError(f"Unexpected p-TFP shape: {ptfp.shape}")

    pecfp = (source_pecfp != 0).astype(np.uint8)
    ptfp = (ptfp != 0).astype(np.uint8)
    audit_records = []
    changed_count = 0

    for row_index, row in mapping.iterrows():
        polymer_id = row["polymer_id"]
        bigsmiles = row["BigSMILES"]
        with_h = add_missing_h_endgroups(bigsmiles)
        expanded = preprocess_bigsmiles(bigsmiles)
        changed = expanded != [bigsmiles]

        if changed:
            candidate_vectors = []
            use_fast_clusters = (
                int(row["original_target_index"]) in PECFP_FAST_CLUSTER_TARGET_INDICES
            )
            for candidate in expanded:
                candidate_vectors.append(
                    compute_one_pecfp(
                        candidate,
                        use_fast_clusters=use_fast_clusters,
                    )
                )
            pecfp[row_index] = np.bitwise_or.reduce(candidate_vectors)
            changed_count += 1
            source = "H-end-group-expanded candidate p-ECFP bitwise union"
        else:
            candidate_vectors = [pecfp[row_index]]
            source = "unchanged standard repository calculation"

        audit_records.append(
            {
                "dataset_position": int(row["dataset_position"]),
                "polymer_id": polymer_id,
                "original_target_index": int(row["original_target_index"]),
                "BigSMILES": bigsmiles,
                "explicit_H_present": "[H]" in bigsmiles,
                "missing_H_added": with_h != bigsmiles,
                "pECFP_recomputed": changed,
                "expanded_case_count": len(expanded),
                "candidate_active_bits": json.dumps(
                    [int(vector.sum()) for vector in candidate_vectors]
                ),
                "union_active_bits": int(pecfp[row_index].sum()),
                "expanded_BigSMILES_json": json.dumps(expanded),
            }
        )
        if changed:
            print(
                f"{polymer_id}: {len(expanded)} case(s), "
                f"union bits={int(pecfp[row_index].sum())}",
                flush=True,
            )

    ids = mapping["polymer_id"].tolist()
    pecfp_similarity = tanimoto_matrix(pecfp)
    ptfp_similarity = tanimoto_matrix(ptfp)
    bigfingerprint_similarity = pecfp_similarity * ptfp_similarity

    validation = {
        "p-ECFP": validate_similarity_matrix(pecfp_similarity, "p-ECFP"),
        "p-TFP": validate_similarity_matrix(ptfp_similarity, "p-TFP"),
        "BigFingerprint": validate_similarity_matrix(
            bigfingerprint_similarity, "BigFingerprint"
        ),
        "bigfingerprint_product_max_abs_error": float(
            np.max(
                np.abs(bigfingerprint_similarity - pecfp_similarity * ptfp_similarity)
            )
        ),
        "pecfp_shape": list(pecfp.shape),
        "ptfp_shape": list(ptfp.shape),
        "pecfp_binary": bool(np.isin(pecfp, [0, 1]).all()),
        "ptfp_binary": bool(np.isin(ptfp, [0, 1]).all()),
        "retained_polymer_count": len(mapping),
        "endgroup_recomputed_polymer_count": changed_count,
    }

    np.save(OUTPUT_DIR / "pecfp_fingerprints.npy", pecfp)
    np.save(OUTPUT_DIR / "ptfp_fingerprints.npy", ptfp)
    pd.DataFrame(pecfp, index=ids).rename_axis("polymer_id").to_csv(
        OUTPUT_DIR / "pecfp_fingerprints.csv",
        header=[f"pECFP_{index}" for index in range(1, PECFP_SIZE + 1)],
    )
    pd.DataFrame(ptfp, index=ids).rename_axis("polymer_id").to_csv(
        OUTPUT_DIR / "ptfp_fingerprints.csv",
        header=[f"pTFP_{index}" for index in range(1, PTFP_SIZE + 1)],
    )
    save_matrix(pecfp_similarity, "pecfp_tanimoto_similarity_482", ids)
    save_matrix(ptfp_similarity, "ptfp_tanimoto_similarity_482", ids)
    save_matrix(
        bigfingerprint_similarity,
        "bigfingerprint_product_similarity_482",
        ids,
    )

    corrected_mapping = mapping.copy()
    corrected_mapping["p-ECFP active bits"] = pecfp.sum(axis=1).astype(int)
    corrected_mapping["p-TFP active bits"] = ptfp.sum(axis=1).astype(int)
    source_by_id = {
        record["polymer_id"]: (
            "H-end-group-expanded candidate p-ECFP bitwise union"
            if record["pECFP_recomputed"]
            else "unchanged standard repository calculation"
        )
        for record in audit_records
    }
    corrected_mapping["p-ECFP source"] = corrected_mapping["polymer_id"].map(
        source_by_id
    )
    corrected_mapping.to_csv(OUTPUT_DIR / "polymer_id_mapping_482.csv", index=False)
    pd.DataFrame(audit_records).to_csv(
        OUTPUT_DIR / "endgroup_preprocessing_audit.csv",
        index=False,
    )

    metadata = {
        "dataset": "482-polymer end-group-corrected vector similarity search dataset",
        "source_directory": str(SOURCE_DIR),
        "polymer_count": EXPECTED_POLYMERS,
        "p-ECFP": {
            "fingerprint_size": PECFP_SIZE,
            "preprocessing": (
                "Missing or explicit [H] end groups are expanded using every "
                "compatible adjacent repeat-unit attachment; candidate binary "
                "p-ECFP vectors are combined by bitwise OR."
            ),
            "recomputed_polymer_count": changed_count,
            "audit_file": "endgroup_preprocessing_audit.csv",
        },
        "p-TFP": {
            "fingerprint_size": PTFP_SIZE,
            "definition": "Calculated on the original BigSMILES; unchanged from results_482",
        },
        "similarity": {
            "p-ECFP": "binary Tanimoto",
            "p-TFP": "binary Tanimoto",
            "BigFingerprint": "p-ECFP Tanimoto multiplied by p-TFP Tanimoto",
        },
        "validation_file": "validation_summary.json",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )

    elapsed = time.monotonic() - started
    print(json.dumps(validation, indent=2), flush=True)
    print(f"Completed in {elapsed:.2f} seconds", flush=True)
    print(f"Results: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

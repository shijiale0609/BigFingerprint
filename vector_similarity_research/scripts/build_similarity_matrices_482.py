#!/usr/bin/env python3
"""Build the 482-polymer similarity dataset without p-TFP timeout cases.

The seven BigSMILES that timed out in the historical standard p-TFP run are
excluded completely.  No inferred or recovered p-TFP bits enter these outputs.
The retained polymer identifiers are kept unchanged from the 489-polymer
dataset so that rows remain traceable to the source mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
SOURCE_RESULTS_DIR = HERE / "results_489"
SOURCE_PTFP_NPY = (
    PROJECT_DIR / "BigSMARTS" / "extracted_target_bigsmiles_unique_ptfp.npy"
)
OUTPUT_DIR = HERE / "results_482"

EXPECTED_SOURCE_POLYMERS = 489
EXPECTED_RETAINED_POLYMERS = 482
PECFP_SIZE = 2048
PTFP_SIZE = 40

EXCLUDED_TIMEOUT_CASES = {
    "P401": 405,
    "P402": 406,
    "P429": 433,
    "P430": 434,
    "P431": 435,
    "P459": 463,
    "P467": 471,
}


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
    expected_shape = (EXPECTED_RETAINED_POLYMERS, EXPECTED_RETAINED_POLYMERS)
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

    mapping = pd.read_csv(SOURCE_RESULTS_DIR / "polymer_id_mapping_489.csv")
    pecfp_489 = np.load(SOURCE_RESULTS_DIR / "pecfp_fingerprints.npy")
    cached_ptfp_493 = np.load(SOURCE_PTFP_NPY)

    if len(mapping) != EXPECTED_SOURCE_POLYMERS:
        raise ValueError(f"Expected 489 mapping rows; found {len(mapping)}")
    if pecfp_489.shape != (EXPECTED_SOURCE_POLYMERS, PECFP_SIZE):
        raise ValueError(f"Unexpected source p-ECFP shape: {pecfp_489.shape}")
    if cached_ptfp_493.shape != (493, PTFP_SIZE):
        raise ValueError(f"Unexpected cached p-TFP shape: {cached_ptfp_493.shape}")

    expected_target_indices = np.arange(5, 494)
    observed_target_indices = mapping["original_target_index"].to_numpy(dtype=int)
    if not np.array_equal(observed_target_indices, expected_target_indices):
        raise ValueError("The 489-polymer mapping is not aligned to targets 5-493")

    observed_exclusions = dict(
        zip(
            mapping.loc[
                mapping["polymer_id"].isin(EXCLUDED_TIMEOUT_CASES),
                "polymer_id",
            ],
            mapping.loc[
                mapping["polymer_id"].isin(EXCLUDED_TIMEOUT_CASES),
                "original_target_index",
            ].astype(int),
        )
    )
    if observed_exclusions != EXCLUDED_TIMEOUT_CASES:
        raise ValueError(f"Unexpected timeout-case mapping: {observed_exclusions}")

    cached_ptfp_489 = cached_ptfp_493[4:].astype(np.int16)
    keep_mask = ~mapping["polymer_id"].isin(EXCLUDED_TIMEOUT_CASES).to_numpy()
    excluded_mask = ~keep_mask

    excluded_cached = cached_ptfp_489[excluded_mask]
    if not (excluded_cached < 0).any(axis=1).all():
        raise ValueError("An excluded timeout case does not contain the expected cache sentinel")

    retained_mapping = mapping.loc[keep_mask].copy().reset_index(drop=True)
    pecfp = (pecfp_489[keep_mask] != 0).astype(np.uint8)
    ptfp = cached_ptfp_489[keep_mask]

    if len(retained_mapping) != EXPECTED_RETAINED_POLYMERS:
        raise ValueError(f"Expected 482 retained polymers; found {len(retained_mapping)}")
    if pecfp.shape != (EXPECTED_RETAINED_POLYMERS, PECFP_SIZE):
        raise ValueError(f"Unexpected retained p-ECFP shape: {pecfp.shape}")
    if ptfp.shape != (EXPECTED_RETAINED_POLYMERS, PTFP_SIZE):
        raise ValueError(f"Unexpected retained p-TFP shape: {ptfp.shape}")
    if not np.isin(pecfp, [0, 1]).all():
        raise ValueError("Retained p-ECFP matrix is not binary")
    if not np.isin(ptfp, [0, 1]).all():
        raise ValueError("Retained p-TFP matrix contains a timeout or non-binary value")
    ptfp = ptfp.astype(np.uint8)

    ids = retained_mapping["polymer_id"].tolist()
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
                np.abs(
                    bigfingerprint_similarity
                    - pecfp_similarity * ptfp_similarity
                )
            )
        ),
        "pecfp_shape": list(pecfp.shape),
        "ptfp_shape": list(ptfp.shape),
        "pecfp_binary": bool(np.isin(pecfp, [0, 1]).all()),
        "ptfp_binary": bool(np.isin(ptfp, [0, 1]).all()),
        "retained_polymer_count": len(retained_mapping),
        "excluded_timeout_case_count": len(EXCLUDED_TIMEOUT_CASES),
        "recovered_or_inferred_ptfp_rows_used": 0,
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

    retained_mapping.insert(0, "dataset_position", np.arange(1, 483))
    retained_mapping["p-ECFP active bits"] = pecfp.sum(axis=1).astype(int)
    retained_mapping["p-TFP active bits"] = ptfp.sum(axis=1).astype(int)
    retained_mapping["p-TFP source"] = "standard cached calculation; no recovery"
    retained_mapping.to_csv(OUTPUT_DIR / "polymer_id_mapping_482.csv", index=False)

    excluded = mapping.loc[
        excluded_mask,
        [
            "polymer_id",
            "original_target_index",
            "source_excel_row",
            "Name",
            "Source",
            "BigSMILES",
        ],
    ].copy()
    excluded.insert(0, "exclusion_reason", "historical standard p-TFP timeout")
    excluded["used_in_482_dataset"] = False
    excluded.to_csv(OUTPUT_DIR / "excluded_ptfp_timeout_cases.csv", index=False)

    pd.DataFrame(
        columns=["polymer_id", "original_target_index", "BigSMILES", "stage", "error"]
    ).to_csv(OUTPUT_DIR / "fingerprint_errors.csv", index=False)

    metadata = {
        "dataset": (
            "482 polymer BigSMILES retained from the 489-polymer dataset after "
            "excluding all seven historical standard p-TFP timeout cases"
        ),
        "source_polymer_count": EXPECTED_SOURCE_POLYMERS,
        "retained_polymer_count": EXPECTED_RETAINED_POLYMERS,
        "polymer_ids": (
            "Stable identifiers from the 489-polymer dataset are retained; "
            "see polymer_id_mapping_482.csv and dataset_position"
        ),
        "excluded_timeout_cases": EXCLUDED_TIMEOUT_CASES,
        "exclusion_audit": "excluded_ptfp_timeout_cases.csv",
        "p-ECFP": {
            "fingerprint_size": PECFP_SIZE,
            "source": "results_489/pecfp_fingerprints.npy",
            "binary_conversion": "nonzero entries mapped to 1",
        },
        "p-TFP": {
            "implementation": "polymer_topfp.bigsmiles_topology_vector",
            "fingerprint_size": PTFP_SIZE,
            "source": "BigSMARTS/extracted_target_bigsmiles_unique_ptfp.npy",
            "recovered_or_inferred_rows_used": 0,
        },
        "similarity": {
            "p-ECFP": "binary Tanimoto",
            "p-TFP": "binary Tanimoto",
            "BigFingerprint": (
                "p-ECFP Tanimoto multiplied elementwise by p-TFP Tanimoto"
            ),
        },
        "validation_file": "validation_summary.json",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    elapsed = time.monotonic() - started
    print(json.dumps(validation, indent=2), flush=True)
    print(f"Completed in {elapsed:.2f} seconds", flush=True)
    print(f"Results: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

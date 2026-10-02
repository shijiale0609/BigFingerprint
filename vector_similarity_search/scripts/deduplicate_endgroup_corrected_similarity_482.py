#!/usr/bin/env python3
"""Deduplicate the end-group-corrected 482-polymer similarity dataset.

BigFingerprint similarity=1 is used only to screen candidate clusters.  Each
cluster is then classified by its BigSMILES chemistry/topology.  Only verified
equivalent representations are removed; distinct-polymer fingerprint
collisions remain in the dataset.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "results_482_endgroup_corrected"
OUTPUT_DIR = HERE / "results_475_endgroup_corrected_deduplicated"

EXPECTED_SOURCE_COUNT = 482
EXPECTED_OUTPUT_COUNT = 475
PECFP_SIZE = 2048
PTFP_SIZE = 40


CLUSTER_REVIEWS = [
    {
        "cluster": "S1-001",
        "ids": ["P10", "P15", "P29"],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "Different n-alkyl side-chain lengths in poly(vinyl ether)s "
            "(hexyl, octyl, and decyl). The radius-2 p-ECFP bit set saturates "
            "and loses chain-length counts."
        ),
    },
    {
        "cluster": "S1-002",
        "ids": ["P20", "P22"],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "Polyoxyhexamethylene and polyoxyoctamethylene have different "
            "methylene spacer lengths; identical set fingerprints do not "
            "preserve feature multiplicity."
        ),
    },
    {
        "cluster": "S1-003",
        "ids": ["P24", "P446"],
        "classification": "Distinct topology — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "P24 is linear poly(ethylene oxide), whereas P446 is a cyclic "
            "ethylene-oxide/crown-ether ensemble. Their local chemistry is "
            "identical, but the macromolecular topology differs."
        ),
    },
    {
        "cluster": "S1-004",
        "ids": ["P28", "P32", "P34", "P35"],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "The n-hexyl, n-heptyl, n-octyl, and n-nonyl acrylates have "
            "different side-chain lengths. The equal score is a local-"
            "fingerprint collision."
        ),
    },
    {
        "cluster": "S1-005",
        "ids": ["P33", "P36"],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "Poly(ε-caprolactone) and poly(decamethylene adipate) have "
            "different repeat-unit organization and composition despite "
            "sharing the same radius-2 local environments."
        ),
    },
    {
        "cluster": "S1-006",
        "ids": ["P38", "P64", "P75"],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "The n-hexyl, n-octyl, and n-dodecyl methacrylates have different "
            "side-chain lengths; equal set fingerprints do not imply the same "
            "polymer."
        ),
    },
    {
        "cluster": "S1-007",
        "ids": ["P49", "P59"],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "Poly(ethylene azelate) and poly(ethylene sebacate) differ by one "
            "methylene in the diacid-derived segment."
        ),
    },
    {
        "cluster": "S1-008",
        "ids": [
            "P100",
            "P102",
            "P105",
            "P109",
            "P112",
            "P118",
            "P119",
            "P128",
            "P130",
            "P137",
        ],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "This cluster contains chemically distinct polyamides with "
            "different methylene counts and/or AA–BB versus AB repeat "
            "organization. Similarity 1 is a fingerprint collision."
        ),
    },
    {
        "cluster": "S1-009",
        "ids": ["P265", "P266"],
        "classification": "Distinct polymers — retain all",
        "representative": "",
        "removed": [],
        "basis": (
            "The two aromatic polysulfones contain different numbers and "
            "placements of sulfone-bearing aromatic units per repeat "
            "description."
        ),
    },
    {
        "cluster": "S1-010",
        "ids": ["P307", "P308"],
        "classification": "Equivalent BigSMILES — deduplicate",
        "representative": "P307",
        "removed": ["P308"],
        "basis": (
            "Both strings describe linear α,ω-dihydroxy poly(ethylene "
            "glycol). They differ only in chain traversal and placement of a "
            "complete ethylene-oxide unit between the stochastic object and "
            "the deterministic terminal fragment."
        ),
    },
    {
        "cluster": "S1-011",
        "ids": ["P336", "P395"],
        "classification": "Equivalent BigSMILES — deduplicate",
        "representative": "P336",
        "removed": ["P395"],
        "basis": (
            "The strings differ only in the order used to list the two repeat "
            "alternatives inside the same stochastic object."
        ),
    },
    {
        "cluster": "S1-012",
        "ids": ["P343", "P352"],
        "classification": "Equivalent BigSMILES — deduplicate",
        "representative": "P343",
        "removed": ["P352"],
        "basis": (
            "The two unspecified-end diblock copolymers are the same chain "
            "written from opposite ends (PS–PI versus PI–PS)."
        ),
    },
    {
        "cluster": "S1-013",
        "ids": ["P345", "P346"],
        "classification": "Equivalent BigSMILES — deduplicate",
        "representative": "P345",
        "removed": ["P346"],
        "basis": (
            "The two unspecified-end diblock copolymers are identical under "
            "reversal of the linear-chain traversal."
        ),
    },
    {
        "cluster": "S1-014",
        "ids": ["P362", "P372"],
        "classification": "Equivalent BigSMILES — deduplicate",
        "representative": "P362",
        "removed": ["P372"],
        "basis": (
            "The two unspecified-end diblock copolymers are the same PS–P2VP "
            "chain written from opposite ends."
        ),
    },
    {
        "cluster": "S1-015",
        "ids": ["P366", "P367"],
        "classification": "Equivalent BigSMILES — deduplicate",
        "representative": "P366",
        "removed": ["P367"],
        "basis": (
            "The two unspecified-end PS–PMMA diblock copolymers differ only "
            "by reversal of the linear-chain traversal."
        ),
    },
    {
        "cluster": "S1-016",
        "ids": ["P389", "P397"],
        "classification": "Equivalent BigSMILES — deduplicate",
        "representative": "P389",
        "removed": ["P397"],
        "basis": (
            "P397 is the reverse traversal of P389, with the same hydrogen/"
            "sec-butyl termini and only a reordered listing of repeat "
            "alternatives in one stochastic block."
        ),
    },
]


def save_matrix(matrix: np.ndarray, stem: str, ids: list[str]) -> None:
    frame = pd.DataFrame(matrix, index=ids, columns=ids)
    frame.index.name = "polymer_id"
    frame.to_csv(OUTPUT_DIR / f"{stem}.csv", float_format="%.8f")
    np.save(OUTPUT_DIR / f"{stem}.npy", matrix)


def similarity_one_clusters(matrix: np.ndarray, ids: list[str]) -> tuple[list, list]:
    graph = nx.Graph()
    pairs = []
    for left_index, right_index in itertools.combinations(range(len(ids)), 2):
        if abs(float(matrix[left_index, right_index]) - 1.0) <= 1e-12:
            left_id, right_id = ids[left_index], ids[right_index]
            graph.add_edge(left_id, right_id)
            pairs.append((left_id, right_id))
    clusters = [
        sorted(component, key=lambda value: int(value[1:]))
        for component in nx.connected_components(graph)
    ]
    clusters.sort(key=lambda values: int(values[0][1:]))
    return pairs, clusters


def validate_matrix(matrix: np.ndarray, count: int, label: str) -> dict:
    if matrix.shape != (count, count):
        raise ValueError(f"{label} shape is {matrix.shape}; expected {(count, count)}")
    symmetry = float(np.max(np.abs(matrix - matrix.T)))
    diagonal = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if symmetry > 1e-12 or diagonal > 1e-12:
        raise ValueError(f"{label} failed symmetry/diagonal validation")
    return {
        "shape": list(matrix.shape),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "symmetry_max_abs_error": symmetry,
        "diagonal_max_abs_error": diagonal,
    }


def main() -> None:
    started = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(SOURCE_DIR / "polymer_id_mapping_482.csv")
    pecfp = np.load(SOURCE_DIR / "pecfp_fingerprints.npy")
    ptfp = np.load(SOURCE_DIR / "ptfp_fingerprints.npy")
    pecfp_similarity = np.load(SOURCE_DIR / "pecfp_tanimoto_similarity_482.npy")
    ptfp_similarity = np.load(SOURCE_DIR / "ptfp_tanimoto_similarity_482.npy")
    bigfingerprint_similarity = np.load(
        SOURCE_DIR / "bigfingerprint_product_similarity_482.npy"
    )

    if len(mapping) != EXPECTED_SOURCE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SOURCE_COUNT} source rows")
    if pecfp.shape != (EXPECTED_SOURCE_COUNT, PECFP_SIZE):
        raise ValueError(f"Unexpected p-ECFP shape: {pecfp.shape}")
    if ptfp.shape != (EXPECTED_SOURCE_COUNT, PTFP_SIZE):
        raise ValueError(f"Unexpected p-TFP shape: {ptfp.shape}")

    ids = mapping["polymer_id"].tolist()
    pairs, observed_clusters = similarity_one_clusters(bigfingerprint_similarity, ids)
    expected_clusters = [review["ids"] for review in CLUSTER_REVIEWS]
    if observed_clusters != expected_clusters:
        raise ValueError(
            "Observed similarity=1 clusters do not match the reviewed clusters:\n"
            f"observed={observed_clusters}\nexpected={expected_clusters}"
        )

    removed_ids = [
        polymer_id for review in CLUSTER_REVIEWS for polymer_id in review["removed"]
    ]
    if len(removed_ids) != 7 or len(set(removed_ids)) != 7:
        raise ValueError(f"Unexpected removed-ID list: {removed_ids}")

    id_to_index = {polymer_id: index for index, polymer_id in enumerate(ids)}
    keep_mask = ~mapping["polymer_id"].isin(removed_ids).to_numpy()
    kept_indices = np.flatnonzero(keep_mask)
    kept_ids = mapping.loc[keep_mask, "polymer_id"].tolist()
    if len(kept_ids) != EXPECTED_OUTPUT_COUNT:
        raise ValueError(f"Expected {EXPECTED_OUTPUT_COUNT} retained polymers")

    dedup_pecfp = pecfp[keep_mask]
    dedup_ptfp = ptfp[keep_mask]
    dedup_pecfp_similarity = pecfp_similarity[np.ix_(kept_indices, kept_indices)]
    dedup_ptfp_similarity = ptfp_similarity[np.ix_(kept_indices, kept_indices)]
    dedup_bigfingerprint_similarity = bigfingerprint_similarity[
        np.ix_(kept_indices, kept_indices)
    ]

    audit_rows = []
    removed_rows = []
    for review in CLUSTER_REVIEWS:
        candidate_ids = ", ".join(review["ids"])
        representative = review["representative"]
        for polymer_id in review["ids"]:
            row = mapping.loc[mapping["polymer_id"] == polymer_id].iloc[0]
            reference_id = representative or review["ids"][0]
            p_i = id_to_index[polymer_id]
            r_i = id_to_index[reference_id]
            removed = polymer_id in review["removed"]
            if removed:
                action = "Remove duplicate"
            elif representative and polymer_id == representative:
                action = "Keep representative"
            else:
                action = "Retain distinct polymer"
            audit_rows.append(
                {
                    "Cluster": review["cluster"],
                    "Candidate IDs": candidate_ids,
                    "Polymer ID": polymer_id,
                    "Original dataset position": int(row["dataset_position"]),
                    "Name": "" if pd.isna(row["Name"]) else row["Name"],
                    "Source": "" if pd.isna(row["Source"]) else row["Source"],
                    "BigSMILES": row["BigSMILES"],
                    "p-ECFP active bits": int(row["p-ECFP active bits"]),
                    "p-TFP active bits": int(row["p-TFP active bits"]),
                    "p-ECFP similarity": float(pecfp_similarity[p_i, r_i]),
                    "p-TFP similarity": float(ptfp_similarity[p_i, r_i]),
                    "BigFingerprint similarity": float(
                        bigfingerprint_similarity[p_i, r_i]
                    ),
                    "Review classification": review["classification"],
                    "Representative ID": representative,
                    "Action": action,
                    "Verification basis": review["basis"],
                }
            )
            if removed:
                removed_rows.append(
                    {
                        "Removed polymer ID": polymer_id,
                        "Representative polymer ID": representative,
                        "Removed original dataset position": int(
                            row["dataset_position"]
                        ),
                        "Removed BigSMILES": row["BigSMILES"],
                        "Representative BigSMILES": mapping.loc[
                            mapping["polymer_id"] == representative,
                            "BigSMILES",
                        ].iloc[0],
                        "Verification basis": review["basis"],
                    }
                )

    dedup_mapping = mapping.loc[keep_mask].copy().reset_index(drop=True)
    dedup_mapping = dedup_mapping.rename(
        columns={"dataset_position": "original_dataset_position"}
    )
    dedup_mapping.insert(
        0,
        "deduplicated_position",
        np.arange(1, EXPECTED_OUTPUT_COUNT + 1),
    )
    dedup_mapping.to_csv(OUTPUT_DIR / "polymer_id_mapping_475.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_DIR / "similarity_one_audit_482.csv",
        index=False,
    )
    pd.DataFrame(removed_rows).to_csv(
        OUTPUT_DIR / "removed_duplicate_representations.csv",
        index=False,
    )

    np.save(OUTPUT_DIR / "pecfp_fingerprints.npy", dedup_pecfp)
    np.save(OUTPUT_DIR / "ptfp_fingerprints.npy", dedup_ptfp)
    pd.DataFrame(dedup_pecfp, index=kept_ids).rename_axis("polymer_id").to_csv(
        OUTPUT_DIR / "pecfp_fingerprints.csv",
        header=[f"pECFP_{index}" for index in range(1, PECFP_SIZE + 1)],
    )
    pd.DataFrame(dedup_ptfp, index=kept_ids).rename_axis("polymer_id").to_csv(
        OUTPUT_DIR / "ptfp_fingerprints.csv",
        header=[f"pTFP_{index}" for index in range(1, PTFP_SIZE + 1)],
    )
    save_matrix(
        dedup_pecfp_similarity,
        "pecfp_tanimoto_similarity_475",
        kept_ids,
    )
    save_matrix(
        dedup_ptfp_similarity,
        "ptfp_tanimoto_similarity_475",
        kept_ids,
    )
    save_matrix(
        dedup_bigfingerprint_similarity,
        "bigfingerprint_product_similarity_475",
        kept_ids,
    )

    remaining_pairs, remaining_clusters = similarity_one_clusters(
        dedup_bigfingerprint_similarity,
        kept_ids,
    )
    expected_remaining_clusters = [
        review["ids"] for review in CLUSTER_REVIEWS if not review["removed"]
    ]
    if remaining_clusters != expected_remaining_clusters:
        raise ValueError("Unexpected similarity=1 clusters remain after deduplication")

    validation = {
        "source_polymer_count": EXPECTED_SOURCE_COUNT,
        "deduplicated_polymer_count": EXPECTED_OUTPUT_COUNT,
        "source_similarity_one_pair_count": len(pairs),
        "source_similarity_one_cluster_count": len(observed_clusters),
        "source_similarity_one_polymer_count": sum(map(len, observed_clusters)),
        "equivalent_representation_cluster_count": sum(
            bool(review["removed"]) for review in CLUSTER_REVIEWS
        ),
        "distinct_polymer_collision_cluster_count": sum(
            not review["removed"] for review in CLUSTER_REVIEWS
        ),
        "removed_duplicate_count": len(removed_ids),
        "removed_ids": removed_ids,
        "remaining_similarity_one_pair_count": len(remaining_pairs),
        "remaining_similarity_one_cluster_count": len(remaining_clusters),
        "p-ECFP": validate_matrix(
            dedup_pecfp_similarity,
            EXPECTED_OUTPUT_COUNT,
            "p-ECFP",
        ),
        "p-TFP": validate_matrix(
            dedup_ptfp_similarity,
            EXPECTED_OUTPUT_COUNT,
            "p-TFP",
        ),
        "BigFingerprint": validate_matrix(
            dedup_bigfingerprint_similarity,
            EXPECTED_OUTPUT_COUNT,
            "BigFingerprint",
        ),
        "bigfingerprint_product_max_abs_error": float(
            np.max(
                np.abs(
                    dedup_bigfingerprint_similarity
                    - dedup_pecfp_similarity * dedup_ptfp_similarity
                )
            )
        ),
    }
    metadata = {
        "dataset": "End-group-corrected 482-polymer dataset after verified deduplication",
        "source_directory": str(SOURCE_DIR),
        "source_polymer_count": EXPECTED_SOURCE_COUNT,
        "deduplicated_polymer_count": EXPECTED_OUTPUT_COUNT,
        "candidate_rule": "Off-diagonal BigFingerprint similarity exactly 1",
        "deduplication_rule": (
            "Remove only BigSMILES verified to represent the same polymer; "
            "retain distinct-polymer fingerprint collisions."
        ),
        "representative_rule": (
            "Keep the lowest original dataset position within each verified "
            "equivalence cluster."
        ),
        "removed_ids": removed_ids,
        "audit_file": "similarity_one_audit_482.csv",
        "removed_file": "removed_duplicate_representations.csv",
        "validation_file": "validation_summary.json",
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(validation, indent=2), flush=True)
    print(f"Completed in {time.monotonic() - started:.2f} seconds", flush=True)
    print(f"Results: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

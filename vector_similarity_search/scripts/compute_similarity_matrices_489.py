#!/usr/bin/env python3
"""Compute p-ECFP, p-TFP, and product BigFingerprint similarities for 489 polymers.

The input order is the exact-string unique target order from the BigSMARTS
query-target dataset after excluding its first four small-molecule entries.

Similarity definitions match the eight-polymer vector similarity search experiment:

    p-ECFP similarity = binary Tanimoto(p-ECFP_i, p-ECFP_j)
    p-TFP similarity  = binary Tanimoto(p-TFP_i, p-TFP_j)
    BigFingerprint similarity = p-ECFP similarity * p-TFP similarity
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import signal
import time

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import rdBase

from bigfingerprint import polymer_ecfp, polymer_topfp, search_tools

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
BIGSMARTS_DIR = PROJECT_DIR / "BigSMARTS"
SOURCE_RECORDS_CSV = BIGSMARTS_DIR / "extracted_target_bigsmiles.csv"
SOURCE_UNIQUE_TXT = BIGSMARTS_DIR / "extracted_target_bigsmiles_unique.txt"
SOURCE_PTFP_NPY = BIGSMARTS_DIR / "extracted_target_bigsmiles_unique_ptfp.npy"
OUTPUT_DIR = HERE / "results_489"
CHECKPOINT_DIR = OUTPUT_DIR / ".checkpoints"

FP_SIZE = 2048
PECFP_RADIUS = 2
PECFP_TIMEOUT_SECONDS = 180
EXPECTED_ALL_UNIQUE = 493
EXPECTED_POLYMERS = 489
EXCLUDED_SMALL_MOLECULES = ["N", "O", "C1CCCCC1", "c1ccccc1"]
PECFP_FAST_CLUSTER_TARGET_INDICES = {440}


class FingerprintTimeoutError(TimeoutError):
    """Raised when one p-ECFP calculation exceeds its safety limit."""


def _timeout_handler(signum, frame):
    raise FingerprintTimeoutError(
        f"p-ECFP calculation exceeded {PECFP_TIMEOUT_SECONDS} seconds"
    )


def load_polymer_records() -> tuple[pd.DataFrame, list[str]]:
    """Load the 493 exact-string unique targets and retain the 489 polymers."""
    occurrences = pd.read_csv(SOURCE_RECORDS_CSV)
    unique = occurrences.drop_duplicates("target_bigsmiles", keep="first").reset_index(
        drop=True
    )
    if len(unique) != EXPECTED_ALL_UNIQUE:
        raise ValueError(f"Expected 493 unique targets, found {len(unique)}")

    first_four = unique.loc[:3, "target_bigsmiles"].tolist()
    if first_four != EXCLUDED_SMALL_MOLECULES:
        raise ValueError(f"Unexpected first four targets: {first_four}")

    source_order = [
        line.strip()
        for line in SOURCE_UNIQUE_TXT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if source_order != unique["target_bigsmiles"].tolist():
        raise ValueError(
            "The unique target text file and source CSV order do not match"
        )

    polymers = unique.iloc[4:].copy().reset_index(drop=True)
    polymers.insert(0, "polymer_id", [f"P{i}" for i in range(1, len(polymers) + 1)])
    polymers.insert(1, "original_target_index", np.arange(5, len(unique) + 1))
    polymers = polymers.rename(
        columns={
            "target_bigsmiles": "BigSMILES",
            "excel_row": "source_excel_row",
            "name": "Name",
            "source": "Source",
        }
    )
    polymers["Name"] = polymers["Name"].fillna("")
    polymers["Source"] = polymers["Source"].fillna("")
    if len(polymers) != EXPECTED_POLYMERS:
        raise ValueError(f"Expected 489 polymers, found {len(polymers)}")
    return polymers, source_order


def identify_cycles_fast(graphs: dict) -> list[set]:
    """Find stochastic-object clusters by SCC instead of enumerating all cycles.

    The original cycle enumeration and this SCC form produced identical
    2048-bit p-ECFP vectors for all 488 targets on which the original method
    completed.  It avoids combinatorial cycle enumeration for target 440.
    """
    topology = graphs["topology"]
    ids = nx.get_node_attributes(topology, "ids")
    explicit = nx.get_node_attributes(graphs["atomistic"], "explicit_atom_ids")
    atom_ids = nx.get_node_attributes(graphs["atomistic"], "ids")
    explicit_ids = {
        atom_ids[node] for node, is_explicit in explicit.items() if is_explicit
    }

    clusters = []
    for component in nx.strongly_connected_components(topology):
        cyclic = len(component) > 1 or any(
            topology.has_edge(node, node) for node in component
        )
        if not cyclic:
            continue
        remaining_ids = {ids[node] for node in component} - explicit_ids
        if remaining_ids:
            clusters.append(remaining_ids)
    return clusters


def compute_one_pecfp(bigsmiles: str, use_fast_clusters: bool = False) -> np.ndarray:
    """Compute the repository's 2048-bit, radius-2 p-ECFP for one BigSMILES."""
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(PECFP_TIMEOUT_SECONDS)
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            graphs = search_tools.generate_NetworkX_graphs(
                bigsmiles, is_bigsmarts=False
            )
            if use_fast_clusters:
                clusters = identify_cycles_fast(graphs)
            else:
                _cycles, clusters, _treat_as_cycle = search_tools.identify_cycles(
                    graphs
                )
            vector = polymer_ecfp.ECFP(
                graphs=graphs,
                individual_clusters=clusters,
                radius=PECFP_RADIUS,
                repeat_unit_property=False,
                backbone_property=False,
                fp_size=FP_SIZE,
                print_identifiers=False,
                print_all=False,
                save_graph=False,
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    vector = (np.asarray(vector).reshape(-1) != 0).astype(np.uint8)
    if vector.shape != (FP_SIZE,):
        raise ValueError(f"Expected a {FP_SIZE}-bit p-ECFP, got {vector.shape}")
    return vector


def compute_all_pecfp(polymers: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """Compute p-ECFP with resumable checkpoints and return any failures."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path = CHECKPOINT_DIR / "pecfp_vectors.npy"
    completed_path = CHECKPOINT_DIR / "pecfp_completed.npy"
    count = len(polymers)

    if vectors_path.exists() and completed_path.exists():
        vectors = np.load(vectors_path)
        completed = np.load(completed_path)
        if vectors.shape != (count, FP_SIZE) or completed.shape != (count,):
            raise ValueError("Existing p-ECFP checkpoint has an unexpected shape")
    else:
        vectors = np.zeros((count, FP_SIZE), dtype=np.uint8)
        completed = np.zeros(count, dtype=bool)

    errors: list[dict] = []
    for row_index, row in polymers.iterrows():
        if completed[row_index]:
            continue
        started = time.monotonic()
        try:
            use_fast_clusters = (
                int(row["original_target_index"]) in PECFP_FAST_CLUSTER_TARGET_INDICES
            )
            vectors[row_index] = compute_one_pecfp(
                row["BigSMILES"],
                use_fast_clusters=use_fast_clusters,
            )
            completed[row_index] = True
            status = "ok"
        except Exception as exc:  # keep a complete audit trail before stopping
            status = "error"
            errors.append(
                {
                    "polymer_id": row["polymer_id"],
                    "original_target_index": int(row["original_target_index"]),
                    "BigSMILES": row["BigSMILES"],
                    "stage": "p-ECFP",
                    "error": repr(exc),
                }
            )
        elapsed = time.monotonic() - started

        if (row_index + 1) % 5 == 0 or status == "error" or row_index + 1 == count:
            np.save(vectors_path, vectors)
            np.save(completed_path, completed)
            print(
                f"p-ECFP {row_index + 1}/{count}: {status} "
                f"({elapsed:.2f} s; completed={int(completed.sum())})",
                flush=True,
            )

    if not completed.all():
        return vectors, errors
    return vectors, errors


RECOVERED_FIRST17 = {
    # Six and ten sequential stochastic objects, respectively.
    405: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    406: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Four-arm, multi-block/H topologies with six object levels along the
    # corresponding high-order topology search.
    433: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    434: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    # Eight-arm star; higher block thresholds are zero because the third
    # block-search bit is already a completed negative match.
    435: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    463: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    471: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
}


def compute_ptfp_tail(bigsmiles: str) -> np.ndarray:
    """Compute p-TFP dimensions 18-40 without the expensive topology search."""
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        fragments = polymer_topfp.bigsmiles_to_molecular_fragment_smiles_list(bigsmiles)
        tail = (
            polymer_topfp.build_chirality_vector(fragments)
            + polymer_topfp.build_stereochemistry_vector(fragments)
            + polymer_topfp.build_linear_statistical_copolymer_vector(bigsmiles)
            + polymer_topfp.build_network_vector(bigsmiles)
        )
    tail = np.asarray(tail, dtype=np.uint8)
    if tail.shape != (23,):
        raise ValueError(f"Expected 23 p-TFP tail dimensions, got {tail.shape}")
    return tail


def load_and_recover_ptfp(
    polymers: pd.DataFrame,
    source_order: list[str],
) -> tuple[np.ndarray, list[dict]]:
    """Load validated cached p-TFP and recover seven historical timeout rows."""
    cached = np.load(SOURCE_PTFP_NPY)
    if cached.shape != (EXPECTED_ALL_UNIQUE, 40):
        raise ValueError(f"Unexpected cached p-TFP shape: {cached.shape}")

    vectors = cached[4:].copy().astype(np.int16)
    timeout_rows = np.flatnonzero((vectors < 0).any(axis=1))
    original_timeout_indices = [int(index + 5) for index in timeout_rows]
    if original_timeout_indices != sorted(RECOVERED_FIRST17):
        raise ValueError(
            "Cached p-TFP timeout indices do not match the audited recovery set: "
            f"{original_timeout_indices}"
        )

    recovery_records = []
    for row_index, original_index in zip(timeout_rows, original_timeout_indices):
        bigsmiles = polymers.iloc[row_index]["BigSMILES"]
        if bigsmiles != source_order[original_index - 1]:
            raise ValueError(
                f"p-TFP recovery order mismatch at target {original_index}"
            )
        first17 = np.asarray(RECOVERED_FIRST17[original_index], dtype=np.uint8)
        tail23 = compute_ptfp_tail(bigsmiles)
        vectors[row_index] = np.concatenate([first17, tail23])
        recovery_records.append(
            {
                "polymer_id": polymers.iloc[row_index]["polymer_id"],
                "original_target_index": original_index,
                "BigSMILES": bigsmiles,
                "recovery_method": (
                    "Completed non-timeout p-TFP dimensions directly; inferred only "
                    "monotonic high-order topology bits from the explicit object connectivity."
                ),
                "pTFP_first17": "".join(str(int(value)) for value in first17),
                "pTFP_tail23": "".join(str(int(value)) for value in tail23),
            }
        )

    if (vectors < 0).any() or not np.isin(vectors, [0, 1]).all():
        raise ValueError("Recovered p-TFP matrix is not complete and binary")
    return vectors.astype(np.uint8), recovery_records


def tanimoto_matrix(fingerprints: np.ndarray) -> np.ndarray:
    """Calculate the all-pairs binary Tanimoto matrix."""
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
        raise ValueError(f"{label} has shape {matrix.shape}, expected {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    diagonal_error = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if symmetry_error > 1e-12 or diagonal_error > 1e-12:
        raise ValueError(
            f"{label} validation failed: symmetry={symmetry_error}, diagonal={diagonal_error}"
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
    polymers, source_order = load_polymer_records()
    ids = polymers["polymer_id"].tolist()

    pecfp, pecfp_errors = compute_all_pecfp(polymers)
    pd.DataFrame(
        pecfp_errors,
        columns=["polymer_id", "original_target_index", "BigSMILES", "stage", "error"],
    ).to_csv(OUTPUT_DIR / "fingerprint_errors.csv", index=False)
    if pecfp_errors:
        raise RuntimeError(
            f"p-ECFP failed for {len(pecfp_errors)} polymers; matrices were not generated"
        )

    ptfp, recovery_records = load_and_recover_ptfp(polymers, source_order)
    pd.DataFrame(recovery_records).to_csv(
        OUTPUT_DIR / "ptfp_recovered_timeout_cases.csv",
        index=False,
    )

    pecfp_similarity = tanimoto_matrix(pecfp)
    ptfp_similarity = tanimoto_matrix(ptfp)
    bigfingerprint_similarity = pecfp_similarity * ptfp_similarity

    validation = {
        "p-ECFP": validate_similarity_matrix(pecfp_similarity, "p-ECFP"),
        "p-TFP": validate_similarity_matrix(ptfp_similarity, "p-TFP"),
        "BigFingerprint": validate_similarity_matrix(
            bigfingerprint_similarity,
            "BigFingerprint",
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
        "ptfp_recovered_timeout_cases": len(recovery_records),
    }

    np.save(OUTPUT_DIR / "pecfp_fingerprints.npy", pecfp)
    np.save(OUTPUT_DIR / "ptfp_fingerprints.npy", ptfp)
    pd.DataFrame(pecfp, index=ids).rename_axis("polymer_id").to_csv(
        OUTPUT_DIR / "pecfp_fingerprints.csv",
        header=[f"pECFP_{i}" for i in range(1, FP_SIZE + 1)],
    )
    pd.DataFrame(ptfp, index=ids).rename_axis("polymer_id").to_csv(
        OUTPUT_DIR / "ptfp_fingerprints.csv",
        header=[f"pTFP_{i}" for i in range(1, 41)],
    )

    save_matrix(pecfp_similarity, "pecfp_tanimoto_similarity_489", ids)
    save_matrix(ptfp_similarity, "ptfp_tanimoto_similarity_489", ids)
    save_matrix(
        bigfingerprint_similarity,
        "bigfingerprint_product_similarity_489",
        ids,
    )

    mapping = polymers[
        [
            "polymer_id",
            "original_target_index",
            "source_excel_row",
            "Name",
            "Source",
            "BigSMILES",
        ]
    ].copy()
    mapping["p-ECFP active bits"] = pecfp.sum(axis=1).astype(int)
    mapping["p-TFP active bits"] = ptfp.sum(axis=1).astype(int)
    mapping["p-ECFP source"] = [
        (
            "SCC-equivalent cycle detection"
            if int(target_index) in PECFP_FAST_CLUSTER_TARGET_INDICES
            else "standard repository calculation"
        )
        for target_index in mapping["original_target_index"]
    ]
    recovered_ids = {record["polymer_id"] for record in recovery_records}
    mapping["p-TFP source"] = [
        (
            "recovered historical timeout"
            if polymer_id in recovered_ids
            else "validated cache"
        )
        for polymer_id in mapping["polymer_id"]
    ]
    mapping.to_csv(OUTPUT_DIR / "polymer_id_mapping_489.csv", index=False)

    metadata = {
        "dataset": "489 unique polymer BigSMILES from query-target-dataset.xlsx",
        "excluded_small_molecules": EXCLUDED_SMALL_MOLECULES,
        "polymer_count": EXPECTED_POLYMERS,
        "polymer_ids": "P1-P489; see polymer_id_mapping_489.csv",
        "p-ECFP": {
            "implementation": "polymer_ecfp.ECFP",
            "radius": PECFP_RADIUS,
            "fingerprint_size": FP_SIZE,
            "binary_conversion": "nonzero entries mapped to 1",
            "SCC_cycle_detection_target_indices": sorted(
                PECFP_FAST_CLUSTER_TARGET_INDICES
            ),
            "SCC_validation": (
                "Identical fingerprints for all 488 targets completed by the "
                "original simple-cycle enumeration (0 mismatches)."
            ),
        },
        "p-TFP": {
            "implementation": "polymer_topfp.bigsmiles_topology_vector",
            "fingerprint_size": 40,
            "historical_timeout_cases_recovered": sorted(RECOVERED_FIRST17),
            "recovery_audit": "ptfp_recovered_timeout_cases.csv",
        },
        "similarity": {
            "p-ECFP": "binary Tanimoto",
            "p-TFP": "binary Tanimoto",
            "BigFingerprint": "p-ECFP Tanimoto multiplied elementwise by p-TFP Tanimoto",
        },
        "rdkit_version": rdBase.rdkitVersion,
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

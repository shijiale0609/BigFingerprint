#!/usr/bin/env python3
"""Remove exact Advanced/Extras query BigSMILES from the corrected 475 dataset.

The Advanced and Extras query headers are extracted from rows 1--4 of the
original query-target workbook with @oai/artifact-tool.  Only exact BigSMILES
matches to the 475-polymer mapping are removed; BigSMARTS query patterns that
are not database members are retained in a separate audit file.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "results_475_endgroup_corrected_deduplicated"
HEADER_SNAPSHOT = (
    HERE.parent.parent
    / ".codex_tmp"
    / "xlsx_inspect_p423"
    / "advanced_extras_headers.json"
)
SOURCE_WORKBOOK = HERE.parent / "BigSMARTS" / "query-target-dataset.xlsx"
EXCLUDED_SHEETS = ("Advanced", "Extras")


def query_header_rows(snapshot: dict) -> list[dict]:
    rows = []
    for sheet_name in EXCLUDED_SHEETS:
        entries = snapshot["sheets"][sheet_name]["entries"]
        for entry in entries:
            # Columns A:D are target metadata. Query headers begin in column E.
            if int(entry["zeroBasedColumnIndex"]) < 4:
                continue
            rows.append(
                {
                    "source_sheet": sheet_name,
                    "source_cell": f"{entry['excelColumn']}1",
                    "query_bigsmarts": entry["value"],
                    "rule": entry.get("rule"),
                    "number_of_matches": entry.get("numberOfMatches"),
                    "case_topology": entry.get("caseTopology"),
                }
            )
    return rows


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


def main() -> None:
    snapshot = json.loads(HEADER_SNAPSHOT.read_text(encoding="utf-8"))
    headers = query_header_rows(snapshot)
    header_frame = pd.DataFrame(headers)

    mapping = pd.read_csv(SOURCE_DIR / "polymer_id_mapping_475.csv").fillna("")
    if len(mapping) != 475 or not mapping["BigSMILES"].is_unique:
        raise ValueError("Expected a 475-row mapping with unique BigSMILES")
    bigsmiles_to_row = mapping.set_index("BigSMILES", drop=False).to_dict("index")

    matched_rows = []
    unmatched_rows = []
    for header in headers:
        match = bigsmiles_to_row.get(header["query_bigsmarts"])
        if match is None:
            unmatched_rows.append(header)
            continue
        matched_rows.append(
            {
                **header,
                "polymer_id": match["polymer_id"],
                "BigSMILES": match["BigSMILES"],
                "deduplicated_position_475": int(match["deduplicated_position"]),
                "original_dataset_position": int(match["original_dataset_position"]),
                "original_target_index": int(match["original_target_index"]),
                "source_excel_row": int(match["source_excel_row"]),
                "Name": match["Name"],
                "Source": match["Source"],
            }
        )

    matched_frame = pd.DataFrame(matched_rows)
    if matched_frame.empty:
        raise ValueError("No Advanced/Extras query headers matched the 475 dataset")
    if matched_frame["polymer_id"].duplicated().any():
        duplicates = matched_frame.loc[
            matched_frame["polymer_id"].duplicated(False), "polymer_id"
        ].tolist()
        raise ValueError(f"Overlapping exact exclusions require review: {duplicates}")

    removed_ids = set(matched_frame["polymer_id"])
    keep_mask = ~mapping["polymer_id"].isin(removed_ids)
    keep_indices = np.flatnonzero(keep_mask.to_numpy())
    filtered_count = int(keep_mask.sum())
    output_dir = (
        HERE
        / f"results_{filtered_count}_endgroup_corrected_deduplicated_no_advanced_extras"
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

    matched_frame = matched_frame.sort_values(
        ["source_sheet", "deduplicated_position_475"]
    )
    matched_frame.to_csv(
        output_dir / "removed_advanced_extras_polymers.csv", index=False
    )
    pd.DataFrame(unmatched_rows).to_csv(
        output_dir / "unmatched_advanced_extras_query_headers.csv", index=False
    )
    header_frame.to_csv(output_dir / "advanced_extras_query_headers.csv", index=False)
    shutil.copy2(
        HEADER_SNAPSHOT, output_dir / "advanced_extras_query_headers_snapshot.json"
    )

    metadata = {
        "dataset": (
            "End-group-corrected, verified-deduplicated polymer dataset after "
            "excluding exact query-header BigSMILES from the Advanced and Extras sheets"
        ),
        "source_directory": str(SOURCE_DIR),
        "source_workbook": str(SOURCE_WORKBOOK),
        "source_polymer_count": int(len(mapping)),
        "filtered_polymer_count": filtered_count,
        "excluded_sheets": list(EXCLUDED_SHEETS),
        "query_header_rule": "Rows 1--4; query columns begin at E; exact BigSMILES match only",
        "query_header_counts": {
            sheet: int((header_frame["source_sheet"] == sheet).sum())
            for sheet in EXCLUDED_SHEETS
        },
        "exact_removed_counts": {
            sheet: int((matched_frame["source_sheet"] == sheet).sum())
            for sheet in EXCLUDED_SHEETS
        },
        "unmatched_query_header_count": int(len(unmatched_rows)),
        "removed_polymer_count": int(len(removed_ids)),
        "removed_ids": sorted(removed_ids, key=lambda value: int(value[1:])),
        "preserved_polymer_ids": True,
        "matrix_operation": "Exact row/column subset of corrected 475 matrices",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    validation_summary = {
        "source_count": int(len(mapping)),
        "removed_count": int(len(removed_ids)),
        "filtered_count": filtered_count,
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
                "removed_count": len(removed_ids),
                "filtered_count": filtered_count,
                "exact_removed_counts": metadata["exact_removed_counts"],
                "removed_ids": metadata["removed_ids"],
                "validation": validation_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

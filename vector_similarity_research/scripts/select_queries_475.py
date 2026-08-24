#!/usr/bin/env python3
"""Screen diverse BigFingerprint query examples in the deduplicated dataset."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "results_475_endgroup_corrected_deduplicated"


def topology_categories(bits: np.ndarray) -> list[str]:
    categories = []
    if bits[1:8].any():
        categories.append("block")
    if bits[8]:
        categories.append("segmented")
    if bits[9]:
        categories.append("graft")
    if bits[11:17].any():
        categories.append("star")
    if bits[22:40].any():
        categories.append("network")
    if not categories and bits[0]:
        categories.append("single")
    if not categories:
        categories.append("other")
    return categories


def primary_topology(categories: list[str]) -> str:
    for category in ["network", "star", "graft", "segmented", "block", "single", "other"]:
        if category in categories:
            return category
    return "other"


def compact(summary: dict) -> dict:
    keys = [
        "query_id",
        "query_topology",
        "query_primary_topology",
        "Name",
        "BigSMILES",
        "BigSMILES_length",
        "unique_primary_topology_count",
        "unique_topology_signature_count",
        "primary_topology_composition",
        "topology_signature_composition",
        "rank1_similarity",
        "rank5_similarity",
        "mean_top5_similarity",
        "distinct_top5_score_count",
        "cutoff_tie_count",
        "cutoff_slots_available",
        "cutoff_is_ambiguous",
        "max_candidate_length",
    ]
    result = {key: summary[key] for key in keys}
    result["top5"] = summary["top5"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    mapping_candidates = sorted(data_dir.glob("polymer_id_mapping_*.csv"))
    if len(mapping_candidates) != 1:
        raise ValueError(
            f"Expected exactly one polymer_id_mapping_*.csv in {data_dir}, "
            f"found {len(mapping_candidates)}"
        )
    mapping = pd.read_csv(mapping_candidates[0]).fillna("")
    count = len(mapping)
    json_path = data_dir / f"query_selection_candidates_{count}.json"
    summary_csv_path = data_dir / f"query_selection_summary_{count}.csv"
    ids = mapping["polymer_id"].tolist()
    id_to_row = mapping.set_index("polymer_id").to_dict("index")
    pecfp_similarity = np.load(data_dir / f"pecfp_tanimoto_similarity_{count}.npy")
    ptfp_similarity = np.load(data_dir / f"ptfp_tanimoto_similarity_{count}.npy")
    bigfingerprint_similarity = np.load(
        data_dir / f"bigfingerprint_product_similarity_{count}.npy"
    )
    ptfp_bits = np.load(data_dir / "ptfp_fingerprints.npy").astype(bool)

    topology = {}
    for polymer_id, bits in zip(ids, ptfp_bits):
        categories = topology_categories(bits)
        topology[polymer_id] = {
            "categories": categories,
            "signature": "+".join(categories),
            "primary": primary_topology(categories),
        }

    summaries = []
    for query_index, query_id in enumerate(ids):
        candidate_indices = [index for index in range(len(ids)) if index != query_index]
        candidate_indices.sort(
            key=lambda index: (
                -float(bigfingerprint_similarity[query_index, index]),
                int(ids[index][1:]),
            )
        )
        cutoff_similarity = float(bigfingerprint_similarity[query_index, candidate_indices[4]])
        cutoff_tie_count = sum(
            abs(float(bigfingerprint_similarity[query_index, index]) - cutoff_similarity)
            <= 1e-12
            for index in candidate_indices
        )
        above_cutoff = sum(
            float(bigfingerprint_similarity[query_index, index])
            > cutoff_similarity + 1e-12
            for index in candidate_indices
        )
        cutoff_slots_available = 5 - above_cutoff

        top5 = []
        for rank, candidate_index in enumerate(candidate_indices[:5], start=1):
            candidate_id = ids[candidate_index]
            candidate_metadata = id_to_row[candidate_id]
            top5.append(
                {
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "topology": topology[candidate_id]["signature"],
                    "primary_topology": topology[candidate_id]["primary"],
                    "Name": candidate_metadata["Name"],
                    "BigSMILES": candidate_metadata["BigSMILES"],
                    "pECFP": float(pecfp_similarity[query_index, candidate_index]),
                    "pTFP": float(ptfp_similarity[query_index, candidate_index]),
                    "BigFingerprint": float(
                        bigfingerprint_similarity[query_index, candidate_index]
                    ),
                }
            )

        query_metadata = id_to_row[query_id]
        unique_primary = sorted({candidate["primary_topology"] for candidate in top5})
        unique_signatures = sorted({candidate["topology"] for candidate in top5})
        summaries.append(
            {
                "query_id": query_id,
                "query_topology": topology[query_id]["signature"],
                "query_primary_topology": topology[query_id]["primary"],
                "Name": query_metadata["Name"],
                "BigSMILES": query_metadata["BigSMILES"],
                "BigSMILES_length": len(str(query_metadata["BigSMILES"])),
                "unique_primary_topology_count": len(unique_primary),
                "unique_topology_signature_count": len(unique_signatures),
                "primary_topology_composition": unique_primary,
                "topology_signature_composition": unique_signatures,
                "rank1_similarity": top5[0]["BigFingerprint"],
                "rank5_similarity": top5[4]["BigFingerprint"],
                "mean_top5_similarity": float(
                    np.mean([candidate["BigFingerprint"] for candidate in top5])
                ),
                "distinct_top5_score_count": len(
                    {round(candidate["BigFingerprint"], 12) for candidate in top5}
                ),
                "cutoff_tie_count": cutoff_tie_count,
                "cutoff_slots_available": cutoff_slots_available,
                "cutoff_is_ambiguous": cutoff_tie_count > cutoff_slots_available,
                "max_candidate_length": max(
                    len(candidate["BigSMILES"]) for candidate in top5
                ),
                "top5": top5,
            }
        )

    screened = [
        summary
        for summary in summaries
        if summary["BigSMILES_length"] <= 180
        and summary["max_candidate_length"] <= 220
        and summary["rank1_similarity"] < 1.0 - 1e-12
    ]
    screened.sort(
        key=lambda row: (
            -row["unique_primary_topology_count"],
            -row["unique_topology_signature_count"],
            row["cutoff_is_ambiguous"],
            row["cutoff_tie_count"],
            -row["distinct_top5_score_count"],
            -row["rank5_similarity"],
            -row["mean_top5_similarity"],
            row["BigSMILES_length"],
        )
    )

    categories = ["single", "block", "segmented", "graft", "star", "network", "other"]
    output = {
        "dataset": str(data_dir),
        "polymer_count": len(ids),
        "topology_counts": {
            category: sum(
                topology[polymer_id]["primary"] == category for polymer_id in ids
            )
            for category in categories
        },
        "global_top50": [compact(summary) for summary in screened[:50]],
        "top_by_query_topology": {
            category: [
                compact(summary)
                for summary in screened
                if summary["query_primary_topology"] == category
            ][:20]
            for category in categories
        },
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    summary_rows = []
    for summary in summaries:
        summary_rows.append(
            {
                key: "|".join(value) if isinstance(value, list) else value
                for key, value in summary.items()
                if key != "top5"
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    print(
        json.dumps(
            {
                "output": str(json_path),
                "summary_csv": str(summary_csv_path),
                "topology_counts": output["topology_counts"],
                "best_overall": [
                    {
                        "query_id": row["query_id"],
                        "query_topology": row["query_topology"],
                        "primary_topology_composition": row[
                            "primary_topology_composition"
                        ],
                        "rank1_similarity": row["rank1_similarity"],
                        "rank5_similarity": row["rank5_similarity"],
                        "mean_top5_similarity": row["mean_top5_similarity"],
                        "cutoff_is_ambiguous": row["cutoff_is_ambiguous"],
                    }
                    for row in output["global_top50"][:20]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

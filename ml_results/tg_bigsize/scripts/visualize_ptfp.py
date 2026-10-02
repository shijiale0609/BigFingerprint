#!/usr/bin/env python3
"""Visualize the corrected 40-dimensional p-TFP matrix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures"
METADATA_DIR = ROOT / "metadata"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["svg.fonttype"] = "none"

COLORMAP = ListedColormap(["white", "#2166ac"])


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    ptfp = np.load(DATA_DIR / "fingerprint_ptfp.npy")
    bigfp = np.load(DATA_DIR / "fingerprint_bigfp.npy", mmap_mode="r")
    cleaned = pd.read_csv(DATA_DIR / "cleaned_tg.csv")

    if ptfp.shape != (len(cleaned), 40):
        raise ValueError(
            f"Expected ({len(cleaned)}, 40) p-TFP matrix; found {ptfp.shape}"
        )
    if not np.array_equal(np.unique(ptfp), np.array([0, 1], dtype=ptfp.dtype)):
        raise ValueError("p-TFP is not a binary matrix containing both 0 and 1")
    if not np.array_equal(ptfp, np.asarray(bigfp[:, -40:])):
        raise ValueError("p-TFP does not match the last 40 BigFingerprint dimensions")

    dimensions = np.arange(1, 41)
    bit_counts = ptfp.sum(axis=0).astype(int)
    bit_prevalence = bit_counts / len(ptfp)
    pd.DataFrame(
        {
            "ptfp_dimension_1_based": dimensions,
            "samples_with_bit_1": bit_counts,
            "fraction_with_bit_1": bit_prevalence,
            "is_constant": (bit_counts == 0) | (bit_counts == len(ptfp)),
        }
    ).to_csv(METADATA_DIR / "ptfp_dimension_statistics.csv", index=False)

    unique_patterns, counts = np.unique(ptfp, axis=0, return_counts=True)
    order = np.argsort(-counts)
    unique_patterns = unique_patterns[order]
    counts = counts[order]

    pattern_rows: list[dict[str, object]] = []
    for pattern_number, (pattern, count) in enumerate(
        zip(unique_patterns, counts), start=1
    ):
        active = (np.flatnonzero(pattern) + 1).tolist()
        pattern_rows.append(
            {
                "pattern": pattern_number,
                "active_dimensions_1_based": ";".join(map(str, active)),
                "sample_count": int(count),
                "sample_fraction": float(count / len(ptfp)),
            }
        )
    pd.DataFrame(pattern_rows).to_csv(
        METADATA_DIR / "ptfp_unique_patterns.csv", index=False
    )

    # Group identical vectors together only for display. The stored matrix and
    # all machine-learning inputs retain their original cleaned_tg.csv row order.
    row_pattern = np.empty(len(ptfp), dtype=int)
    for pattern_number, pattern in enumerate(unique_patterns):
        row_pattern[np.all(ptfp == pattern, axis=1)] = pattern_number
    display_order = np.argsort(row_pattern, kind="stable")
    grouped = ptfp[display_order]
    boundaries = np.cumsum(counts)[:-1]

    figure, (matrix_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.2),
        gridspec_kw={"height_ratios": [3.2, 1.5], "hspace": 0.28},
    )
    matrix_axis.imshow(
        grouped,
        aspect="auto",
        interpolation="nearest",
        cmap=COLORMAP,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    for boundary in boundaries:
        matrix_axis.axhline(boundary - 0.5, color="black", linewidth=0.8)
    matrix_axis.set_xlim(-0.5, 39.5)
    matrix_axis.set_xticks(np.arange(0, 40, 2))
    matrix_axis.set_xticklabels(np.arange(1, 41, 2), fontsize=10)
    matrix_axis.set_ylabel("Polymers grouped by p-TFP pattern", fontsize=13)
    matrix_axis.set_title("p-TFP matrix: 7,157 polymers × 40 dimensions", fontsize=15)
    matrix_axis.tick_params(direction="in", top=True, right=True)

    count_axis.bar(
        dimensions,
        bit_counts,
        width=0.78,
        color="#2166ac",
        edgecolor="black",
        linewidth=0.6,
    )
    count_axis.set_xlim(0.25, 40.75)
    count_axis.set_xticks(np.arange(1, 41, 2))
    count_axis.set_xlabel("p-TFP dimension (1-based)", fontsize=13)
    count_axis.set_ylabel("Samples with bit = 1", fontsize=13)
    count_axis.tick_params(direction="in", top=True, right=True)
    for dimension, count in zip(dimensions, bit_counts):
        if count:
            count_axis.annotate(
                f"{count:,}",
                (dimension, count),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    count_axis.set_ylim(0, int(bit_counts.max() * 1.12))
    save_figure(figure, "ptfp_matrix_and_bit_counts")

    pattern_labels = []
    for pattern, count in zip(unique_patterns, counts):
        active = ", ".join(map(str, (np.flatnonzero(pattern) + 1).tolist()))
        pattern_labels.append(f"[{active}]  (n = {count:,})")
    pattern_figure, pattern_axis = plt.subplots(figsize=(9.0, 3.2))
    pattern_axis.imshow(
        unique_patterns,
        aspect="auto",
        interpolation="nearest",
        cmap=COLORMAP,
        vmin=0,
        vmax=1,
    )
    pattern_axis.set_xticks(np.arange(40))
    pattern_axis.set_xticklabels(dimensions, fontsize=8)
    pattern_axis.set_yticks(np.arange(len(unique_patterns)))
    pattern_axis.set_yticklabels(pattern_labels, fontsize=11)
    pattern_axis.set_xlabel("p-TFP dimension (1-based)", fontsize=13)
    pattern_axis.set_ylabel("Active dimensions", fontsize=13)
    pattern_axis.set_title("Unique p-TFP vectors in the Tg dataset", fontsize=15)
    pattern_axis.set_xticks(np.arange(-0.5, 40, 1), minor=True)
    pattern_axis.set_yticks(np.arange(-0.5, len(unique_patterns), 1), minor=True)
    pattern_axis.grid(which="minor", color="#d0d0d0", linewidth=0.35)
    pattern_axis.tick_params(which="minor", bottom=False, left=False)
    save_figure(pattern_figure, "ptfp_unique_pattern_heatmap")

    summary = {
        "matrix_shape": list(ptfp.shape),
        "dtype": str(ptfp.dtype),
        "binary_values": np.unique(ptfp).astype(int).tolist(),
        "nonzero_dimensions_1_based": (np.flatnonzero(bit_counts) + 1).tolist(),
        "nonconstant_dimensions_1_based": (
            np.flatnonzero((bit_counts > 0) & (bit_counts < len(ptfp))) + 1
        ).tolist(),
        "unique_vector_count": int(len(unique_patterns)),
        "matches_last_40_dimensions_of_bigfingerprint": True,
        "heatmap_row_order": "grouped by identical p-TFP vector for visualization only",
    }
    (METADATA_DIR / "ptfp_visualization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(pattern_rows).to_string(index=False))


if __name__ == "__main__":
    main()

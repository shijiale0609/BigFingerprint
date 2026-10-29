#!/usr/bin/env python3
"""Calculate Morgan, p-ECFP, p-TFP, and product BigFingerprint similarities.

The Morgan baseline intentionally represents only repeat-unit chemistry
available to a pSMILES-style fingerprint.  Random and block EO/PO candidates
use the same union of component fingerprints and therefore cannot be
distinguished.  Star-polymer architectures cannot be represented by this
baseline, so their Morgan fingerprints and pairwise similarities are recorded
as unavailable rather than approximated with an arm-only fingerprint.

BigFingerprint similarity is defined for this experiment as requested:

    S_BigFingerprint = S_p-ECFP * S_p-TFP
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from bigfingerprint import polymer_ecfp, polymer_topfp, search_tools

FP_SIZE = 2048
MORGAN_RADIUS = 2
PECFP_RADIUS = 2

OUTPUT_DIR = Path(__file__).resolve().parent / "vector_search_eight_polymers"
BLUE_OUTPUT_DIR = OUTPUT_DIR / "blue_individual_heatmaps"

POLYMERS = [
    {
        "id": "L-EO",
        "label": "Linear CCO",
        "bigsmiles": "{[][<]CCO[>][]}",
        "morgan_fragments": ["*CCO*"],
    },
    {
        "id": "L-PO",
        "label": "Linear CC(C)O",
        "bigsmiles": "{[][<]CC(C)O[>][]}",
        "morgan_fragments": ["*CC(C)O*"],
    },
    {
        "id": "R-EOPO",
        "label": "Random CCO/CC(C)O",
        "bigsmiles": "{[][<]CCO[>],[<]CC(C)O[>][]}",
        "morgan_fragments": ["*CCO*", "*CC(C)O*"],
    },
    {
        "id": "B-EOPO",
        "label": "Block CCO/CC(C)O",
        "bigsmiles": "{[][<]CCO[>][<]}{[>][<]CC(C)O[>][]}",
        "morgan_fragments": ["*CCO*", "*CC(C)O*"],
    },
    {
        "id": "S3-EO",
        "label": "3-arm CCO star",
        "bigsmiles": (
            "C(CO{[<][>]CCO[<][>]})(CO{[<][>]CCO[<][>]})" "CO{[<][>]CCO[<][>]}"
        ),
        "morgan_fragments": None,
    },
    {
        "id": "S4-EO",
        "label": "4-arm CCO star",
        "bigsmiles": (
            "C(CO{[<][>]CCO[<][>]})(CO{[<][>]CCO[<][>]})"
            "(CO{[<][>]CCO[<][>]})CO{[<][>]CCO[<][>]}"
        ),
        "morgan_fragments": None,
    },
    {
        "id": "S3-PO",
        "label": "3-arm CC(C)O star",
        "bigsmiles": (
            "C(CO{[<][>]CC(C)O[<][>]})(CO{[<][>]CC(C)O[<][>]})" "CO{[<][>]CC(C)O[<][>]}"
        ),
        "morgan_fragments": None,
    },
    {
        "id": "S4-PO",
        "label": "4-arm CC(C)O star",
        "bigsmiles": (
            "C(CO{[<][>]CC(C)O[<][>]})(CO{[<][>]CC(C)O[<][>]})"
            "(CO{[<][>]CC(C)O[<][>]})CO{[<][>]CC(C)O[<][>]}"
        ),
        "morgan_fragments": None,
    },
]


def bit_vector_to_array(bit_vector, size: int) -> np.ndarray:
    array = np.zeros(size, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def morgan_union_fingerprint(fragment_smiles: list[str]) -> np.ndarray:
    """Return the OR-union Morgan fingerprint of unique repeat-unit fragments."""
    generator = GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=FP_SIZE,
        includeChirality=False,
    )
    union = np.zeros(FP_SIZE, dtype=np.uint8)
    for smiles in fragment_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse Morgan fragment: {smiles}")
        union |= bit_vector_to_array(generator.GetFingerprint(mol), FP_SIZE)
    return union


def bigsmiles_fingerprints(bigsmiles: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate p-ECFP and p-TFP using the repository implementation."""
    # The research code prints graph diagnostics; keep the reproducible run log concise.
    with contextlib.redirect_stdout(io.StringIO()):
        graphs = search_tools.generate_NetworkX_graphs(
            bigsmiles,
            is_bigsmarts=False,
        )
        _cycles, clusters, _treat_as_cycle = search_tools.identify_cycles(graphs)
        pecfp = polymer_ecfp.ECFP(
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
        ptfp = polymer_topfp.bigsmiles_topology_vector(bigsmiles)
    return np.asarray(pecfp, dtype=np.uint8), np.asarray(ptfp, dtype=np.uint8)


def tanimoto_matrix(fingerprints: np.ndarray) -> np.ndarray:
    fingerprints = np.asarray(fingerprints, dtype=bool)
    intersections = fingerprints.astype(np.uint16) @ fingerprints.T.astype(np.uint16)
    counts = fingerprints.sum(axis=1, dtype=np.uint16)
    unions = counts[:, None] + counts[None, :] - intersections
    return np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=float),
        where=unions != 0,
    )


def save_matrix(matrix: np.ndarray, stem: str, ids: list[str]) -> None:
    frame = pd.DataFrame(matrix, index=ids, columns=ids)
    frame.index.name = "polymer_id"
    frame.to_csv(
        OUTPUT_DIR / f"{stem}.csv",
        float_format="%.6f",
        na_rep="N/A",
    )
    np.save(OUTPUT_DIR / f"{stem}.npy", matrix)


def plot_heatmaps(matrices: list[tuple[str, np.ndarray]], ids: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.0), constrained_layout=True)
    panel_letters = ["a", "b", "c", "d"]
    image = None

    for axis, panel_letter, (title, matrix) in zip(
        axes.flat,
        panel_letters,
        matrices,
        strict=True,
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title, pad=10)
        axis.text(
            -0.14,
            1.07,
            panel_letter,
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )
        axis.set_xticks(range(len(ids)), labels=ids, rotation=45, ha="right")
        axis.set_yticks(range(len(ids)), labels=ids)
        axis.set_xlabel("Candidate polymer")
        axis.set_ylabel("Query polymer")

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.68 else "black",
                    fontsize=8,
                )

        axis.set_xticks(np.arange(-0.5, len(ids), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(ids), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.6, alpha=0.55)
        axis.tick_params(which="minor", bottom=False, left=False)

    if image is None:
        raise RuntimeError("No heatmaps were plotted")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Tanimoto similarity")
    fig.suptitle(
        "Fingerprint Similarity Across Eight Polymer Architectures",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "eight_polymer_similarity_heatmaps.png", dpi=400)
    fig.savefig(OUTPUT_DIR / "eight_polymer_similarity_heatmaps.pdf")
    fig.savefig(OUTPUT_DIR / "eight_polymer_similarity_heatmaps.svg")
    plt.close(fig)


def plot_blue_heatmap(
    matrix: np.ndarray,
    title: str,
    output_stem: str,
    colorbar_label: str,
) -> None:
    """Write one standalone blue-format heatmap using sequential P labels."""
    polymer_labels = [f"P{index}" for index in range(1, matrix.shape[0] + 1)]
    figure_width = 7.4 if matrix.shape[0] <= 8 else 8.2
    figure_height = 6.8 if matrix.shape[0] <= 8 else 7.4
    cell_font_size = 12 if matrix.shape[0] <= 8 else 10
    tick_font_size = 14 if matrix.shape[0] <= 8 else 12
    axis_label_font_size = 17 if matrix.shape[0] <= 8 else 15
    colorbar_tick_font_size = 13 if matrix.shape[0] <= 8 else 12
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    blue_cmap = plt.get_cmap("Blues")
    normalization = Normalize(vmin=0, vmax=1)

    # Draw each matrix entry as an explicit, clipped rectangle. This gives every
    # similarity value one uniform cell color and prevents interpolation or
    # antialiasing from creating shading beyond the cell boundaries.
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            facecolor = (
                "#d9d9d9" if np.isnan(value) else blue_cmap(normalization(value))
            )
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=0.8,
                    antialiased=False,
                    clip_on=True,
                )
            )

    axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_title(title, pad=12)
    axis.set_xticks(range(len(polymer_labels)), labels=polymer_labels)
    axis.set_yticks(range(len(polymer_labels)), labels=polymer_labels)
    axis.set_xlabel("Polymer", fontsize=axis_label_font_size, labelpad=10)
    axis.set_ylabel("Polymer", fontsize=axis_label_font_size, labelpad=12)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=tick_font_size,
        length=0,
        pad=7,
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                label = "N/A"
                text_color = "#555555"
            else:
                label = f"{value:.2f}"
                text_color = "white" if value >= 0.58 else "black"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=cell_font_size,
            )

    for spine in axis.spines.values():
        spine.set_visible(False)

    color_scale = plt.cm.ScalarMappable(norm=normalization, cmap=blue_cmap)
    color_scale.set_array([])
    colorbar = fig.colorbar(color_scale, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(
        colorbar_label,
        fontsize=axis_label_font_size,
        labelpad=12,
    )
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.ax.tick_params(labelsize=colorbar_tick_font_size, width=1.0, length=5)
    colorbar.outline.set_linewidth(0.8)

    for extension in ("png", "pdf", "svg"):
        destination = BLUE_OUTPUT_DIR / f"{output_stem}.{extension}"
        if extension == "png":
            fig.savefig(destination, dpi=400, bbox_inches="tight")
        else:
            fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BLUE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = [polymer["id"] for polymer in POLYMERS]

    morgan_vectors = np.full((len(POLYMERS), FP_SIZE), np.nan, dtype=float)
    pecfp_vectors = []
    ptfp_vectors = []
    morgan_supported_indices = []
    for index, polymer in enumerate(POLYMERS):
        if polymer["morgan_fragments"] is not None:
            morgan_vectors[index] = morgan_union_fingerprint(
                polymer["morgan_fragments"]
            )
            morgan_supported_indices.append(index)
        pecfp, ptfp = bigsmiles_fingerprints(polymer["bigsmiles"])
        pecfp_vectors.append(pecfp)
        ptfp_vectors.append(ptfp)

    pecfp_vectors = np.stack(pecfp_vectors)
    ptfp_vectors = np.stack(ptfp_vectors)

    morgan_similarity = np.full((len(POLYMERS), len(POLYMERS)), np.nan)
    supported_morgan_similarity = tanimoto_matrix(
        morgan_vectors[morgan_supported_indices]
    )
    morgan_similarity[np.ix_(morgan_supported_indices, morgan_supported_indices)] = (
        supported_morgan_similarity
    )
    pecfp_similarity = tanimoto_matrix(pecfp_vectors)
    ptfp_similarity = tanimoto_matrix(ptfp_vectors)
    bigfingerprint_similarity = pecfp_similarity * ptfp_similarity

    save_matrix(morgan_similarity, "morgan_tanimoto", ids)
    save_matrix(pecfp_similarity, "pecfp_tanimoto", ids)
    save_matrix(ptfp_similarity, "ptfp_tanimoto", ids)
    save_matrix(bigfingerprint_similarity, "bigfingerprint_product_similarity", ids)

    np.save(OUTPUT_DIR / "morgan_fingerprints.npy", morgan_vectors)
    np.save(OUTPUT_DIR / "pecfp_fingerprints.npy", pecfp_vectors)
    np.save(OUTPUT_DIR / "ptfp_fingerprints.npy", ptfp_vectors)

    definitions = pd.DataFrame(
        [
            {
                "id": polymer["id"],
                "label": polymer["label"],
                "BigSMILES": polymer["bigsmiles"],
                "Morgan repeat-unit fragments": (
                    "; ".join(polymer["morgan_fragments"])
                    if polymer["morgan_fragments"] is not None
                    else "N/A (star architecture unsupported)"
                ),
                "Morgan active bits": (
                    int(np.nansum(morgan_vectors[index]))
                    if polymer["morgan_fragments"] is not None
                    else pd.NA
                ),
                "p-ECFP active bits": int(pecfp_vectors[index].sum()),
                "p-TFP active bits": int(ptfp_vectors[index].sum()),
            }
            for index, polymer in enumerate(POLYMERS)
        ]
    )
    definitions.to_csv(
        OUTPUT_DIR / "polymer_definitions_and_bit_counts.csv", index=False
    )

    metadata = {
        "definition": "BigFingerprint similarity = p-ECFP Tanimoto * p-TFP Tanimoto",
        "morgan_baseline": (
            "OR-union of Morgan fingerprints for distinct pSMILES-style repeat-unit "
            "fragments. Random and block sequences have the same fingerprint. "
            "Star architectures are unsupported and recorded as N/A."
        ),
        "morgan_supported_polymer_ids": [
            ids[index] for index in morgan_supported_indices
        ],
        "morgan_radius": MORGAN_RADIUS,
        "morgan_fp_size": FP_SIZE,
        "pecfp_radius": PECFP_RADIUS,
        "pecfp_fp_size": FP_SIZE,
        "ptfp_size": int(ptfp_vectors.shape[1]),
        "rdkit_version": rdBase.rdkitVersion,
        "polymer_ids": ids,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    p_labels = [f"P{index}" for index in range(1, len(POLYMERS) + 1)]
    pd.DataFrame(
        {
            "polymer_id": p_labels,
            "label": [polymer["label"] for polymer in POLYMERS],
            "BigSMILES": [polymer["bigsmiles"] for polymer in POLYMERS],
        }
    ).to_csv(
        BLUE_OUTPUT_DIR / f"p1_p{len(POLYMERS)}_mapping.csv",
        index=False,
    )

    plot_blue_heatmap(
        morgan_similarity,
        title="Morgan Fingerprint",
        output_stem="morgan_tanimoto_blue",
        colorbar_label="Tanimoto similarity",
    )
    plot_blue_heatmap(
        pecfp_similarity,
        title="p-ECFP",
        output_stem="pecfp_tanimoto_blue",
        colorbar_label="Tanimoto similarity",
    )
    plot_blue_heatmap(
        ptfp_similarity,
        title="p-TFP",
        output_stem="ptfp_tanimoto_blue",
        colorbar_label="Tanimoto similarity",
    )
    plot_blue_heatmap(
        bigfingerprint_similarity,
        title="BigFingerprint",
        output_stem="bigfingerprint_product_similarity_blue",
        colorbar_label="Product similarity",
    )

    print(f"Wrote results to {OUTPUT_DIR}")
    print(definitions.to_string(index=False))


if __name__ == "__main__":
    main()

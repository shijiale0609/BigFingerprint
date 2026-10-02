#!/usr/bin/env python3
"""Plot representative BigFingerprint Top-5 vector-search results.

The candidate order is determined solely by BigFingerprint similarity.  The
component p-ECFP and p-TFP Tanimoto similarities are displayed for context.
The small drawings summarize p-TFP topology categories; they are not chemical
structure depictions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "results_475_endgroup_corrected_deduplicated"
INPUT_JSON = DATA_DIR / "query_selection_candidates_475.json"
OUTPUT_DIR = DATA_DIR / "ranked_query_figure"
SELECTED_QUERIES = ["P330", "P410", "P423", "P25"]

TOPOLOGY_LABELS = {
    "single": "Single stochastic object",
    "block": "Block",
    "segmented": "Segmented",
    "graft": "Graft",
    "star": "Star",
    "network": "Network",
    "block+star": "Block + star",
    "block+star+network": "Block + star + network",
}

INK = "#1F2933"
MUTED = "#52606D"
LIGHT_BORDER = "#9FB3C8"
NAVY = "#174A7E"
BLUE = "#2F80C1"
ORANGE = "#E58A3A"
GREEN = "#3A9D78"
PURPLE = "#8265B3"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_query_rows(data: dict) -> list[dict]:
    by_id: dict[str, dict] = {}
    for row in data.get("global_top50", []):
        by_id[row["query_id"]] = row
    for rows in data.get("top_by_query_topology", {}).values():
        for row in rows:
            by_id[row["query_id"]] = row
    missing = [query_id for query_id in SELECTED_QUERIES if query_id not in by_id]
    if missing:
        raise KeyError(f"Selected query rows not found: {missing}")
    return [by_id[query_id] for query_id in SELECTED_QUERIES]


def topology_label(signature: str) -> str:
    return TOPOLOGY_LABELS.get(signature, signature.replace("+", " + ").title())


def star_arm_count(bigsmiles: str) -> int:
    prefix = bigsmiles.split(".", 1)[0]
    count = len(re.findall(r"\[#(?:Arm|R)\]", prefix))
    return min(max(count, 3), 5) if count else 4


def draw_chain(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    colors_: list[str],
    alternating: bool = False,
) -> None:
    n = 8
    xs = np.linspace(x + 0.12 * width, x + 0.88 * width, n)
    ys = y + height * (0.50 + 0.09 * np.sin(np.arange(n) * np.pi))
    ax.plot(xs, ys, color=INK, lw=1.35, solid_capstyle="round", zorder=2)
    for index, (node_x, node_y) in enumerate(zip(xs, ys)):
        if alternating:
            node_color = colors_[index % len(colors_)]
        else:
            block_index = min(len(colors_) - 1, int(index * len(colors_) / n))
            node_color = colors_[block_index]
        ax.add_patch(
            Circle(
                (node_x, node_y),
                radius=0.045 * height,
                facecolor=node_color,
                edgecolor="white",
                lw=0.6,
                zorder=3,
            )
        )


def draw_graft(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    n = 7
    xs = np.linspace(x + 0.12 * width, x + 0.88 * width, n)
    backbone_y = y + 0.45 * height
    ax.plot(xs, np.full(n, backbone_y), color=INK, lw=1.4, zorder=2)
    for index, node_x in enumerate(xs):
        ax.add_patch(
            Circle(
                (node_x, backbone_y),
                radius=0.043 * height,
                facecolor=BLUE,
                edgecolor="white",
                lw=0.55,
                zorder=3,
            )
        )
        if index in (1, 3, 5):
            branch_y = y + (0.76 if index != 3 else 0.82) * height
            ax.plot([node_x, node_x], [backbone_y, branch_y], color=INK, lw=1.15)
            for fraction in (0.68, 1.0):
                this_y = backbone_y + (branch_y - backbone_y) * fraction
                ax.add_patch(
                    Circle(
                        (node_x, this_y),
                        radius=0.039 * height,
                        facecolor=ORANGE,
                        edgecolor="white",
                        lw=0.5,
                        zorder=3,
                    )
                )


def draw_star(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    arms: int,
    block: bool,
) -> None:
    center = np.array([x + 0.50 * width, y + 0.50 * height])
    radius = min(width, height) * 0.36
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, arms, endpoint=False)
    for angle in angles:
        end = center + radius * np.array([np.cos(angle), np.sin(angle)])
        midpoint = center + 0.55 * (end - center)
        ax.plot([center[0], end[0]], [center[1], end[1]], color=INK, lw=1.25)
        ax.add_patch(
            Circle(
                tuple(midpoint),
                radius=0.035 * height,
                facecolor=ORANGE if block else BLUE,
                edgecolor="white",
                lw=0.5,
                zorder=3,
            )
        )
        ax.add_patch(
            Circle(
                tuple(end),
                radius=0.040 * height,
                facecolor=GREEN if block else BLUE,
                edgecolor="white",
                lw=0.5,
                zorder=3,
            )
        )
    ax.add_patch(
        Circle(
            tuple(center),
            radius=0.060 * height,
            facecolor=PURPLE,
            edgecolor="white",
            lw=0.7,
            zorder=4,
        )
    )


def draw_network(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    points = np.array(
        [
            [0.22, 0.28],
            [0.50, 0.18],
            [0.78, 0.31],
            [0.72, 0.72],
            [0.42, 0.82],
            [0.18, 0.61],
            [0.49, 0.51],
        ]
    )
    points[:, 0] = x + width * points[:, 0]
    points[:, 1] = y + height * points[:, 1]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (0, 6),
        (2, 6),
        (4, 6),
        (5, 6),
        (3, 6),
    ]
    for left, right in edges:
        ax.plot(
            points[[left, right], 0],
            points[[left, right], 1],
            color=INK,
            lw=1.05,
            zorder=2,
        )
    node_colors = [BLUE, ORANGE, GREEN, BLUE, ORANGE, GREEN, PURPLE]
    for point, node_color in zip(points, node_colors):
        ax.add_patch(
            Circle(
                tuple(point),
                radius=0.039 * height,
                facecolor=node_color,
                edgecolor="white",
                lw=0.55,
                zorder=3,
            )
        )


def draw_topology_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    signature: str,
    bigsmiles: str,
) -> None:
    categories = signature.split("+")
    if "network" in categories:
        draw_network(ax, x, y, width, height)
    elif "star" in categories:
        draw_star(
            ax,
            x,
            y,
            width,
            height,
            arms=star_arm_count(bigsmiles),
            block="block" in categories,
        )
    elif "graft" in categories:
        draw_graft(ax, x, y, width, height)
    elif "segmented" in categories:
        draw_chain(ax, x, y, width, height, [BLUE, ORANGE, GREEN], alternating=True)
    elif "block" in categories:
        draw_chain(ax, x, y, width, height, [BLUE, ORANGE, GREEN])
    else:
        draw_chain(ax, x, y, width, height, [BLUE])


def readable_text_color(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else INK


def draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    polymer_id: str,
    signature: str,
    bigsmiles: str,
    rank: int | None,
    pecfp: float | None = None,
    ptfp: float | None = None,
    bigfingerprint: float | None = None,
    cmap: mpl.colors.Colormap | None = None,
    norm: mpl.colors.Normalize | None = None,
) -> None:
    is_query = rank is None
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="#F5F7FA" if is_query else "white",
        edgecolor=NAVY if is_query else LIGHT_BORDER,
        linewidth=1.7 if is_query else 1.15,
        zorder=1,
    )
    ax.add_patch(card)

    ribbon_height = 0.18 * height
    if is_query:
        ribbon_color = NAVY
        ribbon_text_color = "white"
        ribbon_text = f"Query {polymer_id}"
    else:
        assert cmap is not None and norm is not None and bigfingerprint is not None
        ribbon_color = cmap(norm(bigfingerprint))
        ribbon_text_color = readable_text_color(ribbon_color)
        ribbon_text = f"Rank {rank}   {polymer_id}"
    ribbon = Rectangle(
        (x, y + height - ribbon_height),
        width,
        ribbon_height,
        facecolor=ribbon_color,
        edgecolor="none",
        zorder=2,
    )
    ribbon.set_clip_path(card)
    ax.add_patch(ribbon)
    ax.text(
        x + width / 2,
        y + height - ribbon_height / 2,
        ribbon_text,
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color=ribbon_text_color,
        zorder=3,
    )

    icon_y = y + (0.39 if is_query else 0.47) * height
    icon_h = (0.35 if is_query else 0.29) * height
    draw_topology_icon(
        ax,
        x + 0.08 * width,
        icon_y,
        0.84 * width,
        icon_h,
        signature,
        bigsmiles,
    )
    ax.text(
        x + width / 2,
        y + (0.34 if is_query else 0.42) * height,
        topology_label(signature),
        ha="center",
        va="center",
        fontsize=9.4,
        fontstyle="italic",
        color=MUTED,
    )

    if is_query:
        ax.text(
            x + width / 2,
            y + 0.145 * height,
            "Search database:\n475 unique polymers",
            ha="center",
            va="center",
            fontsize=9.6,
            color=INK,
            linespacing=1.15,
        )
    else:
        score_y = [0.295, 0.205, 0.105]
        labels = ["p-ECFP", "p-TFP", "BigFingerprint"]
        values = [pecfp, ptfp, bigfingerprint]
        for label, value, y_fraction in zip(labels, values, score_y):
            assert value is not None
            is_final = label == "BigFingerprint"
            ax.text(
                x + 0.08 * width,
                y + y_fraction * height,
                label,
                ha="left",
                va="center",
                fontsize=9.3 if not is_final else 9.6,
                fontweight="bold" if is_final else "normal",
                color=INK,
            )
            ax.text(
                x + 0.92 * width,
                y + y_fraction * height,
                f"{value:.3f}",
                ha="right",
                va="center",
                fontsize=10.0 if not is_final else 10.4,
                fontweight="bold" if is_final else "normal",
                color=INK,
            )
        ax.plot(
            [x + 0.08 * width, x + 0.92 * width],
            [y + 0.155 * height, y + 0.155 * height],
            color="#D9E2EC",
            lw=0.8,
        )


def export_selected_table(query_rows: list[dict]) -> pd.DataFrame:
    records = []
    for query in query_rows:
        for candidate in query["top5"]:
            records.append(
                {
                    "query_id": query["query_id"],
                    "query_topology": query["query_topology"],
                    "query_bigsmiles": query["BigSMILES"],
                    "rank_by_bigfingerprint": candidate["rank"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_topology": candidate["topology"],
                    "candidate_bigsmiles": candidate["BigSMILES"],
                    "p_ecfp_tanimoto": candidate["pECFP"],
                    "p_tfp_tanimoto": candidate["pTFP"],
                    "bigfingerprint_similarity": candidate["BigFingerprint"],
                }
            )
    table = pd.DataFrame.from_records(records)
    table.to_csv(OUTPUT_DIR / "selected_query_top5_475.csv", index=False)
    return table


def validate_rankings(table: pd.DataFrame) -> None:
    for query_id, group in table.groupby("query_id", sort=False):
        ranks = group["rank_by_bigfingerprint"].tolist()
        if ranks != [1, 2, 3, 4, 5]:
            raise ValueError(f"Unexpected ranks for {query_id}: {ranks}")
        scores = group["bigfingerprint_similarity"].to_numpy()
        if np.any(scores[:-1] < scores[1:] - 1e-12):
            raise ValueError(f"BigFingerprint scores are not descending for {query_id}")
        products = group["p_ecfp_tanimoto"] * group["p_tfp_tanimoto"]
        if not np.allclose(products, scores, rtol=0.0, atol=1e-12):
            raise ValueError(f"Product definition failed for {query_id}")


def make_figure(query_rows: list[dict]) -> plt.Figure:
    panel_count = len(query_rows)
    figure_height = 10.8 * panel_count / 3
    fig = plt.figure(figsize=(19.2, figure_height), facecolor="white")
    ax = fig.add_axes([0.025, 0.045, 0.95, 0.925])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cmap = mpl.colormaps["Blues"]
    norm = colors.Normalize(vmin=0.0, vmax=0.75)
    x_positions = [0.02, 0.205, 0.365, 0.525, 0.685, 0.845]
    query_width = 0.14
    candidate_width = 0.135
    vertical_scale = 3 / panel_count
    card_height = 0.255 * vertical_scale
    row_tops = np.linspace(0.93, 0.19, panel_count)
    row_bottoms = [float(top - card_height) for top in row_tops]

    ax.text(
        x_positions[0] + query_width / 2,
        0.982,
        "Query",
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        (x_positions[1] + x_positions[-1] + candidate_width) / 2,
        0.982,
        "Top-five candidates ranked only by BigFingerprint similarity",
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
    )

    for panel_index, (query, y) in enumerate(zip(query_rows, row_bottoms)):
        panel_letter = chr(ord("a") + panel_index)
        ax.text(
            0.002,
            y + card_height - 0.004,
            panel_letter,
            ha="left",
            va="top",
            fontsize=17,
            fontweight="bold",
            color=INK,
        )
        draw_card(
            ax,
            x_positions[0],
            y,
            query_width,
            card_height,
            polymer_id=query["query_id"],
            signature=query["query_topology"],
            bigsmiles=query["BigSMILES"],
            rank=None,
        )
        arrow_y = y + 0.53 * card_height
        ax.add_patch(
            FancyArrowPatch(
                (x_positions[0] + query_width + 0.009, arrow_y),
                (x_positions[1] - 0.009, arrow_y),
                arrowstyle="-|>",
                mutation_scale=15,
                color=NAVY,
                lw=1.5,
            )
        )
        for candidate, x in zip(query["top5"], x_positions[1:]):
            draw_card(
                ax,
                x,
                y,
                candidate_width,
                card_height,
                polymer_id=candidate["candidate_id"],
                signature=candidate["topology"],
                bigsmiles=candidate["BigSMILES"],
                rank=candidate["rank"],
                pecfp=candidate["pECFP"],
                ptfp=candidate["pTFP"],
                bigfingerprint=candidate["BigFingerprint"],
                cmap=cmap,
                norm=norm,
            )
        if panel_index < len(query_rows) - 1:
            ax.plot(
                [0.015, 0.985],
                [y - 0.016, y - 0.016],
                color="#D9E2EC",
                lw=0.9,
            )

    colorbar_ax = fig.add_axes([0.625, 0.010, 0.23, 0.009])
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.25, 0.50, 0.75])
    colorbar.ax.tick_params(labelsize=9.5, length=3, width=0.8, colors=INK)
    colorbar.set_label(
        "BigFingerprint similarity", fontsize=10.5, color=INK, labelpad=2
    )
    colorbar.outline.set_edgecolor(LIGHT_BORDER)
    colorbar.outline.set_linewidth(0.8)
    fig.text(
        0.027,
        0.014,
        "Topology schematics summarize p-TFP architecture classes; they are not chemical structure drawings.",
        ha="left",
        va="center",
        fontsize=9.8,
        color=MUTED,
    )
    return fig


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    query_rows = find_query_rows(data)
    table = export_selected_table(query_rows)
    validate_rankings(table)

    figure = make_figure(query_rows)
    base = OUTPUT_DIR / "bigfingerprint_top5_ranked_queries_475"
    figure.savefig(
        base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    panel_query_labels = [
        f"({chr(ord('a') + index)}) {query_id}"
        for index, query_id in enumerate(SELECTED_QUERIES)
    ]
    panel_queries = (
        panel_query_labels[0]
        if len(panel_query_labels) == 1
        else ", ".join(panel_query_labels[:-1]) + f", and {panel_query_labels[-1]}"
    )
    caption = (
        "Representative vector searches in the deduplicated 475-polymer BigSMILES "
        f"database using {panel_queries} as queries. Candidate order is determined "
        "solely by BigFingerprint similarity. The component p-ECFP and p-TFP "
        "Tanimoto similarities are shown separately, with BigFingerprint similarity "
        "calculated as their product. The selected queries yield top-five lists "
        "spanning single stochastic objects, block, graft, star, and network "
        "architectures. Topology schematics summarize p-TFP architecture classes "
        "and are not chemical structure drawings."
    )
    (OUTPUT_DIR / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")

    summary = {
        "selected_queries": SELECTED_QUERIES,
        "database_size": int(data["polymer_count"]),
        "ranking_metric": "BigFingerprint similarity only",
        "displayed_metrics": ["p-ECFP Tanimoto", "p-TFP Tanimoto", "BigFingerprint"],
        "outputs": [
            str(base.with_suffix(".png")),
            str(base.with_suffix(".pdf")),
            str(base.with_suffix(".svg")),
            str(OUTPUT_DIR / "selected_query_top5_475.csv"),
            str(OUTPUT_DIR / "figure_caption.txt"),
        ],
    }
    (OUTPUT_DIR / "selection_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

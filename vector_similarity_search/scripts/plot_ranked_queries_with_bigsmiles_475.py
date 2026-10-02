#!/usr/bin/env python3
"""Create a readable audit figure with every query and candidate BigSMILES."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch, Rectangle

import plot_ranked_queries_475 as base

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "results_475_endgroup_corrected_deduplicated"
DEFAULT_QUERIES = ["P330", "P410", "P423", "P25"]


def wrap_bigsmiles(value: str, width: int) -> str:
    wrapped = "\n".join(
        textwrap.wrap(
            value,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
    )
    # Matplotlib otherwise interprets BigSMILES `$` bonding descriptors as
    # math-text delimiters.
    return wrapped.replace("$", r"\$")


def score_color(cmap, norm, value: float):
    return cmap(norm(value))


def score_text_color(rgba) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.50 else base.INK


def add_cell_text(
    ax,
    x,
    y,
    text,
    *,
    ha="left",
    va="center",
    fontsize=9.0,
    color=base.INK,
    fontweight="normal",
    fontstyle="normal",
    linespacing=1.05,
):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        fontweight=fontweight,
        fontstyle=fontstyle,
        linespacing=linespacing,
    )


def draw_query_header(
    ax, query, y_top: float, panel_letter: str, scale: float
) -> float:
    height = 0.061 * scale
    y = y_top - height
    ax.add_patch(
        FancyBboxPatch(
            (0.014, y),
            0.972,
            height,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor="#EDF4FA",
            edgecolor=base.NAVY,
            linewidth=1.25,
        )
    )
    ax.add_patch(
        Rectangle((0.014, y), 0.152, height, facecolor=base.NAVY, edgecolor="none")
    )
    add_cell_text(
        ax,
        0.027,
        y + height * 0.67,
        panel_letter,
        fontsize=16,
        color="white",
        fontweight="bold",
    )
    add_cell_text(
        ax,
        0.058,
        y + height * 0.67,
        f"Query {query['query_id']}",
        fontsize=13.2,
        color="white",
        fontweight="bold",
    )
    add_cell_text(
        ax,
        0.090,
        y + height * 0.27,
        base.topology_label(query["query_topology"]),
        ha="center",
        fontsize=9.7,
        color="white",
        fontstyle="italic",
    )
    add_cell_text(
        ax,
        0.180,
        y + height * 0.77,
        "BigSMILES",
        fontsize=9.6,
        color=base.MUTED,
        fontweight="bold",
    )
    add_cell_text(
        ax,
        0.180,
        y + height * 0.38,
        wrap_bigsmiles(query["BigSMILES"], 132),
        fontsize=9.4,
        color=base.INK,
        linespacing=1.08,
    )
    return y


def draw_column_header(ax, y_top: float, scale: float) -> float:
    height = 0.024 * scale
    y = y_top - height
    ax.add_patch(
        Rectangle((0.014, y), 0.972, height, facecolor="#D9E2EC", edgecolor="none")
    )
    headers = [
        (0.026, "Rank / ID", "left"),
        (0.122, "p-TFP topology", "left"),
        (0.292, "Candidate BigSMILES", "left"),
        (0.830, "p-ECFP", "center"),
        (0.891, "p-TFP", "center"),
        (0.952, "BigFingerprint", "center"),
    ]
    for x, label, alignment in headers:
        add_cell_text(
            ax,
            x,
            y + height / 2,
            label,
            ha=alignment,
            fontsize=9.7,
            color=base.INK,
            fontweight="bold",
        )
    return y


def draw_candidate_row(
    ax, candidate, y_top: float, row_index: int, cmap, norm, scale: float
) -> float:
    height = 0.044 * scale
    y = y_top - height
    background = "#FAFCFE" if row_index % 2 else "white"
    ax.add_patch(
        Rectangle(
            (0.014, y),
            0.972,
            height,
            facecolor=background,
            edgecolor="#CBD5E1",
            linewidth=0.65,
        )
    )
    for x in (0.105, 0.280, 0.799, 0.860, 0.922):
        ax.plot([x, x], [y, y + height], color="#D9E2EC", lw=0.65)

    add_cell_text(
        ax,
        0.026,
        y + height * 0.63,
        f"Rank {candidate['rank']}",
        fontsize=10.2,
        fontweight="bold",
    )
    add_cell_text(
        ax,
        0.026,
        y + height * 0.27,
        candidate["candidate_id"],
        fontsize=10.7,
        color=base.NAVY,
        fontweight="bold",
    )

    base.draw_topology_icon(
        ax,
        0.115,
        y + height * 0.18,
        0.067,
        height * 0.65,
        candidate["topology"],
        candidate["BigSMILES"],
    )
    add_cell_text(
        ax,
        0.192,
        y + height / 2,
        base.topology_label(candidate["topology"]),
        fontsize=8.8,
        color=base.MUTED,
        fontstyle="italic",
    )

    add_cell_text(
        ax,
        0.292,
        y + height / 2,
        wrap_bigsmiles(candidate["BigSMILES"], 92),
        fontsize=8.5,
        color=base.INK,
        linespacing=1.02,
    )
    add_cell_text(
        ax,
        0.830,
        y + height / 2,
        f"{candidate['pECFP']:.3f}",
        ha="center",
        fontsize=10.4,
    )
    add_cell_text(
        ax,
        0.891,
        y + height / 2,
        f"{candidate['pTFP']:.3f}",
        ha="center",
        fontsize=10.4,
    )

    final_color = score_color(cmap, norm, candidate["BigFingerprint"])
    ax.add_patch(
        Rectangle((0.922, y), 0.064, height, facecolor=final_color, edgecolor="none")
    )
    add_cell_text(
        ax,
        0.954,
        y + height / 2,
        f"{candidate['BigFingerprint']:.3f}",
        ha="center",
        fontsize=10.7,
        color=score_text_color(final_color),
        fontweight="bold",
    )
    return y


def make_figure(query_rows):
    panel_count = len(query_rows)
    vertical_scale = 3 / panel_count
    figure_height = 14.5 * panel_count / 3
    figure = plt.figure(figsize=(20.0, figure_height), facecolor="white")
    ax = figure.add_axes([0.02, 0.035, 0.96, 0.95])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cmap = mpl.colormaps["Blues"]
    norm = colors.Normalize(vmin=0.0, vmax=0.75)
    panel_height = (0.061 + 0.005 + 0.024 + 5 * 0.044) * vertical_scale
    panel_gap = max(0.003, (0.945 - panel_count * panel_height) / (panel_count - 1))
    panel_tops = [
        0.982 - index * (panel_height + panel_gap) for index in range(panel_count)
    ]
    for panel_index, (query, panel_top) in enumerate(zip(query_rows, panel_tops)):
        y = draw_query_header(
            ax,
            query,
            panel_top,
            chr(ord("a") + panel_index),
            vertical_scale,
        )
        y = draw_column_header(ax, y - 0.005 * vertical_scale, vertical_scale)
        for row_index, candidate in enumerate(query["top5"], start=1):
            y = draw_candidate_row(
                ax, candidate, y, row_index, cmap, norm, vertical_scale
            )

    colorbar_ax = figure.add_axes([0.742, 0.010, 0.22, 0.009])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(sm, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.25, 0.50, 0.75])
    colorbar.ax.tick_params(labelsize=9.5, length=3, width=0.8, colors=base.INK)
    colorbar.set_label(
        "BigFingerprint similarity", fontsize=10.5, color=base.INK, labelpad=2
    )
    colorbar.outline.set_edgecolor(base.LIGHT_BORDER)
    colorbar.outline.set_linewidth(0.8)
    figure.text(
        0.033,
        0.014,
        "Candidate order is determined solely by BigFingerprint similarity; topology schematics summarize p-TFP architecture classes.",
        ha="left",
        va="center",
        fontsize=10.1,
        color=base.MUTED,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    mapping_candidates = sorted(data_dir.glob("polymer_id_mapping_*.csv"))
    if len(mapping_candidates) != 1:
        raise ValueError(
            f"Expected exactly one polymer_id_mapping_*.csv in {data_dir}, "
            f"found {len(mapping_candidates)}"
        )
    count = len(pd.read_csv(mapping_candidates[0]))
    input_json = data_dir / f"query_selection_candidates_{count}.json"
    output_dir = data_dir / "ranked_query_figure"
    output_stem = (
        output_dir / f"bigfingerprint_top5_ranked_queries_{count}_with_bigsmiles"
    )

    base.configure_matplotlib()
    base.SELECTED_QUERIES = args.queries
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(input_json.read_text(encoding="utf-8"))
    query_rows = base.find_query_rows(data)
    figure = make_figure(query_rows)
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)
    print(output_stem)


if __name__ == "__main__":
    main()

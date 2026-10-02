#!/usr/bin/env python3
"""Regenerate the manuscript-style Fold-1 plots from saved predictions."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
PROPERTY_ROOT = HERE.parent
RESULTS_ROOT = PROPERTY_ROOT / "results"
PREDICTIONS = RESULTS_ROOT / "fold1_predictions"
PLOTS = RESULTS_ROOT / "figures"

FP_ORDER = ("morgan", "pecfp", "bigfp")
MODEL_ORDER = ("svr", "rf", "gpr")
FP_FILE_LABELS = {
    "morgan": "psmiles_MorganFP",
    "pecfp": "ECFP",
    "bigfp": "BigFP",
}
MODEL_LABELS = {"svr": "SVR", "rf": "RF", "gpr": "GP"}

AXIS_MIN = -200
AXIS_MAX = 600
AXIS_TICKS = [-200, 0, 200, 400]
AXIS_TICKS_WITH_ENDPOINT = [-200, 0, 200, 400, 600]

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["text.usetex"] = shutil.which("latex") is not None
mac_times = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
if mac_times.exists():
    LEGEND_FONT = FontProperties(fname=mac_times, size=16)
else:
    LEGEND_FONT = FontProperties(family="Times New Roman", size=16)


def original_style_plot(
    y_train: np.ndarray,
    train_prediction: np.ndarray,
    y_test: np.ndarray,
    test_prediction: np.ndarray,
    destination: Path,
) -> None:
    plt.figure(figsize=(4, 4), dpi=300)
    plt.rcParams["axes.linewidth"] = 3
    plt.tick_params(length=8, width=2)
    line = np.linspace(AXIS_MIN, AXIS_MAX)
    plt.plot(line, line, c="k")
    plt.scatter(y_train, train_prediction, c="b", edgecolors="black", label="Train")
    plt.scatter(
        y_test,
        test_prediction,
        c="r",
        edgecolors="black",
        marker="s",
        label="Test",
    )
    plt.xlim(AXIS_MIN, AXIS_MAX)
    plt.ylim(AXIS_MIN, AXIS_MAX)
    plt.xticks(AXIS_TICKS, fontsize=14)
    plt.yticks(AXIS_TICKS, fontsize=14)
    plt.tick_params(axis="x", direction="in", top=True, labelbottom=False)
    plt.tick_params(axis="y", direction="in", right=True, labelleft=False)
    legend = plt.legend(loc="lower right", prop=LEGEND_FONT)
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_edgecolor("none")
    for label in legend.get_texts():
        label.set_fontproperties(LEGEND_FONT)
        label.set_usetex(False)
    plt.margins(0)
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.savefig(destination.with_suffix(".svg"), transparent=True, bbox_inches="tight")
    plt.savefig(
        destination.with_suffix(".png"),
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()


def axis_plot() -> None:
    plt.figure(figsize=(4, 4), dpi=300)
    plt.rcParams["axes.linewidth"] = 3
    plt.tick_params(length=8, width=2)
    line = np.linspace(AXIS_MIN, AXIS_MAX)
    plt.plot(line, line, c="k")
    plt.xlim(AXIS_MIN, AXIS_MAX)
    plt.ylim(AXIS_MIN, AXIS_MAX)
    plt.ylabel(r"Predicted $T_g$ ($^{\circ}\mathrm{C}$)", fontsize=18)
    plt.xlabel(r"Experimental $T_g$ ($^{\circ}\mathrm{C}$)", fontsize=18)
    plt.xticks(AXIS_TICKS_WITH_ENDPOINT, fontsize=14)
    plt.yticks(AXIS_TICKS_WITH_ENDPOINT, fontsize=14)
    plt.tick_params(axis="x", direction="in", top=True)
    plt.tick_params(axis="y", direction="in", right=True)
    plt.margins(0)
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.savefig(PLOTS / "axis.svg", transparent=True, bbox_inches="tight")
    plt.savefig(PLOTS / "axis.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close()


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for fingerprint in FP_ORDER:
        for model in MODEL_ORDER:
            path = PREDICTIONS / f"{fingerprint}__{model}__fold1_train_test.csv"
            predictions = pd.read_csv(path)
            train = predictions[predictions["split"] == "train"]
            test = predictions[predictions["split"] == "test"]
            if len(train) != 5725 or len(test) != 1432:
                raise ValueError(f"Unexpected Fold-1 sizes in {path}")
            destination = PLOTS / (
                f"{FP_FILE_LABELS[fingerprint]}_Tg_prediction_{MODEL_LABELS[model]}"
            )
            original_style_plot(
                train["tg_actual"].to_numpy(dtype=float),
                train["tg_predicted"].to_numpy(dtype=float),
                test["tg_actual"].to_numpy(dtype=float),
                test["tg_predicted"].to_numpy(dtype=float),
                destination,
            )
    axis_plot()


if __name__ == "__main__":
    main()

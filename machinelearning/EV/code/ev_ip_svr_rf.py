#!/usr/bin/env python3
"""Optimize or evaluate the shared EA/IP SVR and RF workflow."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.svm import SVR

TARGET_COLUMNS = {
    "ea": "EA vs SHE (eV)",
    "ip": "IP vs SHE (eV)",
}
FINGERPRINT_FILES = {
    "morgan": "psmiles_mfp_arr.npy",
    "pecfp": "EVIP_pecfp_arr.npy",
    "bigfp": "EVIP_bigfp.npy",
}
STD_DDOF = 0

DEFAULT_PARAMETERS = {
    "ea": {
        "svr": {
            "morgan": {"kernel": "rbf", "C": 1.0, "epsilon": 0.01},
            "pecfp": {"kernel": "rbf", "C": 1.0, "epsilon": 0.01},
            "bigfp": {"kernel": "rbf", "C": 1.0, "epsilon": 0.01},
        },
        "rf": {
            "morgan": {
                "n_estimators": 50,
                "max_depth": 200,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "max_leaf_nodes": 600,
                "random_state": 0,
                "n_jobs": -1,
            },
            "pecfp": {
                "n_estimators": 150,
                "max_depth": 100,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "max_leaf_nodes": 600,
                "random_state": 0,
                "n_jobs": -1,
            },
            "bigfp": {
                "n_estimators": 50,
                "max_depth": 100,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "max_leaf_nodes": 600,
                "random_state": 0,
                "n_jobs": -1,
            },
        },
    },
    "ip": {
        "svr": {
            "morgan": {"kernel": "rbf", "C": 1.0, "epsilon": 0.01},
            "pecfp": {"kernel": "rbf", "C": 1.0, "epsilon": 0.01},
            "bigfp": {"kernel": "rbf", "C": 10.0, "epsilon": 0.001},
        },
        "rf": {
            "morgan": {
                "n_estimators": 150,
                "max_depth": 100,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "max_leaf_nodes": 600,
                "random_state": 0,
                "n_jobs": -1,
            },
            "pecfp": {
                "n_estimators": 200,
                "max_depth": 100,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "max_leaf_nodes": 800,
                "random_state": 0,
                "n_jobs": -1,
            },
            "bigfp": {
                "n_estimators": 200,
                "max_depth": 100,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "max_leaf_nodes": 1000,
                "random_state": 0,
                "n_jobs": -1,
            },
        },
    },
}

DEFAULT_GRIDS = {
    "svr": {
        "kernel": ["rbf"],
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.001, 0.01, 0.1],
    },
    "rf": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [100, 200],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": [1.0],
        "max_leaf_nodes": [600, 800, 1000],
        "random_state": [0],
        "n_jobs": [-1],
    },
}


def build_sampled_global_indices(seed=0):
    """Build the intended 9 x 100 x 3 random sample of full-data indices.

    Parameters
    ----------
    seed : int, default=0
        Seed used to select 100 of the 682 monomer-B identities.

    Returns
    -------
    tuple[numpy.ndarray, list[int]]
        The 2,700 global row indices and selected monomer-B identities.
    """
    sampled_monomer_b = random.Random(seed).sample(range(682), 100)
    global_indices = np.asarray(
        [
            monomer_a * 682 * 3 + monomer_b * 3 + topology
            for monomer_a in range(9)
            for monomer_b in sampled_monomer_b
            for topology in range(3)
        ],
        dtype=np.int64,
    )
    return global_indices, sampled_monomer_b


def make_test_splits(n_samples=2700):
    """Create the five fixed test-fold position arrays.

    Parameters
    ----------
    n_samples : int, default=2700
        Number of rows in the sampled data subset.

    Returns
    -------
    list[numpy.ndarray]
        Five non-overlapping test-fold positions generated with seed 42.
    """
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    return [
        np.asarray(test_positions, dtype=np.int64)
        for _, test_positions in splitter.split(np.arange(n_samples))
    ]


def load_selected_data(data_dir, target):
    """Load labels and three fingerprints, then apply the sampled global indices.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory containing ``dataset.csv`` and the three NumPy arrays.
    target : {"ea", "ip"}
        Property column to load.

    Returns
    -------
    tuple
        Selected target values, selected fingerprint matrices, global indices,
        and sampled monomer-B identities.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError("target must be 'ea' or 'ip'")
    global_indices, sampled_monomer_b = build_sampled_global_indices()
    labels = pd.read_csv(data_dir / "dataset.csv")[TARGET_COLUMNS[target]].to_numpy(
        dtype=float
    )
    if global_indices.max() >= len(labels):
        raise IndexError("A sampled global index exceeds the target array.")

    matrices = {}
    for fingerprint, filename in FINGERPRINT_FILES.items():
        full_matrix = np.load(data_dir / filename, mmap_mode="r")
        if global_indices.max() >= full_matrix.shape[0]:
            raise IndexError(f"A sampled global index exceeds {filename}.")
        matrices[fingerprint] = np.asarray(full_matrix[global_indices])

    if len(global_indices) != 2700 or len(np.unique(global_indices)) != 2700:
        raise ValueError("Expected exactly 2,700 unique sampled global indices.")
    return labels[global_indices], matrices, global_indices, sampled_monomer_b


def calculate_metrics(actual, predicted):
    """Return R2, MSE, RMSE, and MAE for one prediction array."""
    mse = float(mean_squared_error(actual, predicted))
    return {
        "r2": float(r2_score(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def build_model(model_name, parameters):
    if model_name == "svr":
        return SVR(**parameters)
    if model_name == "rf":
        return RandomForestRegressor(**parameters)
    raise ValueError("model_name must be 'svr' or 'rf'")


def load_parameter_map(path):
    if path is None:
        return DEFAULT_PARAMETERS
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_metrics(metric_frame):
    rows = []
    test_metrics = metric_frame.loc[metric_frame["split"] == "test"]
    for (model, fingerprint), group in test_metrics.groupby(
        ["model", "fingerprint"], sort=False
    ):
        row = {
            "model": model,
            "fingerprint": fingerprint,
            "n_folds": int(len(group)),
            "std_ddof": STD_DDOF,
        }
        for metric in ("r2", "mse", "rmse", "mae"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=STD_DDOF))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fold1(
    actual_train, predicted_train, actual_test, predicted_test, target, path
):
    """Save one manuscript-style Fold-1 observed-versus-predicted panel."""
    limits = (-5, 1) if target == "ea" else (0, 4)
    property_label = "EA" if target == "ea" else "IP"

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    fig, axis = plt.subplots(figsize=(4, 4), dpi=300)
    axis.plot(limits, limits, color="black")
    axis.scatter(
        actual_train,
        predicted_train,
        color="blue",
        edgecolors="black",
        label="Train",
    )
    axis.scatter(
        actual_test,
        predicted_test,
        color="red",
        edgecolors="black",
        marker="s",
        label="Test",
    )
    axis.set(xlim=limits, ylim=limits)
    axis.set_xlabel(f"Simulated {property_label} (eV)")
    axis.set_ylabel(f"Predicted {property_label} (eV)")
    axis.tick_params(direction="in", top=True, right=True, length=8, width=2)
    axis.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, transparent=True, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_models(
    data_dir,
    output_dir,
    target,
    model_names,
    fingerprints,
    parameter_map,
):
    """Evaluate fixed SVR/RF parameters across the five matched folds.

    Parameters
    ----------
    data_dir, output_dir : pathlib.Path
        Input-data and result directories.
    target : {"ea", "ip"}
        Property to predict.
    model_names : list[str]
        Any subset of ``["svr", "rf"]``.
    fingerprints : list[str]
        Any subset of ``["morgan", "pecfp", "bigfp"]``.
    parameter_map : dict
        Nested target/model/fingerprint estimator parameters.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Per-fold metrics and their five-fold mean/standard deviation summary.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError("target must be 'ea' or 'ip'")
    if set(model_names) - {"svr", "rf"}:
        raise ValueError("model_names can contain only 'svr' and 'rf'")
    if set(fingerprints) - set(FINGERPRINT_FILES):
        raise ValueError("Unknown fingerprint name")

    target_values, matrices, global_indices, sampled_monomer_b = load_selected_data(
        data_dir, target
    )
    test_splits = make_test_splits(len(global_indices))
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    figure_dir = output_dir / "fold1_figures"
    prediction_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    metric_rows = []
    for fold_index, test_positions in enumerate(test_splits):
        train_positions = np.concatenate(
            [
                positions
                for other_fold, positions in enumerate(test_splits)
                if other_fold != fold_index
            ]
        )
        for model_name in model_names:
            for fingerprint in fingerprints:
                model = build_model(
                    model_name,
                    parameter_map[target][model_name][fingerprint],
                )
                matrix = matrices[fingerprint]
                model.fit(matrix[train_positions], target_values[train_positions])
                train_prediction = model.predict(matrix[train_positions])
                test_prediction = model.predict(matrix[test_positions])

                for split, positions, predictions in (
                    ("train", train_positions, train_prediction),
                    ("test", test_positions, test_prediction),
                ):
                    metric_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "fingerprint": fingerprint,
                            "fold": fold_index + 1,
                            "split": split,
                            "n_samples": len(positions),
                            **calculate_metrics(target_values[positions], predictions),
                        }
                    )

                np.savez_compressed(
                    prediction_dir
                    / f"fold{fold_index + 1}_{target}_{model_name}_{fingerprint}.npz",
                    train_subset_positions=train_positions,
                    test_subset_positions=test_positions,
                    train_global_indices=global_indices[train_positions],
                    test_global_indices=global_indices[test_positions],
                    y_train=target_values[train_positions],
                    y_test=target_values[test_positions],
                    prediction_train=train_prediction,
                    prediction_test=test_prediction,
                )
                if fold_index == 0:
                    property_label = "EV" if target == "ea" else "IP"
                    fingerprint_label = {
                        "morgan": "psmiles_MorganFP",
                        "pecfp": "ECFP",
                        "bigfp": "BigFP",
                    }[fingerprint]
                    plot_fold1(
                        target_values[train_positions],
                        train_prediction,
                        target_values[test_positions],
                        test_prediction,
                        target,
                        figure_dir / f"{fingerprint_label}_{property_label}_prediction_"
                        f"{model_name.upper()}.svg",
                    )

    metric_frame = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metric_frame)
    metric_frame.to_csv(output_dir / "five_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "five_fold_summary.csv", index=False)
    metadata = {
        "target": target,
        "sample_random_seed": 0,
        "sampled_monomer_b": sampled_monomer_b,
        "kfold_random_state": 42,
        "n_samples": 2700,
        "std_ddof": STD_DDOF,
        "parameters": parameter_map[target],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metric_frame, summary


def optimize_model(data_dir, output_dir, target, model_name, fingerprint, grid):
    """Optimize one SVR/RF model on Folds 2-5 while preserving Fold 1.

    Parameters
    ----------
    data_dir, output_dir : pathlib.Path
        Input-data and result directories.
    target, model_name, fingerprint : str
        Property, estimator family, and fingerprint to optimize.
    grid : dict
        Scikit-learn parameter grid.

    Returns
    -------
    pandas.DataFrame
        Candidate parameters and matched four-fold validation scores, sorted
        from highest to lowest mean R2.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError("target must be 'ea' or 'ip'")
    if model_name not in {"svr", "rf"}:
        raise ValueError("model_name must be 'svr' or 'rf'")
    if fingerprint not in FINGERPRINT_FILES:
        raise ValueError("Unknown fingerprint name")

    target_values, matrices, _, _ = load_selected_data(data_dir, target)
    folds = make_test_splits(len(target_values))
    development_folds = folds[1:]
    matrix = matrices[fingerprint]
    rows = []
    for parameters in ParameterGrid(grid):
        fold_scores = []
        for validation_index, validation_positions in enumerate(development_folds):
            train_positions = np.concatenate(
                [
                    positions
                    for other_index, positions in enumerate(development_folds)
                    if other_index != validation_index
                ]
            )
            model = build_model(model_name, parameters)
            model.fit(matrix[train_positions], target_values[train_positions])
            prediction = model.predict(matrix[validation_positions])
            fold_scores.append(
                r2_score(target_values[validation_positions], prediction)
            )
        rows.append(
            {
                "target": target,
                "model": model_name,
                "fingerprint": fingerprint,
                "parameters": json.dumps(parameters, sort_keys=True),
                "r2_mean": float(np.mean(fold_scores)),
                "r2_std": float(np.std(fold_scores, ddof=STD_DDOF)),
                **{
                    f"validation_fold_{index + 2}_r2": float(score)
                    for index, score in enumerate(fold_scores)
                },
            }
        )

    results = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        output_dir / f"optimization_{target}_{model_name}_{fingerprint}.csv",
        index=False,
    )
    return results


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--target", choices=TARGET_COLUMNS, required=True)
    evaluate_parser.add_argument("--data-dir", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument(
        "--models", nargs="+", choices=("svr", "rf"), default=["svr", "rf"]
    )
    evaluate_parser.add_argument(
        "--fingerprints",
        nargs="+",
        choices=tuple(FINGERPRINT_FILES),
        default=list(FINGERPRINT_FILES),
    )
    evaluate_parser.add_argument("--parameters", type=Path)

    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--target", choices=TARGET_COLUMNS, required=True)
    optimize_parser.add_argument("--model", choices=("svr", "rf"), required=True)
    optimize_parser.add_argument(
        "--fingerprint", choices=tuple(FINGERPRINT_FILES), required=True
    )
    optimize_parser.add_argument("--data-dir", type=Path, required=True)
    optimize_parser.add_argument("--output-dir", type=Path, required=True)
    optimize_parser.add_argument(
        "--grid",
        type=Path,
        help="JSON parameter grid; defaults to the compact built-in grid.",
    )
    return parser.parse_args()


def main():
    args = parse_cli()
    if args.command == "evaluate":
        _, summary = evaluate_models(
            args.data_dir.resolve(),
            args.output_dir.resolve(),
            args.target,
            args.models,
            args.fingerprints,
            load_parameter_map(args.parameters),
        )
        print(summary.to_string(index=False))
        return

    grid = (
        json.loads(args.grid.read_text(encoding="utf-8"))
        if args.grid
        else DEFAULT_GRIDS[args.model]
    )
    results = optimize_model(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        args.target,
        args.model,
        args.fingerprint,
        grid,
    )
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

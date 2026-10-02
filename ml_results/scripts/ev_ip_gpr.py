#!/usr/bin/env python3
"""Run or summarize one fold of the shared EA/IP exact-GPR workflow."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COLUMNS = {"ea": "ea_eV", "ip": "ip_eV"}
FINGERPRINT_FILES = {
    "morgan": "fingerprint_morgan.npy",
    "pecfp": "fingerprint_pecfp.npy",
    "bigfp": "fingerprint_bigfp.npy",
}
EXPECTED_DIMENSIONS = {"morgan": 2048, "pecfp": 2048, "bigfp": 2088}
N_RESTARTS = 20
STD_DDOF = 0


def build_kernel():
    """Return the kernel used by the original EA/IP GPR calculations."""
    return ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e2)) + WhiteKernel(
        1.0, (1e-3, 1.0)
    )


def calculate_metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {
        "r2": float(r2_score(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def load_folds(data_dir):
    with np.load(data_dir / "split_indices.npz") as split_file:
        return [
            np.asarray(split_file[f"fold{fold}"], dtype=np.int64)
            for fold in range(1, 6)
        ]


def validate_data(data_dir):
    """Validate the prepared 2,700-row EA/IP SCC input bundle.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory containing targets, fold indices, and fingerprint arrays.

    Returns
    -------
    dict
        Validated sample, fold, and fingerprint dimensions.
    """
    required = [
        "targets.csv",
        "split_indices.npz",
        *FINGERPRINT_FILES.values(),
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    targets = pd.read_csv(data_dir / "targets.csv")
    if len(targets) != 2700:
        raise ValueError("Expected exactly 2,700 target rows.")
    if targets["subset_position"].tolist() != list(range(2700)):
        raise ValueError("targets.csv subset positions must be 0..2699.")
    if targets["global_index"].nunique() != 2700:
        raise ValueError("Sampled global indices must be unique.")
    if any(targets[column].isna().any() for column in TARGET_COLUMNS.values()):
        raise ValueError("EA/IP target columns contain missing values.")

    folds = load_folds(data_dir)
    if [len(fold) for fold in folds] != [540] * 5:
        raise ValueError("Expected five test folds of 540 rows.")
    if not np.array_equal(np.sort(np.concatenate(folds)), np.arange(2700)):
        raise ValueError("The test folds overlap or do not cover all 2,700 rows.")

    shapes = {}
    for fingerprint, filename in FINGERPRINT_FILES.items():
        matrix = np.load(data_dir / filename, mmap_mode="r")
        expected_shape = (2700, EXPECTED_DIMENSIONS[fingerprint])
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{filename}: expected shape {expected_shape}, found {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{filename} contains non-finite values.")
        shapes[fingerprint] = list(matrix.shape)

    return {
        "sample_count": 2700,
        "fold_sizes": [len(fold) for fold in folds],
        "train_samples_per_fold": 2160,
        "test_samples_per_fold": 540,
        "fingerprint_shapes": shapes,
        "input_preprocessing": "none",
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_fold(data_dir, output_dir, target, fold_number, fingerprints, dry_run=False):
    """Fit three exact-GPR embeddings for one property and one held-out fold.

    Parameters
    ----------
    data_dir, output_dir : pathlib.Path
        Prepared input and result directories.
    target : {"ea", "ip"}
        Property to predict.
    fold_number : int
        Held-out fold number from 1 through 5.
    fingerprints : list[str]
        Fingerprint names to fit.
    dry_run : bool, default=False
        Validate inputs and print dimensions without fitting.

    Returns
    -------
    None
        Metrics and predictions are written beneath ``output_dir``.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError("target must be 'ea' or 'ip'")
    unknown_fingerprints = set(fingerprints) - set(FINGERPRINT_FILES)
    if unknown_fingerprints:
        raise ValueError(f"Unknown fingerprints: {sorted(unknown_fingerprints)}")
    if fold_number not in range(1, 6):
        raise ValueError("fold_number must be 1, 2, 3, 4, or 5")
    validation = validate_data(data_dir)
    print(json.dumps(validation, indent=2), flush=True)
    if dry_run:
        return

    targets = pd.read_csv(data_dir / "targets.csv")
    target_values = targets[TARGET_COLUMNS[target]].to_numpy(dtype=float)
    folds = load_folds(data_dir)
    test_indices = folds[fold_number - 1]
    train_indices = np.concatenate(
        [fold for index, fold in enumerate(folds, start=1) if index != fold_number]
    )
    fold_dir = output_dir / target / f"fold{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for fingerprint in fingerprints:
        matrix = np.load(data_dir / FINGERPRINT_FILES[fingerprint], mmap_mode="r")
        train_matrix = np.asarray(matrix[train_indices])
        test_matrix = np.asarray(matrix[test_indices])
        model = GaussianProcessRegressor(
            kernel=build_kernel(),
            alpha=1e-10,
            n_restarts_optimizer=N_RESTARTS,
            normalize_y=False,
            random_state=None,
        )
        started = time.time()
        model.fit(train_matrix, target_values[train_indices])
        fit_seconds = time.time() - started
        train_prediction = model.predict(train_matrix)
        test_prediction = model.predict(test_matrix)
        result = {
            "target": target,
            "fingerprint": fingerprint,
            "model": "gpr",
            "test_fold": fold_number,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "input_dimensions": matrix.shape[1],
            "input_preprocessing": "none",
            "alpha": 1e-10,
            "normalize_y": False,
            "n_restarts_optimizer": N_RESTARTS,
            "learned_kernel": str(model.kernel_),
            "fit_seconds": fit_seconds,
            "train_metrics": calculate_metrics(
                target_values[train_indices], train_prediction
            ),
            "test_metrics": calculate_metrics(
                target_values[test_indices], test_prediction
            ),
        }
        write_json(fold_dir / f"{fingerprint}__gpr_metrics.json", result)
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "target": target,
                        "fingerprint": fingerprint,
                        "model": "gpr",
                        "test_fold": fold_number,
                        "split": "train",
                        "subset_position": train_indices,
                        "global_index": targets.iloc[train_indices][
                            "global_index"
                        ].to_numpy(),
                        "actual": target_values[train_indices],
                        "predicted": train_prediction,
                    }
                ),
                pd.DataFrame(
                    {
                        "target": target,
                        "fingerprint": fingerprint,
                        "model": "gpr",
                        "test_fold": fold_number,
                        "split": "test",
                        "subset_position": test_indices,
                        "global_index": targets.iloc[test_indices][
                            "global_index"
                        ].to_numpy(),
                        "actual": target_values[test_indices],
                        "predicted": test_prediction,
                    }
                ),
            ],
            ignore_index=True,
        ).to_csv(fold_dir / f"{fingerprint}__gpr_predictions.csv", index=False)
        print(
            f"{target} fold{fold_number} {fingerprint}: "
            f"R2={result['test_metrics']['r2']:.6f}, "
            f"RMSE={result['test_metrics']['rmse']:.6f}",
            flush=True,
        )
        del model, matrix, train_matrix, test_matrix
        gc.collect()


def summarize_target(data_dir, output_dir, target):
    """Aggregate five completed GPR folds into mean and standard deviation."""
    if target not in TARGET_COLUMNS:
        raise ValueError("target must be 'ea' or 'ip'")
    validate_data(data_dir)
    rows = []
    prediction_frames = []
    for fold_number in range(1, 6):
        for fingerprint in FINGERPRINT_FILES:
            fold_dir = output_dir / target / f"fold{fold_number}"
            result = json.loads(
                (fold_dir / f"{fingerprint}__gpr_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            row = {
                key: result[key]
                for key in (
                    "target",
                    "fingerprint",
                    "model",
                    "test_fold",
                    "n_train",
                    "n_test",
                    "input_dimensions",
                    "input_preprocessing",
                    "alpha",
                    "normalize_y",
                    "n_restarts_optimizer",
                    "learned_kernel",
                    "fit_seconds",
                )
            }
            row.update(
                {
                    f"test_{metric}": value
                    for metric, value in result["test_metrics"].items()
                }
            )
            rows.append(row)
            predictions = pd.read_csv(fold_dir / f"{fingerprint}__gpr_predictions.csv")
            prediction_frames.append(predictions.loc[predictions["split"] == "test"])

    fold_metrics = pd.DataFrame(rows)
    summary_rows = []
    for fingerprint, group in fold_metrics.groupby("fingerprint", sort=False):
        row = {
            "target": target,
            "model": "gpr",
            "fingerprint": fingerprint,
            "n_folds": len(group),
            "std_ddof": STD_DDOF,
        }
        for metric in ("r2", "mse", "rmse", "mae"):
            values = group[f"test_{metric}"].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=STD_DDOF))
        summary_rows.append(row)

    target_dir = output_dir / target
    fold_metrics.to_csv(target_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(target_dir / "summary_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        target_dir / "oof_predictions.csv", index=False
    )
    return pd.DataFrame(summary_rows)


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data-dir", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--target", choices=TARGET_COLUMNS, required=True)
    run_parser.add_argument("--fold", type=int, required=True)
    run_parser.add_argument("--data-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument(
        "--fingerprints",
        nargs="+",
        choices=tuple(FINGERPRINT_FILES),
        default=list(FINGERPRINT_FILES),
    )
    run_parser.add_argument("--dry-run", action="store_true")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--target", choices=("ea", "ip", "all"), default="all")
    summary_parser.add_argument("--data-dir", type=Path, required=True)
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_cli()
    if args.command == "validate":
        print(json.dumps(validate_data(args.data_dir.resolve()), indent=2))
        return
    if args.command == "run":
        run_fold(
            args.data_dir.resolve(),
            args.output_dir.resolve(),
            args.target,
            args.fold,
            args.fingerprints,
            args.dry_run,
        )
        return
    targets = ("ea", "ip") if args.target == "all" else (args.target,)
    for target in targets:
        print(
            summarize_target(
                args.data_dir.resolve(), args.output_dir.resolve(), target
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()

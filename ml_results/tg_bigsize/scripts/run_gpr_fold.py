#!/usr/bin/env python3
"""Run one large-PolyInfo Tg exact-GPR fold or summarize all five folds.

This is the large-dataset counterpart of ThermalProcess/
final_prediction_tg_five_fold.py. It keeps the original GPR kernel, alpha,
optimizer restarts, and random seed. The only data-level changes are loading the
7,170-sample fingerprint matrices and the same five folders already used for
the large-dataset SVR and Random Forest models.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "gpr"
FINGERPRINTS = ("morgan", "pecfp", "bigfp")
FINGERPRINT_FILES = {
    "morgan": "fingerprint_morgan.npy",
    "pecfp": "fingerprint_pecfp.npy",
    "bigfp": "fingerprint_bigfp.npy",
}
EXPECTED_DIMENSIONS = {"morgan": 2048, "pecfp": 2048, "bigfp": 2088}
FIXED_ALPHA = {"morgan": 500.0, "pecfp": 300.0, "bigfp": 300.0}
N_RESTARTS = 20
RANDOM_STATE = 42
STD_DDOF = 0


def exact_small_dataset_kernel():
    """Return the kernel used by the original ThermalProcess Tg GPR."""
    return ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-6, 1e6),
    ) * RBF(
        length_scale=1.0,
        length_scale_bounds=(1e-6, 1e6),
    )


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(actual, predicted))
    return {
        "r2": float(r2_score(actual, predicted)),
        "mse": mse,
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_folds(data_dir: Path) -> list[np.ndarray]:
    with np.load(data_dir / "split_indices.npz") as split_file:
        return [np.asarray(split_file[f"fold{i}"], dtype=np.int64) for i in range(1, 6)]


def validate_data(data_dir: Path) -> dict[str, Any]:
    required = [
        "cleaned_tg.csv",
        "split_indices.npz",
        "split_assignments.csv",
        *FINGERPRINT_FILES.values(),
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required large-Tg data: {', '.join(missing)}")

    cleaned = pd.read_csv(data_dir / "cleaned_tg.csv")
    assignments = pd.read_csv(data_dir / "split_assignments.csv")
    if len(cleaned) != 7170 or len(assignments) != 7170:
        raise ValueError("Expected exactly 7,170 samples in cleaned data and assignments")
    if cleaned["tg"].isna().any():
        raise ValueError("Tg target contains missing values")

    folds = load_folds(data_dir)
    combined = np.concatenate(folds)
    if [len(fold) for fold in folds] != [1434] * 5:
        raise ValueError("Expected the same five 1,434-sample folds used by SVR/RF")
    if len(np.unique(combined)) != 7170 or set(combined.tolist()) != set(range(7170)):
        raise ValueError("Five-fold indices overlap or do not cover all 7,170 samples")
    for fold_number, indices in enumerate(folds, start=1):
        assigned = assignments.iloc[indices]["folder"].to_numpy(dtype=int)
        if not np.all(assigned == fold_number):
            raise ValueError(f"split_assignments.csv disagrees with fold{fold_number} indices")

    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    for fingerprint, filename in FINGERPRINT_FILES.items():
        matrix = np.load(data_dir / filename, mmap_mode="r")
        expected_shape = (7170, EXPECTED_DIMENSIONS[fingerprint])
        if matrix.shape != expected_shape:
            raise ValueError(f"{filename}: expected {expected_shape}, found {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{filename} contains non-finite values")
        shapes[fingerprint] = [int(value) for value in matrix.shape]
        dtypes[fingerprint] = str(matrix.dtype)

    return {
        "sample_count": 7170,
        "fold_sizes": [int(len(fold)) for fold in folds],
        "train_samples_per_job": 5736,
        "test_samples_per_job": 1434,
        "fingerprint_shapes": shapes,
        "fingerprint_dtypes": dtypes,
        "input_preprocessing": "none",
        "split_source": "the same split_indices.npz used by large-Tg SVR/RF",
    }


def run_fold(
    data_dir: Path,
    output_dir: Path,
    fold_number: int,
    fingerprints: list[str],
    dry_run: bool,
) -> None:
    if fold_number not in range(1, 6):
        raise ValueError("fold must be one of 1, 2, 3, 4, or 5")
    validation = validate_data(data_dir)
    cleaned = pd.read_csv(data_dir / "cleaned_tg.csv")
    y = cleaned["tg"].to_numpy(dtype=float)
    folds = load_folds(data_dir)
    test_indices = folds[fold_number - 1]
    train_indices = np.concatenate(
        [fold for number, fold in enumerate(folds, start=1) if number != fold_number]
    )

    print(json.dumps(validation, indent=2), flush=True)
    print(
        f"Fold {fold_number}: train={len(train_indices)}, test={len(test_indices)}",
        flush=True,
    )
    for fingerprint in fingerprints:
        print(
            f"{fingerprint}: dimensions={EXPECTED_DIMENSIONS[fingerprint]}, "
            f"alpha={FIXED_ALPHA[fingerprint]:g}, restarts={N_RESTARTS}",
            flush=True,
        )
    if dry_run:
        print("Dry run complete; no GPR model was fitted.", flush=True)
        return

    fold_dir = output_dir / f"fold{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fingerprint in fingerprints:
        metrics_path = fold_dir / f"{fingerprint}__gpr_metrics.json"
        predictions_path = fold_dir / f"{fingerprint}__gpr_predictions.csv"
        if metrics_path.exists() and predictions_path.exists():
            print(f"Checkpoint exists; skip fold{fold_number} {fingerprint}", flush=True)
            continue

        X = np.load(data_dir / FINGERPRINT_FILES[fingerprint], mmap_mode="r")
        # This preserves every generated embedding column. No scaler, feature
        # selector, variance filter, PCA, or other preprocessing is applied.
        X_train = np.asarray(X[train_indices])
        X_test = np.asarray(X[test_indices])
        alpha = FIXED_ALPHA[fingerprint]
        model = GaussianProcessRegressor(
            kernel=exact_small_dataset_kernel(),
            alpha=alpha,
            n_restarts_optimizer=N_RESTARTS,
            random_state=RANDOM_STATE,
        )
        print(
            f"Start fold{fold_number} {fingerprint}: X_train={X_train.shape}, "
            f"X_test={X_test.shape}, dtype={X_train.dtype}",
            flush=True,
        )
        fit_started = time.time()
        model.fit(X_train, y[train_indices])
        fit_seconds = time.time() - fit_started
        predict_started = time.time()
        prediction = model.predict(X_test)
        predict_seconds = time.time() - predict_started
        scores = calculate_metrics(y[test_indices], prediction)

        result = {
            "fingerprint": fingerprint,
            "model": "GaussianProcessRegressor",
            "test_fold": f"fold{fold_number}",
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "input_dimensions": int(X.shape[1]),
            "input_dtype": str(X.dtype),
            "input_preprocessing": "none",
            "kernel_template": "ConstantKernel(1.0, (1e-6, 1e6)) * RBF(1.0, (1e-6, 1e6))",
            "alpha": alpha,
            "n_restarts_optimizer": N_RESTARTS,
            "random_state": RANDOM_STATE,
            "learned_kernel": str(model.kernel_),
            "log_marginal_likelihood": float(model.log_marginal_likelihood_value_),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            **scores,
        }
        selected = cleaned.iloc[test_indices]
        predictions = pd.DataFrame(
            {
                "fingerprint": fingerprint,
                "model": "gpr",
                "test_fold": f"fold{fold_number}",
                "sample_index": test_indices,
                "folder": selected["folder"].to_numpy(dtype=int),
                "canonical_psmiles": selected["canonical_psmiles"].to_numpy(),
                "tg_actual": y[test_indices],
                "tg_predicted": prediction,
            }
        )
        write_json(metrics_path, result)
        write_csv(predictions_path, predictions)
        print(
            f"Finished fold{fold_number} {fingerprint}: R2={scores['r2']:.5f}, "
            f"RMSE={scores['rmse']:.4f}, MAE={scores['mae']:.4f}, "
            f"fit={fit_seconds / 60:.1f} min; learned={model.kernel_}",
            flush=True,
        )
        del model, X_train, X_test, X
        gc.collect()


def summarize(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    validate_data(data_dir)
    folds = load_folds(data_dir)
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number in range(1, 6):
        for fingerprint in FINGERPRINTS:
            fold_dir = output_dir / f"fold{fold_number}"
            metrics_path = fold_dir / f"{fingerprint}__gpr_metrics.json"
            predictions_path = fold_dir / f"{fingerprint}__gpr_predictions.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                raise RuntimeError(
                    f"Missing result for fold{fold_number} {fingerprint}; "
                    "all five qsub jobs must finish first"
                )
            result = json.loads(metrics_path.read_text())
            predictions = pd.read_csv(predictions_path)
            expected_indices = set(folds[fold_number - 1].tolist())
            actual_indices = set(predictions["sample_index"].astype(int).tolist())
            if actual_indices != expected_indices or len(predictions) != 1434:
                raise RuntimeError(f"Prediction rows are incomplete for fold{fold_number} {fingerprint}")
            if predictions[["tg_actual", "tg_predicted"]].isna().any().any():
                raise RuntimeError(f"Missing prediction value for fold{fold_number} {fingerprint}")
            fold_rows.append(result)
            prediction_frames.append(predictions)

    fold_frame = pd.DataFrame(fold_rows).sort_values(["fingerprint", "test_fold"])
    summary_rows: list[dict[str, Any]] = []
    for fingerprint, group in fold_frame.groupby("fingerprint", sort=True):
        row: dict[str, Any] = {
            "fingerprint": fingerprint,
            "model": "gpr",
            "n_folds": 5,
            "n_train_per_fold": 5736,
            "n_test_per_fold": 1434,
            "input_dimensions": int(group["input_dimensions"].iloc[0]),
            "input_preprocessing": "none",
            "alpha": float(group["alpha"].iloc[0]),
            "n_restarts_optimizer": N_RESTARTS,
            "std_ddof": STD_DDOF,
        }
        for metric_name in ("r2", "mse", "rmse", "mae"):
            values = group[metric_name].to_numpy(dtype=float)
            row[f"{metric_name}_mean"] = float(np.mean(values))
            row[f"{metric_name}_std"] = float(np.std(values, ddof=STD_DDOF))
        summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows).sort_values("fingerprint")
    write_csv(output_dir / "fold_metrics.csv", fold_frame)
    write_csv(output_dir / "summary_metrics.csv", summary_frame)
    write_csv(output_dir / "predictions.csv", pd.concat(prediction_frames, ignore_index=True))
    write_json(
        output_dir / "run_config.json",
        {
            "small_dataset_reference": "ThermalProcess/final_prediction_tg_five_fold.py",
            "kernel": "ConstantKernel * RBF",
            "fixed_alpha": FIXED_ALPHA,
            "n_restarts_optimizer": N_RESTARTS,
            "random_state": RANDOM_STATE,
            "split": "same five folders used by large-Tg SVR/RF",
            "input_preprocessing": "none",
            "standard_deviation_ddof": STD_DDOF,
        },
    )
    print(summary_frame.to_string(index=False), flush=True)
    return summary_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    run = subparsers.add_parser("run")
    run.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    run.add_argument("--fingerprints", nargs="+", choices=FINGERPRINTS, default=list(FINGERPRINTS))
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--dry-run", action="store_true")

    collect = subparsers.add_parser("summarize")
    collect.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        print(json.dumps(validate_data(args.data_dir.resolve()), indent=2), flush=True)
    elif args.command == "run":
        run_fold(
            args.data_dir.resolve(),
            args.output_dir.resolve(),
            args.fold,
            list(args.fingerprints),
            args.dry_run,
        )
    elif args.command == "summarize":
        summarize(args.data_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

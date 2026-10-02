#!/usr/bin/env python3
"""Run one corrected large-PolyInfo Tg exact-GPR fold or summarize all folds.

This is the large-dataset counterpart of ThermalProcess/
final_prediction_tg_five_fold.py. It keeps the original GPR kernel, alpha,
optimizer restarts, and random seed. It loads the corrected 7,157-sample
fingerprints after excluding explicit [2H], and uses exactly the same five
folders as the corrected large-dataset SVR and Random Forest models.
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
DEFAULT_DATA_DIR = HERE / "data"
DEFAULT_OUTPUT_DIR = HERE / "results"
FINGERPRINTS = ("morgan", "pecfp", "bigfp")
FINGERPRINT_FILES = {
    "morgan": "fingerprint_morgan.npy",
    "pecfp": "fingerprint_pecfp.npy",
    "bigfp": "fingerprint_bigfp.npy",
}
EXPECTED_DIMENSIONS = {"morgan": 2048, "pecfp": 2048, "bigfp": 2088}
EXPECTED_SAMPLE_COUNT = 7157
EXPECTED_FOLD_SIZES = [1432, 1432, 1431, 1431, 1431]
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
        "fingerprint_ptfp.npy",
        *FINGERPRINT_FILES.values(),
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required large-Tg data: {', '.join(missing)}")

    cleaned = pd.read_csv(data_dir / "cleaned_tg.csv")
    assignments = pd.read_csv(data_dir / "split_assignments.csv")
    if (
        len(cleaned) != EXPECTED_SAMPLE_COUNT
        or len(assignments) != EXPECTED_SAMPLE_COUNT
    ):
        raise ValueError(
            f"Expected exactly {EXPECTED_SAMPLE_COUNT:,} samples in cleaned data "
            "and assignments"
        )
    required_columns = {
        "sample_index",
        "canonical_psmiles",
        "bigsmiles",
        "tg",
        "folder",
    }
    if not required_columns.issubset(cleaned.columns):
        raise ValueError(
            "cleaned_tg.csv lacks columns: "
            + ", ".join(sorted(required_columns - set(cleaned.columns)))
        )
    if not np.array_equal(
        cleaned["sample_index"].to_numpy(dtype=int),
        np.arange(EXPECTED_SAMPLE_COUNT),
    ):
        raise ValueError(
            "cleaned_tg.csv sample_index is not the contiguous matrix row index"
        )
    if cleaned["tg"].isna().any():
        raise ValueError("Tg target contains missing values")
    if cleaned["canonical_psmiles"].str.contains(r"\[2H\]", regex=True).any():
        raise ValueError("Explicit [2H] remains in corrected canonical p-SMILES")
    if (
        not cleaned["canonical_psmiles"]
        .map(lambda text: str(text).count("*") == 2)
        .all()
    ):
        raise ValueError(
            "Corrected canonical p-SMILES must contain exactly two bare * endpoints"
        )
    if not cleaned["bigsmiles"].astype(str).str.startswith("{").all():
        raise ValueError("Corrected BigSMILES rows must begin with a stochastic object")

    folds = load_folds(data_dir)
    combined = np.concatenate(folds)
    if [len(fold) for fold in folds] != EXPECTED_FOLD_SIZES:
        raise ValueError(
            "Expected corrected SVR/RF fold sizes "
            f"{EXPECTED_FOLD_SIZES}, found {[len(fold) for fold in folds]}"
        )
    if len(np.unique(combined)) != EXPECTED_SAMPLE_COUNT or set(
        combined.tolist()
    ) != set(range(EXPECTED_SAMPLE_COUNT)):
        raise ValueError(
            f"Five-fold indices overlap or do not cover all {EXPECTED_SAMPLE_COUNT:,} samples"
        )
    for fold_number, indices in enumerate(folds, start=1):
        assigned = assignments.iloc[indices]["folder"].to_numpy(dtype=int)
        if not np.all(assigned == fold_number):
            raise ValueError(
                f"split_assignments.csv disagrees with fold{fold_number} indices"
            )

    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    for fingerprint, filename in FINGERPRINT_FILES.items():
        matrix = np.load(data_dir / filename, mmap_mode="r")
        expected_shape = (EXPECTED_SAMPLE_COUNT, EXPECTED_DIMENSIONS[fingerprint])
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{filename}: expected {expected_shape}, found {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{filename} contains non-finite values")
        shapes[fingerprint] = [int(value) for value in matrix.shape]
        dtypes[fingerprint] = str(matrix.dtype)

    ptfp = np.load(data_dir / "fingerprint_ptfp.npy", mmap_mode="r")
    if ptfp.shape != (EXPECTED_SAMPLE_COUNT, 40):
        raise ValueError(
            f"fingerprint_ptfp.npy: expected ({EXPECTED_SAMPLE_COUNT}, 40), found {ptfp.shape}"
        )
    if not np.array_equal(np.unique(ptfp), np.array([0, 1], dtype=ptfp.dtype)):
        raise ValueError("p-TFP must be binary and contain both 0 and 1")
    pecfp = np.load(data_dir / FINGERPRINT_FILES["pecfp"], mmap_mode="r")
    bigfp = np.load(data_dir / FINGERPRINT_FILES["bigfp"], mmap_mode="r")
    if not np.array_equal(np.asarray(bigfp[:, :2048]), np.asarray(pecfp)):
        raise ValueError("BigFingerprint first 2,048 dimensions do not equal p-ECFP")
    if not np.array_equal(np.asarray(bigfp[:, 2048:]), np.asarray(ptfp)):
        raise ValueError("BigFingerprint final 40 dimensions do not equal p-TFP")
    nonconstant_ptfp = (
        np.flatnonzero((ptfp.sum(axis=0) > 0) & (ptfp.sum(axis=0) < len(ptfp))) + 1
    ).tolist()
    if nonconstant_ptfp != [20, 21]:
        raise ValueError(
            f"Expected nonconstant p-TFP dimensions [20, 21], found {nonconstant_ptfp}"
        )

    return {
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "fold_sizes": [int(len(fold)) for fold in folds],
        "train_samples_by_test_fold": [
            int(EXPECTED_SAMPLE_COUNT - len(fold)) for fold in folds
        ],
        "test_samples_by_fold": [int(len(fold)) for fold in folds],
        "fingerprint_shapes": shapes,
        "fingerprint_dtypes": dtypes,
        "ptfp_nonconstant_dimensions_1_based": nonconstant_ptfp,
        "bigfp_composition_verified": True,
        "explicit_deuterium_excluded": True,
        "input_preprocessing": "none",
        "split_source": "the same corrected split_indices.npz used by large-Tg SVR/RF",
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
            print(
                f"Checkpoint exists; skip fold{fold_number} {fingerprint}", flush=True
            )
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
                "bigsmiles": selected["bigsmiles"].to_numpy(),
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
            if actual_indices != expected_indices or len(predictions) != len(
                folds[fold_number - 1]
            ):
                raise RuntimeError(
                    f"Prediction rows are incomplete for fold{fold_number} {fingerprint}"
                )
            if predictions[["tg_actual", "tg_predicted"]].isna().any().any():
                raise RuntimeError(
                    f"Missing prediction value for fold{fold_number} {fingerprint}"
                )
            fold_rows.append(result)
            prediction_frames.append(predictions)

    fold_frame = pd.DataFrame(fold_rows).sort_values(["fingerprint", "test_fold"])
    summary_rows: list[dict[str, Any]] = []
    for fingerprint, group in fold_frame.groupby("fingerprint", sort=True):
        row: dict[str, Any] = {
            "fingerprint": fingerprint,
            "model": "gpr",
            "n_folds": 5,
            "n_train_by_fold": ";".join(
                str(EXPECTED_SAMPLE_COUNT - len(fold)) for fold in folds
            ),
            "n_test_by_fold": ";".join(str(len(fold)) for fold in folds),
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
    write_csv(
        output_dir / "predictions.csv", pd.concat(prediction_frames, ignore_index=True)
    )
    write_json(
        output_dir / "run_config.json",
        {
            "small_dataset_reference": "ThermalProcess/final_prediction_tg_five_fold.py",
            "kernel": "ConstantKernel * RBF",
            "fixed_alpha": FIXED_ALPHA,
            "n_restarts_optimizer": N_RESTARTS,
            "random_state": RANDOM_STATE,
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "fold_sizes": EXPECTED_FOLD_SIZES,
            "split": "same corrected five folders used by large-Tg SVR/RF",
            "explicit_deuterium_excluded": True,
            "morgan_input": "canonical p-SMILES",
            "pecfp_input": "BigSMILES",
            "ptfp_input": "BigSMILES",
            "bigfp_definition": "concatenate(p-ECFP[2048], p-TFP[40])",
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
    run.add_argument(
        "--fingerprints", nargs="+", choices=FINGERPRINTS, default=list(FINGERPRINTS)
    )
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

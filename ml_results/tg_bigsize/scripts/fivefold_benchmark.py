#!/usr/bin/env python3
"""Reproduce the ThermalProcess five-folder Tg modeling protocol.

Protocol
--------
1. Randomly divide all analyzed samples into five equal, reproducible folders.
2. Keep folder 1 completely held out as the final 20% test set.
3. Tune parameters on folders 2--5 using four-fold CV: three folders train and
   one folder validates in every CV split.
4. Select the candidate with the highest mean validation R2, matching the old
   ThermalProcess optimization scripts.
5. Refit on folders 2--5 and predict folder 1 exactly once.

RF and SVR retain the search spaces used by the current PolyInfo benchmark.
Exact GP reproduces the old Tg model and search: ConstantKernel * RBF,
alpha in [100, 200, 300, 400, 500, 600], 20 optimizer restarts, and seed 42.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from sklearn.svm import SVR

from tg_pipeline import FINGERPRINT_FILES, _json_safe, _write_json

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "artifacts"
DEFAULT_OUTPUT = HERE / "artifacts_fivefold"
FINGERPRINTS = ("morgan", "pecfp", "bigfp")
CLASSICAL_MODELS = ("rf", "svr")
GP_ALPHAS = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)
_CONDENSED_DISTANCE_CACHE: dict[str, np.ndarray] = {}
_DISTANCE_CACHE_LOCK = threading.Lock()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


class CachedRBF(RBF):
    """Isotropic RBF with cached unscaled train/train squared distances.

    GaussianProcessRegressor evaluates the same train/train RBF kernel many
    times while optimizing only its length scale. Caching ``pdist(X)`` removes
    repeated O(n^2 d) distance construction without changing the kernel,
    likelihood, bounds, optimizer, or predictions.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        length_scale_bounds: tuple[float, float] = (1e-5, 1e5),
        cache_key: str | None = None,
    ):
        super().__init__(
            length_scale=length_scale, length_scale_bounds=length_scale_bounds
        )
        self.cache_key = cache_key

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None or self.cache_key is None or np.ndim(self.length_scale) != 0:
            return super().__call__(X, Y=Y, eval_gradient=eval_gradient)

        key = f"{self.cache_key}:{X.shape}:{X.dtype}"
        with _DISTANCE_CACHE_LOCK:
            unscaled = _CONDENSED_DISTANCE_CACHE.get(key)
            if unscaled is None:
                unscaled = pdist(X, metric="sqeuclidean")
                _CONDENSED_DISTANCE_CACHE[key] = unscaled

        length_scale = float(self.length_scale)
        scaled = unscaled / (length_scale * length_scale)
        kernel = np.exp(-0.5 * scaled)
        kernel = squareform(kernel)
        np.fill_diagonal(kernel, 1.0)
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return kernel, np.empty((X.shape[0], X.shape[0], 0))
            gradient = (kernel * squareform(scaled))[:, :, np.newaxis]
            return kernel, gradient
        return kernel


def exact_kernel(cache_key: str | None = None):
    return ConstantKernel(1.0, (1e-6, 1e6)) * CachedRBF(
        1.0, (1e-6, 1e6), cache_key=cache_key
    )


def _load_folds(output_dir: Path) -> list[np.ndarray]:
    split_file = np.load(output_dir / "split_indices.npz")
    return [split_file[f"fold{i}"] for i in range(1, 6)]


def _validate_folds(folds: list[np.ndarray], sample_count: int) -> None:
    if len(folds) != 5:
        raise ValueError("Exactly five folders are required")
    combined = np.concatenate(folds)
    if len(combined) != sample_count:
        raise ValueError("The five folders do not cover every sample")
    if len(np.unique(combined)) != sample_count:
        raise ValueError("The five folders overlap or contain duplicates")
    if set(combined.tolist()) != set(range(sample_count)):
        raise ValueError("Folder indices do not match the cleaned data")
    if max(len(fold) for fold in folds) - min(len(fold) for fold in folds) > 1:
        raise ValueError("Folder sizes differ by more than one sample")


def create_fivefold_split(
    source_dir: Path, output_dir: Path, seed: int
) -> dict[str, Any]:
    """Create the same five-folder data layout used by ThermalProcess."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = pd.read_csv(source_dir / "cleaned_tg.csv")
    sample_count = len(cleaned)

    # The original project stores five independently shuffled index arrays.
    # Its generation script is absent, so reproduce that format with a fixed
    # RandomState permutation and five equal consecutive chunks.
    permutation = np.random.RandomState(seed).permutation(sample_count)
    folds = [
        np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(permutation, 5)
    ]
    _validate_folds(folds, sample_count)

    folder = np.empty(sample_count, dtype=np.int8)
    role = np.empty(sample_count, dtype=object)
    for number, indices in enumerate(folds, start=1):
        folder[indices] = number
        role[indices] = "heldout_test" if number == 1 else "cv_pool"

    output_cleaned = cleaned.copy()
    output_cleaned["folder"] = folder
    output_cleaned["split"] = role
    output_cleaned.to_csv(output_dir / "cleaned_tg.csv", index=False)
    output_cleaned[
        ["sample_index", "source_rows", "canonical_psmiles", "tg", "folder", "split"]
    ].to_csv(output_dir / "split_assignments.csv", index=False)

    split_payload = {f"fold{i}": fold for i, fold in enumerate(folds, start=1)}
    split_payload["train_pool"] = np.concatenate(folds[1:])
    split_payload["test"] = folds[0]
    np.savez(output_dir / "split_indices.npz", **split_payload)
    for number, indices in enumerate(folds, start=1):
        np.save(output_dir / f"tg_{sample_count}_indices_folder{number}.npy", indices)

    for filename in [
        *FINGERPRINT_FILES.values(),
        "fingerprint_ptfp.npy",
        "excluded_rows.csv",
    ]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    summary = {
        "protocol": "ThermalProcess five-folder held-out-test protocol",
        "seed": seed,
        "sample_count": sample_count,
        "folder_sizes": {
            f"fold{i}": int(len(fold)) for i, fold in enumerate(folds, start=1)
        },
        "heldout_test_folder": "fold1",
        "heldout_test_samples": int(len(folds[0])),
        "cv_pool_folders": ["fold2", "fold3", "fold4", "fold5"],
        "cv_train_samples_per_split": int(sum(len(fold) for fold in folds[1:4])),
        "cv_validation_samples_per_split": int(len(folds[1])),
        "final_train_samples": int(sum(len(fold) for fold in folds[1:])),
        "test_fraction": float(len(folds[0]) / sample_count),
        "selection_rule": "highest mean validation R2 across four CV folds",
        "test_is_excluded_from_all_hyperparameter_search": True,
        "source_artifacts": str(source_dir.resolve()),
    }
    _write_json(output_dir / "split_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2), flush=True)
    return summary


def _classical_estimator_and_grid(model_name: str, seed: int, n_jobs: int):
    if model_name == "rf":
        return (
            RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
            {
                "n_estimators": [300],
                "max_depth": [None, 20, 50],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.5],
            },
            1,
        )
    if model_name == "svr":
        return (
            SVR(kernel="rbf"),
            {
                "C": [10.0, 100.0, 1000.0],
                "epsilon": [0.1, 1.0, 5.0],
                "gamma": ["scale", 0.001],
            },
            n_jobs,
        )
    raise ValueError(f"Unknown classical model: {model_name}")


def _cv_layout(folds: list[np.ndarray]) -> tuple[np.ndarray, PredefinedSplit]:
    train_pool = np.concatenate(folds[1:])
    validation_group = np.concatenate(
        [np.full(len(fold), group, dtype=int) for group, fold in enumerate(folds[1:])]
    )
    return train_pool, PredefinedSplit(validation_group)


def _prediction_frame(
    cleaned: pd.DataFrame,
    indices: np.ndarray,
    predictions: np.ndarray,
    fingerprint: str,
    model_name: str,
    stage: str,
) -> pd.DataFrame:
    selected = cleaned.iloc[indices]
    return pd.DataFrame(
        {
            "fingerprint": fingerprint,
            "model": model_name,
            "stage": stage,
            "sample_index": indices,
            "folder": selected["folder"].to_numpy(),
            "canonical_psmiles": selected["canonical_psmiles"].to_numpy(),
            "tg_actual": selected["tg"].to_numpy(dtype=float),
            "tg_predicted": predictions,
        }
    )


def optimize_classical(
    output_dir: Path,
    fingerprints: Iterable[str],
    models: Iterable[str],
    seed: int,
    n_jobs: int,
) -> None:
    cleaned = pd.read_csv(output_dir / "cleaned_tg.csv")
    y = cleaned["tg"].to_numpy(dtype=float)
    folds = _load_folds(output_dir)
    _validate_folds(folds, len(cleaned))
    train_pool, predefined = _cv_layout(folds)
    test = folds[0]

    results_dir = output_dir / "results"
    search_dir = results_dir / "search"
    models_dir = results_dir / "models"
    predictions_dir = results_dir / "predictions"
    for directory in (search_dir, models_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    best_path = results_dir / "best_parameters_classical.json"
    metrics_path = results_dir / "metrics_classical.csv"
    if best_path.exists():
        with best_path.open(encoding="utf-8") as handle:
            best_parameters: dict[str, Any] = json.load(handle)
    else:
        best_parameters = {}
    if metrics_path.exists():
        metric_rows: list[dict[str, Any]] = pd.read_csv(metrics_path).to_dict("records")
    else:
        metric_rows = []
    for fingerprint in fingerprints:
        X = np.load(output_dir / FINGERPRINT_FILES[fingerprint], mmap_mode="r")
        X_pool = np.asarray(X[train_pool], dtype=np.float32)
        y_pool = y[train_pool]
        for model_name in models:
            combination = f"{fingerprint}__{model_name}"
            if (
                combination in best_parameters
                and (models_dir / f"{combination}.joblib").exists()
                and (predictions_dir / f"{combination}.csv").exists()
            ):
                print(f"skip completed {combination}", flush=True)
                continue
            estimator, grid, search_jobs = _classical_estimator_and_grid(
                model_name, seed, n_jobs
            )
            print(
                f"search {combination}: {len(list(ParameterGrid(grid)))} candidates, "
                "4 CV folds, held-out fold1 excluded",
                flush=True,
            )
            started = time.time()
            search = GridSearchCV(
                estimator,
                grid,
                scoring={
                    "r2": "r2",
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                },
                refit="r2",
                cv=predefined,
                n_jobs=search_jobs,
                return_train_score=True,
                error_score="raise",
            )
            # Threading avoids the host's restricted process-semaphore API.
            # Every candidate fit is independent and numerically unchanged.
            with joblib.parallel_backend("threading", n_jobs=search_jobs):
                search.fit(X_pool, y_pool)

            cv = pd.DataFrame(search.cv_results_)
            keep = [
                "params",
                "mean_fit_time",
                "mean_score_time",
                "mean_train_r2",
                "mean_test_r2",
                "std_test_r2",
                "mean_train_rmse",
                "mean_test_rmse",
                "mean_train_mae",
                "mean_test_mae",
                "rank_test_r2",
            ]
            search_frame = cv[keep].copy()
            search_frame["params"] = search_frame["params"].map(
                lambda value: json.dumps(_json_safe(value), sort_keys=True)
            )
            for column in (
                "mean_train_rmse",
                "mean_test_rmse",
                "mean_train_mae",
                "mean_test_mae",
            ):
                search_frame[column] = -search_frame[column]
            search_frame.to_csv(search_dir / f"{combination}.csv", index=False)

            best_estimator = search.best_estimator_
            oof = np.empty(len(train_pool), dtype=float)
            fold_scores: list[dict[str, Any]] = []
            for cv_number, (local_train, local_validation) in enumerate(
                predefined.split(), start=2
            ):
                fold_model = clone(estimator).set_params(**search.best_params_)
                fold_model.fit(X_pool[local_train], y_pool[local_train])
                fold_prediction = fold_model.predict(X_pool[local_validation])
                oof[local_validation] = fold_prediction
                fold_scores.append(
                    {
                        "validation_folder": f"fold{cv_number}",
                        "n_train": int(len(local_train)),
                        "n_validation": int(len(local_validation)),
                        **metrics(y_pool[local_validation], fold_prediction),
                    }
                )

            test_prediction = best_estimator.predict(
                np.asarray(X[test], dtype=np.float32)
            )
            pool_prediction = best_estimator.predict(X_pool)
            oof_metrics = metrics(y_pool, oof)
            test_metrics = metrics(y[test], test_prediction)
            pool_metrics = metrics(y_pool, pool_prediction)
            best_parameters[combination] = {
                "parameters": search.best_params_,
                "selection_metric": "mean_validation_r2_across_fold2_to_fold5",
                "mean_validation_r2": float(search.best_score_),
                "mean_validation_rmse": float(
                    -cv.loc[search.best_index_, "mean_test_rmse"]
                ),
                "mean_validation_mae": float(
                    -cv.loc[search.best_index_, "mean_test_mae"]
                ),
                "candidate_count": int(len(list(ParameterGrid(grid)))),
                "fold_scores": fold_scores,
                "heldout_test": test_metrics,
                "elapsed_seconds": float(time.time() - started),
            }

            for stage, indices, values in (
                ("cv_validation_oof", train_pool, oof_metrics),
                ("train_pool_refit", train_pool, pool_metrics),
                ("heldout_test", test, test_metrics),
            ):
                metric_rows.append(
                    {
                        "fingerprint": fingerprint,
                        "model": model_name,
                        "stage": stage,
                        "n_samples": int(len(indices)),
                        **values,
                    }
                )

            predictions = pd.concat(
                [
                    _prediction_frame(
                        cleaned,
                        train_pool,
                        oof,
                        fingerprint,
                        model_name,
                        "cv_validation_oof",
                    ),
                    _prediction_frame(
                        cleaned,
                        test,
                        test_prediction,
                        fingerprint,
                        model_name,
                        "heldout_test",
                    ),
                ],
                ignore_index=True,
            )
            predictions.to_csv(predictions_dir / f"{combination}.csv", index=False)
            joblib.dump(
                best_estimator, models_dir / f"{combination}.joblib", compress=3
            )
            _write_json(best_path, best_parameters)
            pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
            print(
                f"finished {combination} in {time.time() - started:.1f}s; "
                f"CV R2={search.best_score_:.4f}; held-out RMSE={test_metrics['rmse']:.3f}",
                flush=True,
            )

    summary_rows: list[dict[str, Any]] = []
    for combination, result in sorted(best_parameters.items()):
        fingerprint, model_name = combination.split("__", maxsplit=1)
        fold_scores = result["fold_scores"]
        row: dict[str, Any] = {
            "fingerprint": fingerprint,
            "model": model_name,
            "cv_folds": len(fold_scores),
            "cv_train_samples_per_fold": fold_scores[0]["n_train"],
            "cv_validation_samples": ";".join(
                str(score["n_validation"]) for score in fold_scores
            ),
            "heldout_samples": len(test),
            "best_parameters": json.dumps(
                _json_safe(result["parameters"]), sort_keys=True
            ),
            "cv_r2_mean": result["mean_validation_r2"],
            "cv_r2_std_ddof0": float(
                np.std([score["r2"] for score in fold_scores], ddof=0)
            ),
            "cv_rmse_mean": result["mean_validation_rmse"],
            "cv_rmse_std_ddof0": float(
                np.std([score["rmse"] for score in fold_scores], ddof=0)
            ),
            "heldout_r2": result["heldout_test"]["r2"],
            "heldout_rmse": result["heldout_test"]["rmse"],
            "heldout_mae": result["heldout_test"]["mae"],
        }
        summary_rows.append(row)
    pd.DataFrame(summary_rows).sort_values(["model", "fingerprint"]).to_csv(
        results_dir / "classical_optimization_heldout_summary.csv", index=False
    )


def evaluate_classical_fixed_fivefold(
    output_dir: Path,
    fingerprints: Iterable[str],
    models: Iterable[str],
    seed: int,
    n_jobs: int,
    force: bool = False,
) -> None:
    """Evaluate fixed optimized RF/SVR parameters with each folder held out once.

    The parameters are read from ``best_parameters_classical.json`` and are
    never searched or changed here. For every test folder, the other four
    folders (5,736 samples) are used to fit a fresh model. Reported standard
    deviations use NumPy's population definition (``ddof=0``), matching the
    existing exact-GP aggregation in this module.
    """
    fingerprints = list(fingerprints)
    models = list(models)
    cleaned = pd.read_csv(output_dir / "cleaned_tg.csv")
    y = cleaned["tg"].to_numpy(dtype=float)
    folds = _load_folds(output_dir)
    _validate_folds(folds, len(cleaned))

    results_dir = output_dir / "results"
    best_path = results_dir / "best_parameters_classical.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Fixed-parameter evaluation requires optimized parameters: {best_path}"
        )
    with best_path.open(encoding="utf-8") as handle:
        best_parameters: dict[str, Any] = json.load(handle)

    evaluation_dir = results_dir / "fixed_parameter_fivefold"
    task_dir = evaluation_dir / "tasks"
    prediction_task_dir = evaluation_dir / "prediction_tasks"
    for directory in (evaluation_dir, task_dir, prediction_task_dir):
        directory.mkdir(parents=True, exist_ok=True)

    expected_combinations = [
        f"{fingerprint}__{model_name}"
        for fingerprint in fingerprints
        for model_name in models
    ]
    missing = [name for name in expected_combinations if name not in best_parameters]
    if missing:
        raise KeyError(f"Missing optimized parameters for: {', '.join(missing)}")

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fingerprint in fingerprints:
        X = np.load(output_dir / FINGERPRINT_FILES[fingerprint], mmap_mode="r")
        for model_name in models:
            combination = f"{fingerprint}__{model_name}"
            parameters = best_parameters[combination]["parameters"]
            estimator, _, _ = _classical_estimator_and_grid(model_name, seed, n_jobs)
            for fold_number, test_indices in enumerate(folds, start=1):
                task_path = task_dir / f"{combination}__fold{fold_number}.json"
                prediction_path = (
                    prediction_task_dir / f"{combination}__fold{fold_number}.csv"
                )
                if task_path.exists() and prediction_path.exists() and not force:
                    with task_path.open(encoding="utf-8") as handle:
                        fold_result = json.load(handle)
                    prediction_frame = pd.read_csv(prediction_path)
                    print(f"skip completed {combination} fold{fold_number}", flush=True)
                else:
                    train_indices = np.concatenate(
                        [
                            fold
                            for index, fold in enumerate(folds, start=1)
                            if index != fold_number
                        ]
                    )
                    model = clone(estimator).set_params(**parameters)
                    X_train = np.asarray(X[train_indices], dtype=np.float32)
                    X_test = np.asarray(X[test_indices], dtype=np.float32)
                    print(
                        f"fit {combination}: test=fold{fold_number}, "
                        f"train={len(train_indices)}, test_samples={len(test_indices)}",
                        flush=True,
                    )
                    started = time.time()
                    if model_name == "rf":
                        with joblib.parallel_backend("threading", n_jobs=n_jobs):
                            model.fit(X_train, y[train_indices])
                    else:
                        model.fit(X_train, y[train_indices])
                    prediction = model.predict(X_test)
                    fold_result = {
                        "fingerprint": fingerprint,
                        "model": model_name,
                        "test_fold": f"fold{fold_number}",
                        "n_train": int(len(train_indices)),
                        "n_test": int(len(test_indices)),
                        "parameters": parameters,
                        "fit_and_predict_seconds": float(time.time() - started),
                        **metrics(y[test_indices], prediction),
                    }
                    prediction_frame = _prediction_frame(
                        cleaned,
                        test_indices,
                        prediction,
                        fingerprint,
                        model_name,
                        "fixed_parameter_test",
                    )
                    prediction_frame.insert(3, "test_fold", f"fold{fold_number}")
                    _write_json(task_path, fold_result)
                    prediction_frame.to_csv(prediction_path, index=False)
                    print(
                        f"finished {combination} fold{fold_number}: "
                        f"R2={fold_result['r2']:.4f}, "
                        f"RMSE={fold_result['rmse']:.3f}, "
                        f"MAE={fold_result['mae']:.3f}",
                        flush=True,
                    )
                fold_rows.append(fold_result)
                prediction_frames.append(prediction_frame)

    fold_frame = pd.DataFrame(
        [
            {
                "fingerprint": row["fingerprint"],
                "model": row["model"],
                "test_fold": row["test_fold"],
                "n_train": row["n_train"],
                "n_test": row["n_test"],
                "r2": row["r2"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "fit_and_predict_seconds": row["fit_and_predict_seconds"],
                "parameters": json.dumps(_json_safe(row["parameters"]), sort_keys=True),
            }
            for row in fold_rows
        ]
    ).sort_values(["fingerprint", "model", "test_fold"])

    summary_rows: list[dict[str, Any]] = []
    for (fingerprint, model_name), group in fold_frame.groupby(
        ["fingerprint", "model"], sort=True
    ):
        group = group.sort_values("test_fold")
        row: dict[str, Any] = {
            "fingerprint": fingerprint,
            "model": model_name,
            "n_folds": int(len(group)),
            "n_train_by_fold": ";".join(group["n_train"].astype(int).astype(str)),
            "n_test_by_fold": ";".join(group["n_test"].astype(int).astype(str)),
            "std_ddof": 0,
        }
        for metric_name in ("r2", "rmse", "mae"):
            values = group[metric_name].to_numpy(dtype=float)
            row[f"{metric_name}_mean"] = float(np.mean(values))
            row[f"{metric_name}_std"] = float(np.std(values, ddof=0))
        row["parameters"] = group["parameters"].iloc[0]
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows).sort_values(["model", "fingerprint"])

    fold_frame.to_csv(evaluation_dir / "fold_metrics.csv", index=False)
    summary_frame.to_csv(evaluation_dir / "summary_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        evaluation_dir / "predictions.csv", index=False
    )
    _write_json(
        evaluation_dir / "run_config.json",
        {
            "protocol": "fixed optimized parameters; each fold1-fold5 is test once; other four folds train",
            "sample_count": int(len(cleaned)),
            "n_folds": 5,
            "n_train_by_fold": {
                f"fold{fold_number}": int(len(cleaned) - len(fold))
                for fold_number, fold in enumerate(folds, start=1)
            },
            "n_test_by_fold": {
                f"fold{fold_number}": int(len(fold))
                for fold_number, fold in enumerate(folds, start=1)
            },
            "parameter_source": str(best_path.resolve()),
            "parameters_held_fixed": True,
            "standard_deviation_ddof": 0,
            "seed": seed,
            "fingerprints": list(fingerprints),
            "models": list(models),
        },
    )
    print(summary_frame.to_string(index=False), flush=True)


def _gp_cv_task(payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(payload["output_dir"])
    fingerprint = payload["fingerprint"]
    alpha = float(payload["alpha"])
    validation_folder = int(payload["validation_folder"])
    n_restarts = int(payload["n_restarts"])
    seed = int(payload["seed"])
    destination = Path(payload["destination"])
    if destination.exists():
        with destination.open(encoding="utf-8") as handle:
            return json.load(handle)

    cleaned = pd.read_csv(output_dir / "cleaned_tg.csv")
    y = cleaned["tg"].to_numpy(dtype=float)
    folds = _load_folds(output_dir)
    validation = folds[validation_folder - 1]
    train = np.concatenate(
        [folds[index - 1] for index in range(2, 6) if index != validation_folder]
    )
    X = np.load(output_dir / FINGERPRINT_FILES[fingerprint], mmap_mode="r")
    model = GaussianProcessRegressor(
        kernel=exact_kernel(cache_key=f"{fingerprint}-validation{validation_folder}"),
        alpha=alpha,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )
    started = time.time()
    model.fit(np.asarray(X[train], dtype=np.float64), y[train])
    fit_seconds = time.time() - started
    prediction = model.predict(np.asarray(X[validation], dtype=np.float64))
    result = {
        "fingerprint": fingerprint,
        "alpha": alpha,
        "n_restarts_optimizer": n_restarts,
        "validation_folder": f"fold{validation_folder}",
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "fit_seconds": fit_seconds,
        "learned_kernel": str(model.kernel_),
        "log_marginal_likelihood": float(model.log_marginal_likelihood_value_),
        **metrics(y[validation], prediction),
    }
    _write_json(destination, result)
    pd.DataFrame(
        {
            "sample_index": validation,
            "tg_actual": y[validation],
            "tg_predicted": prediction,
        }
    ).to_csv(destination.with_name(destination.stem + "__predictions.csv"), index=False)
    return result


def _gp_search_tasks(
    output_dir: Path,
    fingerprints: Iterable[str],
    alphas: Iterable[float],
    n_restarts: int,
    seed: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    task_dir = output_dir / "results" / "search" / "exact_gp_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for fingerprint in fingerprints:
        for alpha in alphas:
            for validation_folder in range(2, 6):
                tag = f"{fingerprint}__alpha{alpha:g}__validation_fold{validation_folder}.json"
                tasks.append(
                    {
                        "output_dir": str(output_dir),
                        "fingerprint": fingerprint,
                        "alpha": float(alpha),
                        "validation_folder": validation_folder,
                        "n_restarts": n_restarts,
                        "seed": seed,
                        "destination": str(task_dir / tag),
                    }
                )
    return tasks


def _collect_gp_search(
    output_dir: Path,
    fingerprints: Iterable[str],
    alphas: Iterable[float],
    n_restarts: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    task_dir = output_dir / "results" / "search" / "exact_gp_tasks"
    rows: list[dict[str, Any]] = []
    fingerprints = list(fingerprints)
    for fingerprint in fingerprints:
        for alpha in alphas:
            fold_rows = []
            for validation_folder in range(2, 6):
                path = task_dir / (
                    f"{fingerprint}__alpha{alpha:g}__validation_fold{validation_folder}.json"
                )
                if not path.exists():
                    raise RuntimeError(f"Missing exact-GP task result: {path}")
                with path.open(encoding="utf-8") as handle:
                    fold_rows.append(json.load(handle))
            rows.append(
                {
                    "fingerprint": fingerprint,
                    "alpha": float(alpha),
                    "n_restarts_optimizer": n_restarts,
                    "mean_validation_r2": float(
                        np.mean([row["r2"] for row in fold_rows])
                    ),
                    "std_validation_r2": float(
                        np.std([row["r2"] for row in fold_rows])
                    ),
                    "mean_validation_rmse": float(
                        np.mean([row["rmse"] for row in fold_rows])
                    ),
                    "mean_validation_mae": float(
                        np.mean([row["mae"] for row in fold_rows])
                    ),
                    "total_fit_seconds": float(
                        sum(row["fit_seconds"] for row in fold_rows)
                    ),
                    "cv_reused_from": None,
                }
            )
    frame = pd.DataFrame(rows)
    best_alpha = {
        fingerprint: float(
            frame[frame["fingerprint"] == fingerprint]
            .sort_values(
                ["mean_validation_r2", "mean_validation_rmse"], ascending=[False, True]
            )
            .iloc[0]["alpha"]
        )
        for fingerprint in fingerprints
    }
    return frame, best_alpha


def _fit_final_gp(
    output_dir: Path,
    fingerprint: str,
    alpha: float,
    n_restarts: int,
    seed: int,
) -> dict[str, Any]:
    results_dir = output_dir / "results"
    final_dir = results_dir / "final_exact_gp"
    models_dir = results_dir / "models"
    predictions_dir = results_dir / "predictions"
    for directory in (final_dir, models_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)
    result_path = final_dir / f"{fingerprint}__alpha{alpha:g}.json"
    if result_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    cleaned = pd.read_csv(output_dir / "cleaned_tg.csv")
    y = cleaned["tg"].to_numpy(dtype=float)
    folds = _load_folds(output_dir)
    train_pool = np.concatenate(folds[1:])
    test = folds[0]
    X = np.load(output_dir / FINGERPRINT_FILES[fingerprint], mmap_mode="r")
    model = GaussianProcessRegressor(
        kernel=exact_kernel(cache_key=f"{fingerprint}-final"),
        alpha=alpha,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )
    started = time.time()
    model.fit(np.asarray(X[train_pool], dtype=np.float64), y[train_pool])
    fit_seconds = time.time() - started
    test_prediction = model.predict(np.asarray(X[test], dtype=np.float64))
    train_prediction = model.predict(np.asarray(X[train_pool], dtype=np.float64))
    result = {
        "fingerprint": fingerprint,
        "alpha": alpha,
        "n_restarts_optimizer": n_restarts,
        "n_train": int(len(train_pool)),
        "n_heldout_test": int(len(test)),
        "fit_seconds": fit_seconds,
        "learned_kernel": str(model.kernel_),
        "log_marginal_likelihood": float(model.log_marginal_likelihood_value_),
        "train_pool": metrics(y[train_pool], train_prediction),
        "heldout_test": metrics(y[test], test_prediction),
    }
    _write_json(result_path, result)
    _prediction_frame(
        cleaned, test, test_prediction, fingerprint, "gp", "heldout_test"
    ).to_csv(predictions_dir / f"{fingerprint}__gp.csv", index=False)
    joblib.dump(model, models_dir / f"{fingerprint}__gp.joblib", compress=3)
    return result


def optimize_exact_gp(
    output_dir: Path,
    fingerprints: Iterable[str],
    alphas: Iterable[float],
    n_restarts: int,
    seed: int,
    max_workers: int,
    search_only: bool,
) -> None:
    fingerprints = list(fingerprints)
    alphas = list(alphas)

    tasks = _gp_search_tasks(output_dir, fingerprints, alphas, n_restarts, seed)
    pending = [task for task in tasks if not Path(task["destination"]).exists()]
    print(
        f"exact GP CV: {len(tasks)} unique fits total, {len(pending)} pending, "
        f"workers={max_workers}; held-out fold1 excluded",
        flush=True,
    )
    if max_workers == 1:
        for completed, task in enumerate(pending, start=1):
            result = _gp_cv_task(task)
            print(
                f"GP task {completed}/{len(pending)} finished: {result['fingerprint']} "
                f"alpha={result['alpha']:g} {result['validation_folder']} "
                f"R2={result['r2']:.4f} fit={result['fit_seconds'] / 60:.1f}min",
                flush=True,
            )
    else:
        # Threads avoid the host's restricted process-semaphore API. NumPy,
        # SciPy and scikit-learn release the GIL in the heavy linear-algebra
        # sections, so independent folds can still make concurrent progress.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gp_cv_task, task): task for task in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                print(
                    f"GP task {completed}/{len(pending)} finished: {result['fingerprint']} "
                    f"alpha={result['alpha']:g} {result['validation_folder']} "
                    f"R2={result['r2']:.4f} fit={result['fit_seconds'] / 60:.1f}min",
                    flush=True,
                )

    search_frame, best_alpha = _collect_gp_search(
        output_dir, fingerprints, alphas, n_restarts
    )
    search_path = output_dir / "results" / "search" / "exact_gp.csv"
    search_frame.to_csv(search_path, index=False)
    _write_json(
        output_dir / "results" / "best_parameters_exact_gp.json",
        {
            fingerprint: {
                "alpha": alpha,
                "n_restarts_optimizer": n_restarts,
                "selection_metric": "mean_validation_r2_across_fold2_to_fold5",
            }
            for fingerprint, alpha in best_alpha.items()
        },
    )
    print(f"best exact-GP alphas: {best_alpha}", flush=True)
    if search_only:
        return

    final_results = []
    for fingerprint in fingerprints:
        print(
            f"final exact GP fit: {fingerprint}, alpha={best_alpha[fingerprint]:g}",
            flush=True,
        )
        result = _fit_final_gp(
            output_dir, fingerprint, best_alpha[fingerprint], n_restarts, seed
        )
        final_results.append(result)
        print(
            f"finished final {fingerprint} GP: held-out R2={result['heldout_test']['r2']:.4f}, "
            f"RMSE={result['heldout_test']['rmse']:.3f}",
            flush=True,
        )
    _write_json(
        output_dir / "results" / "final_exact_gp_summary.json",
        {"results": final_results},
    )


def run_exact_gp_array_task(
    output_dir: Path,
    task_id: int,
    alphas: Iterable[float],
    n_restarts: int,
    seed: int,
) -> dict[str, Any]:
    """Run one SCC/Slurm CV task from the 3 fingerprints x alphas x 4 folds grid."""
    alphas = list(alphas)
    unique_fingerprints = ["morgan", "pecfp", "bigfp"]
    task_count = len(unique_fingerprints) * len(alphas) * 4
    if task_id < 0 or task_id >= task_count:
        raise ValueError(f"task_id must be in [0, {task_count - 1}], got {task_id}")
    fingerprint_block, within_fingerprint = divmod(task_id, len(alphas) * 4)
    alpha_index, fold_offset = divmod(within_fingerprint, 4)
    fingerprint = unique_fingerprints[fingerprint_block]
    alpha = float(alphas[alpha_index])
    validation_folder = fold_offset + 2
    destination = (
        output_dir
        / "results"
        / "search"
        / "exact_gp_tasks"
        / f"{fingerprint}__alpha{alpha:g}__validation_fold{validation_folder}.json"
    )
    payload = {
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "alpha": alpha,
        "validation_folder": validation_folder,
        "n_restarts": n_restarts,
        "seed": seed,
        "destination": str(destination),
    }
    print(
        f"array task {task_id}/{task_count - 1}: {fingerprint}, alpha={alpha:g}, "
        f"validation=fold{validation_folder}",
        flush=True,
    )
    result = _gp_cv_task(payload)
    print(json.dumps(_json_safe(result), indent=2), flush=True)
    return result


def collect_exact_gp_search(
    output_dir: Path,
    fingerprints: Iterable[str],
    alphas: Iterable[float],
    n_restarts: int,
) -> dict[str, float]:
    """Aggregate completed SCC CV tasks and select alpha without fitting anything."""
    fingerprints = list(fingerprints)
    frame, best_alpha = _collect_gp_search(output_dir, fingerprints, alphas, n_restarts)
    search_dir = output_dir / "results" / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(search_dir / "exact_gp.csv", index=False)
    _write_json(
        output_dir / "results" / "best_parameters_exact_gp.json",
        {
            fingerprint: {
                "alpha": alpha,
                "n_restarts_optimizer": n_restarts,
                "selection_metric": "mean_validation_r2_across_fold2_to_fold5",
            }
            for fingerprint, alpha in best_alpha.items()
        },
    )
    print(f"best exact-GP alphas: {best_alpha}", flush=True)
    return best_alpha


def run_final_exact_gp_array_task(
    output_dir: Path,
    task_id: int,
    n_restarts: int,
    seed: int,
) -> dict[str, Any]:
    """Fit one of the three final fingerprints and predict held-out folder 1."""
    fingerprints = list(FINGERPRINTS)
    if task_id < 0 or task_id >= len(fingerprints):
        raise ValueError(
            f"task_id must be in [0, {len(fingerprints) - 1}], got {task_id}"
        )
    best_path = output_dir / "results" / "best_parameters_exact_gp.json"
    with best_path.open(encoding="utf-8") as handle:
        best = json.load(handle)
    fingerprint = fingerprints[task_id]
    alpha = float(best[fingerprint]["alpha"])
    print(f"final array task {task_id}: {fingerprint}, alpha={alpha:g}", flush=True)
    result = _fit_final_gp(output_dir, fingerprint, alpha, n_restarts, seed)
    print(json.dumps(_json_safe(result), indent=2), flush=True)
    return result


def summarize_final_exact_gp(output_dir: Path) -> dict[str, Any]:
    """Collect the three final SCC fits and verify held-out prediction integrity."""
    best_path = output_dir / "results" / "best_parameters_exact_gp.json"
    with best_path.open(encoding="utf-8") as handle:
        best = json.load(handle)
    split_file = np.load(output_dir / "split_indices.npz")
    heldout = set(split_file["fold1"].tolist())
    results = []
    combined_predictions = []
    for fingerprint in FINGERPRINTS:
        alpha = float(best[fingerprint]["alpha"])
        result_path = (
            output_dir
            / "results"
            / "final_exact_gp"
            / f"{fingerprint}__alpha{alpha:g}.json"
        )
        if not result_path.exists():
            raise RuntimeError(f"Missing final GP result: {result_path}")
        with result_path.open(encoding="utf-8") as handle:
            results.append(json.load(handle))
        prediction_path = (
            output_dir / "results" / "predictions" / f"{fingerprint}__gp.csv"
        )
        prediction = pd.read_csv(prediction_path)
        predicted_indices = set(prediction["sample_index"].astype(int).tolist())
        if predicted_indices != heldout or len(prediction) != len(heldout):
            raise RuntimeError(f"Held-out predictions are incomplete for {fingerprint}")
        if prediction.isna().any().any():
            raise RuntimeError(
                f"Held-out predictions contain missing values for {fingerprint}"
            )
        combined_predictions.append(prediction)
    payload = {
        "protocol": "fold1 held-out; fold2-fold5 four-fold alpha optimization; final refit on fold2-fold5",
        "results": results,
    }
    results_dir = output_dir / "results"
    _write_json(results_dir / "final_exact_gp_summary.json", payload)
    pd.concat(combined_predictions, ignore_index=True).to_csv(
        results_dir / "predictions_exact_gp_heldout.csv", index=False
    )
    print(json.dumps(_json_safe(payload), indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    split.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    split.add_argument("--seed", type=int, default=42)

    classical = subparsers.add_parser("classical")
    classical.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    classical.add_argument(
        "--fingerprints", nargs="+", choices=FINGERPRINTS, default=list(FINGERPRINTS)
    )
    classical.add_argument(
        "--models", nargs="+", choices=CLASSICAL_MODELS, default=list(CLASSICAL_MODELS)
    )
    classical.add_argument("--seed", type=int, default=42)
    classical.add_argument("--n-jobs", type=int, default=-1)

    fixed_cv = subparsers.add_parser("classical-fixed-cv")
    fixed_cv.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    fixed_cv.add_argument(
        "--fingerprints", nargs="+", choices=FINGERPRINTS, default=list(FINGERPRINTS)
    )
    fixed_cv.add_argument(
        "--models", nargs="+", choices=CLASSICAL_MODELS, default=list(CLASSICAL_MODELS)
    )
    fixed_cv.add_argument("--seed", type=int, default=42)
    fixed_cv.add_argument("--n-jobs", type=int, default=-1)
    fixed_cv.add_argument("--force", action="store_true")

    gp = subparsers.add_parser("exact-gp")
    gp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    gp.add_argument(
        "--fingerprints", nargs="+", choices=FINGERPRINTS, default=list(FINGERPRINTS)
    )
    gp.add_argument("--alphas", nargs="+", type=float, default=list(GP_ALPHAS))
    gp.add_argument("--n-restarts", type=int, default=20)
    gp.add_argument("--seed", type=int, default=42)
    gp.add_argument("--max-workers", type=int, default=1)
    gp.add_argument("--search-only", action="store_true")

    gp_task = subparsers.add_parser("gp-task")
    gp_task.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    gp_task.add_argument("--task-id", type=int, required=True)
    gp_task.add_argument("--alphas", nargs="+", type=float, default=list(GP_ALPHAS))
    gp_task.add_argument("--n-restarts", type=int, default=20)
    gp_task.add_argument("--seed", type=int, default=42)

    gp_collect = subparsers.add_parser("gp-collect")
    gp_collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    gp_collect.add_argument(
        "--fingerprints", nargs="+", choices=FINGERPRINTS, default=list(FINGERPRINTS)
    )
    gp_collect.add_argument("--alphas", nargs="+", type=float, default=list(GP_ALPHAS))
    gp_collect.add_argument("--n-restarts", type=int, default=20)

    gp_final = subparsers.add_parser("gp-final-task")
    gp_final.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    gp_final.add_argument("--task-id", type=int, required=True)
    gp_final.add_argument("--n-restarts", type=int, default=20)
    gp_final.add_argument("--seed", type=int, default=42)

    gp_summary = subparsers.add_parser("gp-summary")
    gp_summary.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "split":
        create_fivefold_split(
            args.source_dir.resolve(), args.output_dir.resolve(), args.seed
        )
    elif args.command == "classical":
        optimize_classical(
            args.output_dir.resolve(),
            args.fingerprints,
            args.models,
            args.seed,
            args.n_jobs,
        )
    elif args.command == "classical-fixed-cv":
        evaluate_classical_fixed_fivefold(
            args.output_dir.resolve(),
            args.fingerprints,
            args.models,
            args.seed,
            args.n_jobs,
            args.force,
        )
    elif args.command == "exact-gp":
        optimize_exact_gp(
            args.output_dir.resolve(),
            args.fingerprints,
            args.alphas,
            args.n_restarts,
            args.seed,
            args.max_workers,
            args.search_only,
        )
    elif args.command == "gp-task":
        run_exact_gp_array_task(
            args.output_dir.resolve(),
            args.task_id,
            args.alphas,
            args.n_restarts,
            args.seed,
        )
    elif args.command == "gp-collect":
        collect_exact_gp_search(
            args.output_dir.resolve(), args.fingerprints, args.alphas, args.n_restarts
        )
    elif args.command == "gp-final-task":
        run_final_exact_gp_array_task(
            args.output_dir.resolve(), args.task_id, args.n_restarts, args.seed
        )
    elif args.command == "gp-summary":
        summarize_final_exact_gp(args.output_dir.resolve())


if __name__ == "__main__":
    # Record BLAS threading in summaries/logs; set these before launching Python
    # when using multiple exact-GP workers.
    print(
        "thread env:",
        {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        flush=True,
    )
    main()

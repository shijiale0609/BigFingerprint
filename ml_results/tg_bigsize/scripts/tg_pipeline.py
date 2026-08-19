#!/usr/bin/env python3
"""End-to-end Tg benchmark for the PolyInfo p-SMILES data set.

The workflow deliberately keeps only linear repeat units with exactly two
terminal wildcard atoms.  It generates the same three representation families
used by the earlier ThermalProcess experiments and tunes RF, SVR and a scalable
Gaussian-process approximation on a held-out validation set.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from joblib import Memory
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_INPUT = HERE / "tg_raw.csv"
DEFAULT_OUTPUT = HERE / "artifacts"
FINGERPRINT_FILES = {
    "morgan": "fingerprint_morgan.npy",
    "pecfp": "fingerprint_pecfp.npy",
    "bigfp": "fingerprint_bigfp.npy",
}
FINGERPRINT_LABELS = {
    "morgan": "p-SMILES MorganFP",
    "pecfp": "p-ECFP",
    "bigfp": "BigFP (p-ECFP + p-TFP)",
}
MODEL_LABELS = {
    "rf": "Random forest",
    "svr": "Support vector regression",
    "gp": "Approximate GP (Nyström kernel ridge)",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _regression_bins(values: Iterable[float], n_bins: int) -> np.ndarray:
    series = pd.Series(np.asarray(list(values), dtype=float))
    q = min(n_bins, int(series.nunique()), len(series))
    if q < 2:
        return np.zeros(len(series), dtype=int)
    bins = pd.qcut(series, q=q, labels=False, duplicates="drop")
    result = bins.to_numpy(dtype=int)
    counts = pd.Series(result).value_counts()
    if counts.min() < 2:
        return np.zeros(len(series), dtype=int)
    return result


def _canonicalize_linear_psmiles(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "rdkit_parse_failed"
    wildcard_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(wildcard_atoms) != 2:
        return None, "non_linear_connection_count"
    if any(atom.GetDegree() != 1 for atom in wildcard_atoms):
        return None, "non_terminal_connection_point"
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical, None


def _psmiles_to_linear_bigsmiles(psmiles: str) -> str:
    """Encode a two-connection-point p-SMILES as one linear BigSMILES object."""
    if psmiles.count("*") != 2:
        raise ValueError(f"Expected exactly two '*' tokens: {psmiles}")
    repeat_unit = re.sub(r"\*", "[$]", psmiles)
    return "{[]" + repeat_unit + "[]}"


def _morgan_fp(mol: Chem.Mol, generator, fp_size: int) -> np.ndarray:
    bit_vector = generator.GetFingerprint(mol)
    array = np.zeros(fp_size, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def _periodic_oligomer(mol: Chem.Mol, copies: int = 7) -> tuple[Chem.Mol, list[int]]:
    """Build a capped oligomer and return atom indices in its central repeat.

    Seven copies are sufficient for every radius-2 environment centered in the
    middle repeat, including the one-atom repeat-unit edge case.  Terminal dummy
    atoms are retained only at the two remote chain ends; internal dummy pairs
    are replaced by the corresponding polymerization bonds.
    """
    dummy_indices = sorted(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    if len(dummy_indices) != 2:
        raise ValueError("Periodic p-ECFP requires exactly two connection atoms")
    if copies < 7 or copies % 2 == 0:
        raise ValueError("copies must be an odd integer >= 7")

    connections = []
    for dummy_index in dummy_indices:
        dummy = mol.GetAtomWithIdx(dummy_index)
        bond = next(iter(dummy.GetBonds()))
        connections.append((dummy.GetNeighbors()[0].GetIdx(), bond.GetBondType()))
    if connections[0][1] != connections[1][1]:
        raise ValueError("The two connection-point bond types do not match")

    atoms_per_copy = mol.GetNumAtoms()
    combined = mol
    for _ in range(copies - 1):
        combined = Chem.CombineMols(combined, mol)
    editable = Chem.RWMol(combined)

    removed_dummy_indices: list[int] = []
    for copy_index in range(copies - 1):
        editable.AddBond(
            copy_index * atoms_per_copy + connections[1][0],
            (copy_index + 1) * atoms_per_copy + connections[0][0],
            connections[0][1],
        )
        removed_dummy_indices.extend(
            [
                copy_index * atoms_per_copy + dummy_indices[1],
                (copy_index + 1) * atoms_per_copy + dummy_indices[0],
            ]
        )

    removed = set(removed_dummy_indices)
    middle_copy = copies // 2
    central_old_indices = [
        middle_copy * atoms_per_copy + atom_index
        for atom_index in range(atoms_per_copy)
        if atom_index not in dummy_indices
    ]
    central_new_indices = [
        old_index - sum(removed_index < old_index for removed_index in removed)
        for old_index in central_old_indices
    ]
    for dummy_index in sorted(removed_dummy_indices, reverse=True):
        editable.RemoveAtom(dummy_index)

    oligomer = editable.GetMol()
    Chem.SanitizeMol(oligomer)
    return oligomer, central_new_indices


def _periodic_pecfp(mol: Chem.Mol, generator, fp_size: int) -> np.ndarray:
    oligomer, central_atoms = _periodic_oligomer(mol, copies=7)
    bit_vector = generator.GetFingerprint(oligomer, fromAtoms=central_atoms)
    array = np.zeros(fp_size, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def _linear_homopolymer_topology_fp(mol: Chem.Mol) -> np.ndarray:
    """Return the exact 40-d p-TFP layout for one linear homopolymer object.

    For ``{[][$]repeat[$][]}``, the topology-search and network portions are
    constant: only the generic polymer bit is on.  Chirality and E/Z bits are
    still molecule-specific and occupy the same positions as polymer_topfp.py.
    """
    vector = np.zeros(40, dtype=np.uint8)
    vector[0] = 1  # generic polymer BigSMARTS

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    if any(label == "R" for _, label in centers):
        vector[17] = 1
    if any(label == "S" for _, label in centers):
        vector[18] = 1

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE:
            vector[19] = 1
        elif bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
            vector[20] = 1
    return vector


def _stratified_limit(frame: pd.DataFrame, limit: int, seed: int, n_bins: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame.reset_index(drop=True)
    chosen, _ = train_test_split(
        np.arange(len(frame)),
        train_size=limit,
        random_state=seed,
        shuffle=True,
        stratify=_regression_bins(frame["tg"], n_bins),
    )
    return frame.iloc[np.sort(chosen)].reset_index(drop=True)


def _make_split(
    y: np.ndarray,
    train_size: float,
    validation_size: float,
    test_size: float,
    n_bins: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1")
    all_indices = np.arange(len(y))
    test_count = int(round(test_size * len(y)))
    validation_count = int(round(validation_size * len(y)))
    train_validation, test = train_test_split(
        all_indices,
        test_size=test_count,
        random_state=seed,
        shuffle=True,
        stratify=_regression_bins(y, n_bins),
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_count,
        random_state=seed,
        shuffle=True,
        stratify=_regression_bins(y[train_validation], n_bins),
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


def prepare_dataset(
    input_csv: Path,
    output_dir: Path,
    seed: int = 42,
    train_size: float = 0.70,
    validation_size: float = 0.10,
    test_size: float = 0.20,
    n_bins: int = 10,
    fp_size: int = 2048,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_csv)
    required = {"SMILES", "tg"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    RDLogger.DisableLog("rdApp.error")
    valid_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    for row_index, row in raw.iterrows():
        smiles = str(row["SMILES"]).strip()
        try:
            tg = float(row["tg"])
        except (TypeError, ValueError):
            excluded_records.append(
                {"source_row": int(row_index + 2), "SMILES": smiles, "tg": row["tg"], "reason": "invalid_tg"}
            )
            continue
        if not math.isfinite(tg):
            excluded_records.append(
                {"source_row": int(row_index + 2), "SMILES": smiles, "tg": tg, "reason": "invalid_tg"}
            )
            continue
        canonical, reason = _canonicalize_linear_psmiles(smiles)
        if reason is not None:
            excluded_records.append(
                {"source_row": int(row_index + 2), "SMILES": smiles, "tg": tg, "reason": reason}
            )
            continue
        valid_records.append(
            {
                "source_row": int(row_index + 2),
                "source_smiles": smiles,
                "canonical_psmiles": canonical,
                "tg": tg,
            }
        )

    valid = pd.DataFrame(valid_records)
    if valid.empty:
        raise ValueError("No valid two-connection-point p-SMILES remained")

    # Aggregate canonical duplicates before splitting so that the same structure
    # can never leak across train, validation and test sets.
    grouped = (
        valid.groupby("canonical_psmiles", sort=True, as_index=False)
        .agg(
            source_smiles=("source_smiles", "first"),
            tg=("tg", "mean"),
            tg_std=("tg", "std"),
            measurement_count=("tg", "size"),
            source_rows=("source_row", lambda values: ";".join(str(int(v)) for v in values)),
        )
        .fillna({"tg_std": 0.0})
    )
    grouped = _stratified_limit(grouped, limit or 0, seed, n_bins)
    grouped["bigsmiles"] = grouped["canonical_psmiles"].map(_psmiles_to_linear_bigsmiles)

    retained_rows: list[int] = []
    morgan_vectors: list[np.ndarray] = []
    pecfp_vectors: list[np.ndarray] = []
    topology_vectors: list[np.ndarray] = []
    started = time.time()
    morgan_generator = GetMorganGenerator(radius=2, fpSize=fp_size, includeChirality=False)
    pecfp_generator = GetMorganGenerator(radius=2, fpSize=fp_size, includeChirality=False)
    for completed, (position, row) in enumerate(grouped.iterrows(), start=1):
        mol = Chem.MolFromSmiles(row["canonical_psmiles"])
        try:
            if mol is None:
                raise ValueError("RDKit failed to reparse canonical p-SMILES")
            morgan = _morgan_fp(mol, morgan_generator, fp_size)
            pecfp = _periodic_pecfp(mol, pecfp_generator, fp_size)
            topology = _linear_homopolymer_topology_fp(mol)
        except Exception as exc:  # record the exact failed molecule; do not misalign matrices
            excluded_records.append(
                {
                    "source_row": row["source_rows"],
                    "SMILES": row["source_smiles"],
                    "tg": float(row["tg"]),
                    "reason": "fingerprint_generation_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        retained_rows.append(position)
        morgan_vectors.append(morgan)
        pecfp_vectors.append(pecfp)
        topology_vectors.append(topology)
        if completed % 250 == 0 or completed == len(grouped):
            elapsed = time.time() - started
            print(f"fingerprints: {completed}/{len(grouped)} ({elapsed:.1f} s)", flush=True)

    cleaned = grouped.loc[retained_rows].reset_index(drop=True)
    morgan_matrix = np.stack(morgan_vectors).astype(np.uint8, copy=False)
    pecfp_matrix = np.stack(pecfp_vectors).astype(np.uint8, copy=False)
    topology_matrix = np.stack(topology_vectors).astype(np.uint8, copy=False)
    bigfp_matrix = np.concatenate([pecfp_matrix, topology_matrix], axis=1).astype(np.uint8, copy=False)

    split = _make_split(
        cleaned["tg"].to_numpy(dtype=float),
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        n_bins=n_bins,
        seed=seed,
    )
    split_name = np.empty(len(cleaned), dtype=object)
    for name, indices in split.items():
        split_name[indices] = name
    cleaned["split"] = split_name
    cleaned.insert(0, "sample_index", np.arange(len(cleaned)))

    cleaned.to_csv(output_dir / "cleaned_tg.csv", index=False)
    pd.DataFrame(excluded_records).to_csv(output_dir / "excluded_rows.csv", index=False)
    cleaned[
        ["sample_index", "source_rows", "canonical_psmiles", "tg", "split"]
    ].to_csv(output_dir / "split_assignments.csv", index=False)
    np.save(output_dir / FINGERPRINT_FILES["morgan"], morgan_matrix)
    np.save(output_dir / FINGERPRINT_FILES["pecfp"], pecfp_matrix)
    np.save(output_dir / "fingerprint_ptfp.npy", topology_matrix)
    np.save(output_dir / FINGERPRINT_FILES["bigfp"], bigfp_matrix)
    np.savez(output_dir / "split_indices.npz", **split)

    excluded_frame = pd.DataFrame(excluded_records)
    exclusion_counts = (
        excluded_frame["reason"].value_counts().to_dict() if not excluded_frame.empty else {}
    )
    summary = {
        "input_csv": str(input_csv.resolve()),
        "raw_rows": int(len(raw)),
        "eligible_linear_rows_before_canonical_aggregation": int(len(valid)),
        "canonical_samples": int(len(grouped)),
        "analyzed_samples": int(len(cleaned)),
        "excluded_rows": int(len(excluded_frame)),
        "exclusion_counts": exclusion_counts,
        "duplicate_measurements_aggregated": int(len(valid) - valid["canonical_psmiles"].nunique()),
        "tg": {
            "min": float(cleaned["tg"].min()),
            "max": float(cleaned["tg"].max()),
            "mean": float(cleaned["tg"].mean()),
            "median": float(cleaned["tg"].median()),
            "std": float(cleaned["tg"].std()),
        },
        "split": {name: int(len(indices)) for name, indices in split.items()},
        "split_fractions": {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
        "seed": seed,
        "stratification_quantile_bins": n_bins,
        "fingerprints": {
            "morgan": {"shape": list(morgan_matrix.shape), "dtype": str(morgan_matrix.dtype)},
            "pecfp": {"shape": list(pecfp_matrix.shape), "dtype": str(pecfp_matrix.dtype)},
            "ptfp": {
                "shape": list(topology_matrix.shape),
                "dtype": str(topology_matrix.dtype),
                "unique_vectors": int(len(np.unique(topology_matrix, axis=0))),
                "nonconstant_columns": np.flatnonzero(np.ptp(topology_matrix, axis=0)).tolist(),
            },
            "bigfp": {"shape": list(bigfp_matrix.shape), "dtype": str(bigfp_matrix.dtype)},
        },
        "elapsed_seconds": float(time.time() - started),
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2))
    return summary


def _estimator_and_grid(
    model_name: str,
    profile: str,
    seed: int,
    n_jobs: int,
    cache_dir: Path,
) -> tuple[Any, dict[str, list[Any]], int]:
    if model_name == "rf":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=n_jobs)
        grids = {
            "smoke": {
                "n_estimators": [50],
                "max_depth": [None, 15],
                "min_samples_split": [2],
                "min_samples_leaf": [1],
                "max_features": ["sqrt"],
            },
            "quick": {
                "n_estimators": [300],
                "max_depth": [None, 20, 50],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.5],
            },
            "full": {
                "n_estimators": [500, 1000],
                "max_depth": [None, 20, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3, 0.7],
            },
        }
        return estimator, grids[profile], 1

    if model_name == "svr":
        estimator = SVR(kernel="rbf")
        grids = {
            "smoke": {"C": [10.0, 100.0], "epsilon": [0.1], "gamma": ["scale"]},
            "quick": {
                "C": [10.0, 100.0, 1000.0],
                "epsilon": [0.1, 1.0, 5.0],
                "gamma": ["scale", 0.001],
            },
            "full": {
                "C": [1.0, 10.0, 100.0, 1000.0, 3000.0],
                "epsilon": [0.01, 0.1, 1.0, 5.0, 10.0],
                "gamma": ["scale", 0.0001, 0.001, 0.01],
            },
        }
        return estimator, grids[profile], n_jobs

    if model_name == "gp":
        # Exact sklearn GPR requires O(n^3) time and O(n^2) memory.  Nyström RBF
        # features plus ridge regression are the standard low-rank approximation
        # to the GP posterior mean and allow all ~5,700 train+validation samples.
        svd_components = 32 if profile == "smoke" else 64
        estimator = Pipeline(
            [
                ("svd", TruncatedSVD(n_components=svd_components, random_state=seed)),
                ("scale", StandardScaler()),
                ("kernel", Nystroem(kernel="rbf", random_state=seed, n_jobs=1)),
                ("ridge", Ridge()),
            ],
            memory=Memory(cache_dir, verbose=0),
        )
        grids = {
            "smoke": {
                "kernel__gamma": [0.01],
                "kernel__n_components": [64],
                "ridge__alpha": [1.0, 10.0],
            },
            "quick": {
                "kernel__gamma": [0.001, 0.01, 0.1],
                "kernel__n_components": [256],
                "ridge__alpha": [0.1, 1.0, 10.0],
            },
            "full": {
                "kernel__gamma": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                "kernel__n_components": [256, 512],
                "ridge__alpha": [0.01, 0.1, 1.0, 10.0],
            },
        }
        return estimator, grids[profile], n_jobs

    raise ValueError(f"Unknown model: {model_name}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def _prediction_rows(
    cleaned: pd.DataFrame,
    indices: np.ndarray,
    predicted: np.ndarray,
    fingerprint: str,
    model: str,
    stage: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, prediction in zip(indices, predicted):
        record = cleaned.iloc[int(index)]
        rows.append(
            {
                "fingerprint": fingerprint,
                "model": model,
                "stage": stage,
                "sample_index": int(index),
                "split": record["split"],
                "canonical_psmiles": record["canonical_psmiles"],
                "tg_actual": float(record["tg"]),
                "tg_predicted": float(prediction),
            }
        )
    return rows


def _plot_parity(predictions: pd.DataFrame, output_path: Path, stage: str) -> None:
    selected = predictions[predictions["stage"] == stage]
    if selected.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fingerprints = [name for name in FINGERPRINT_FILES if name in set(selected["fingerprint"])]
    models = [name for name in MODEL_LABELS if name in set(selected["model"])]
    fig, axes = plt.subplots(
        len(models), len(fingerprints), figsize=(4.2 * len(fingerprints), 3.9 * len(models)), squeeze=False
    )
    for row, model in enumerate(models):
        for column, fingerprint in enumerate(fingerprints):
            ax = axes[row, column]
            subset = selected[
                (selected["model"] == model) & (selected["fingerprint"] == fingerprint)
            ]
            if subset.empty:
                ax.axis("off")
                continue
            actual = subset["tg_actual"].to_numpy()
            predicted = subset["tg_predicted"].to_numpy()
            low = float(min(actual.min(), predicted.min()))
            high = float(max(actual.max(), predicted.max()))
            ax.scatter(actual, predicted, s=10, alpha=0.55, edgecolors="none")
            ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
            metrics = _metrics(actual, predicted)
            ax.text(
                0.04,
                0.96,
                f"R²={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.1f} °C",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
            if row == 0:
                ax.set_title(FINGERPRINT_LABELS[fingerprint])
            if column == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nPredicted Tg (°C)")
            else:
                ax.set_ylabel("Predicted Tg (°C)")
            ax.set_xlabel("Experimental Tg (°C)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def optimize_models(
    output_dir: Path,
    profile: str = "quick",
    fingerprints: list[str] | None = None,
    models: list[str] | None = None,
    seed: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    fingerprints = fingerprints or list(FINGERPRINT_FILES)
    models = models or list(MODEL_LABELS)
    cleaned = pd.read_csv(output_dir / "cleaned_tg.csv")
    split_file = np.load(output_dir / "split_indices.npz")
    split = {name: split_file[name] for name in ("train", "validation", "test")}
    y = cleaned["tg"].to_numpy(dtype=float)

    results_dir = output_dir / "results"
    search_dir = results_dir / "search"
    models_dir = results_dir / "models"
    cache_dir = results_dir / ".cache"
    for directory in (search_dir, models_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_indices = split["train"]
    validation_indices = split["validation"]
    test_indices = split["test"]
    train_validation_indices = np.concatenate([train_indices, validation_indices])
    predefined = PredefinedSplit(
        np.concatenate(
            [
                np.full(len(train_indices), -1, dtype=int),
                np.zeros(len(validation_indices), dtype=int),
            ]
        )
    )

    all_search: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] = {}
    run_started = time.time()

    for fingerprint in fingerprints:
        matrix_path = output_dir / FINGERPRINT_FILES[fingerprint]
        X = np.load(matrix_path, mmap_mode="r")
        X_train_validation = np.asarray(X[train_validation_indices], dtype=np.float32)
        y_train_validation = y[train_validation_indices]

        for model_name in models:
            combination = f"{fingerprint}__{model_name}"
            estimator, param_grid, search_jobs = _estimator_and_grid(
                model_name,
                profile,
                seed,
                n_jobs,
                cache_dir / combination,
            )
            candidate_count = len(list(ParameterGrid(param_grid)))
            print(
                f"search {combination}: {candidate_count} candidates; "
                f"train={len(train_indices)}, validation={len(validation_indices)}",
                flush=True,
            )
            search_started = time.time()
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring={
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                    "r2": "r2",
                },
                refit=False,
                cv=predefined,
                n_jobs=search_jobs,
                return_train_score=True,
                error_score="raise",
                verbose=0,
            )
            search.fit(X_train_validation, y_train_validation)
            cv = pd.DataFrame(search.cv_results_)
            best_index = int(cv["mean_test_rmse"].to_numpy().argmax())
            best_params = cv.loc[best_index, "params"]
            best_parameters[combination] = {
                "parameters": best_params,
                "selection_metric": "validation_rmse",
                "validation_rmse": float(-cv.loc[best_index, "mean_test_rmse"]),
                "validation_r2": float(cv.loc[best_index, "mean_test_r2"]),
                "candidate_count": candidate_count,
            }

            keep_columns = [
                "params",
                "mean_fit_time",
                "mean_score_time",
                "mean_train_rmse",
                "mean_test_rmse",
                "mean_train_mae",
                "mean_test_mae",
                "mean_train_r2",
                "mean_test_r2",
                "rank_test_rmse",
            ]
            search_frame = cv[keep_columns].copy()
            search_frame["params"] = search_frame["params"].map(
                lambda value: json.dumps(_json_safe(value), sort_keys=True)
            )
            for column in ("mean_train_rmse", "mean_test_rmse", "mean_train_mae", "mean_test_mae"):
                search_frame[column] = -search_frame[column]
            search_frame.insert(0, "model", model_name)
            search_frame.insert(0, "fingerprint", fingerprint)
            search_frame.to_csv(search_dir / f"{combination}.csv", index=False)
            all_search.append(search_frame)

            selection_model = clone(estimator).set_params(**best_params)
            selection_model.fit(np.asarray(X[train_indices], dtype=np.float32), y[train_indices])
            selection_predictions = {
                "train_selection": (
                    train_indices,
                    selection_model.predict(np.asarray(X[train_indices], dtype=np.float32)),
                ),
                "validation_selection": (
                    validation_indices,
                    selection_model.predict(np.asarray(X[validation_indices], dtype=np.float32)),
                ),
            }

            final_model = clone(estimator).set_params(**best_params)
            final_model.fit(X_train_validation, y_train_validation)
            final_predictions = {
                "train_validation_refit": (
                    train_validation_indices,
                    final_model.predict(X_train_validation),
                ),
                "test_final": (
                    test_indices,
                    final_model.predict(np.asarray(X[test_indices], dtype=np.float32)),
                ),
            }

            for stage, (indices, predicted) in {**selection_predictions, **final_predictions}.items():
                values = _metrics(y[indices], predicted)
                metric_rows.append(
                    {
                        "fingerprint": fingerprint,
                        "model": model_name,
                        "stage": stage,
                        "n_samples": int(len(indices)),
                        **values,
                    }
                )
                prediction_rows.extend(
                    _prediction_rows(cleaned, indices, predicted, fingerprint, model_name, stage)
                )

            if isinstance(final_model, Pipeline):
                final_model.memory = None
            joblib.dump(final_model, models_dir / f"{combination}.joblib", compress=3)
            print(
                f"finished {combination} in {time.time() - search_started:.1f} s; "
                f"validation RMSE={best_parameters[combination]['validation_rmse']:.3f}",
                flush=True,
            )

            pd.DataFrame(metric_rows).to_csv(results_dir / "metrics.csv", index=False)
            pd.DataFrame(prediction_rows).to_csv(results_dir / "predictions.csv", index=False)
            _write_json(results_dir / "best_parameters.json", best_parameters)

    search_summary = pd.concat(all_search, ignore_index=True)
    search_summary.to_csv(results_dir / "all_search_results.csv", index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    metrics_frame.to_csv(results_dir / "metrics.csv", index=False)
    predictions_frame.to_csv(results_dir / "predictions.csv", index=False)
    _plot_parity(predictions_frame, results_dir / "validation_parity.png", "validation_selection")
    _plot_parity(predictions_frame, results_dir / "test_parity.png", "test_final")

    run_summary = {
        "profile": profile,
        "fingerprints": fingerprints,
        "models": models,
        "seed": seed,
        "n_jobs": n_jobs,
        "selection_data": "train fit + validation selection only",
        "final_data": "train+validation refit followed by one test evaluation",
        "gp_implementation": "TruncatedSVD + RBF Nyström features + ridge (low-rank GP posterior-mean approximation)",
        "elapsed_seconds": float(time.time() - run_started),
        "best_parameters": best_parameters,
    }
    _write_json(results_dir / "run_summary.json", run_summary)
    print(json.dumps(_json_safe(run_summary), indent=2))
    return run_summary


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Stratified sample limit for smoke tests")


def _add_optimize_arguments(parser: argparse.ArgumentParser, include_output: bool = True) -> None:
    if include_output:
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("smoke", "quick", "full"), default="quick")
    parser.add_argument("--fingerprints", nargs="+", choices=tuple(FINGERPRINT_FILES), default=None)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_LABELS), default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="clean, split and generate fingerprints")
    _add_prepare_arguments(prepare)
    optimize = subparsers.add_parser("optimize", help="tune and evaluate all model/fingerprint combinations")
    _add_optimize_arguments(optimize)
    optimize.add_argument("--seed", type=int, default=42)
    all_parser = subparsers.add_parser("all", help="run prepare followed by optimize")
    _add_prepare_arguments(all_parser)
    _add_optimize_arguments(all_parser, include_output=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in ("prepare", "all"):
        prepare_dataset(
            input_csv=args.input,
            output_dir=args.output_dir,
            seed=args.seed,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            n_bins=args.n_bins,
            fp_size=args.fp_size,
            limit=args.limit,
        )
    if args.command in ("optimize", "all"):
        optimize_models(
            output_dir=args.output_dir,
            profile=args.profile,
            fingerprints=args.fingerprints,
            models=args.models,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()

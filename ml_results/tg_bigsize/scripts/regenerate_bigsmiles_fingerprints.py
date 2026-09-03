#!/usr/bin/env python3
"""Rebuild the large-Tg fingerprints with the repository's BigSMILES workflow.

Structures containing explicit deuterium are excluded before a new five-fold
split is generated with seed 42. Morgan is generated from canonical p-SMILES.
p-ECFP and p-TFP are generated from each corresponding BigSMILES string using
the standard repository implementations; BigFingerprint is their direct
concatenation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from tg_pipeline import (
    _bigfp_search_tools,
    _canonicalize_linear_psmiles,
    _psmiles_to_linear_bigsmiles,
    _standard_bigsmiles_fingerprints,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "artifacts_fivefold"
DEFAULT_OUTPUT = HERE / "artifacts_fivefold_bigsmiles_corrected_no2H"
FP_SIZE = 2048
P_TFP_SIZE = 40
SPLIT_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "sample_index",
        "canonical_psmiles",
        "bigsmiles",
        "tg",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    expected = np.arange(len(frame), dtype=int)
    observed = frame["sample_index"].to_numpy(dtype=int)
    if not np.array_equal(observed, expected):
        raise ValueError("sample_index is not identical to row order")
    if frame["canonical_psmiles"].astype(str).str.contains(r"\[2H\]", regex=True).any():
        raise ValueError("A [2H] record remained after the exclusion step")
    if set(frame["folder"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("The regenerated split does not contain folders 1 through 5")
    for row in frame.itertuples(index=False):
        psmiles = str(row.canonical_psmiles)
        bigsmiles = str(row.bigsmiles)
        if psmiles.count("*") != 2:
            raise ValueError(f"sample {row.sample_index}: p-SMILES does not contain two '*' tokens")
        if not (bigsmiles.startswith("{[]") and bigsmiles.endswith("[]}")):
            raise ValueError(f"sample {row.sample_index}: malformed linear-homopolymer BigSMILES")
        if bigsmiles.count("[$]") != 2:
            raise ValueError(f"sample {row.sample_index}: BigSMILES does not contain two '[$]' descriptors")
        if "[[$]]" in bigsmiles or "*" in bigsmiles:
            raise ValueError(f"sample {row.sample_index}: unconverted or nested wildcard descriptor")


def allocate_partial(output_dir: Path, name: str, shape: tuple[int, int]) -> np.memmap:
    return open_memmap(output_dir / f".{name}.partial.npy", mode="w+", dtype=np.uint8, shape=shape)


def finalize_partial(output_dir: Path, name: str) -> Path:
    partial = output_dir / f".{name}.partial.npy"
    final = output_dir / name
    os.replace(partial, final)
    return final


def rebuild(source_dir: Path, output_dir: Path) -> None:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from the legacy source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_csv = source_dir / "cleaned_tg.csv"
    source_frame = pd.read_csv(source_csv)
    deuterium_mask = source_frame["canonical_psmiles"].astype(str).str.contains(
        r"\[2H\]", regex=True
    )
    deuterium_excluded = source_frame.loc[deuterium_mask].copy()
    frame = source_frame.loc[~deuterium_mask].copy().reset_index(drop=True)
    normalized = frame["canonical_psmiles"].map(_canonicalize_linear_psmiles)
    failures = normalized.map(lambda result: result[1] is not None)
    if failures.any():
        failed = frame.loc[failures, "canonical_psmiles"].tolist()
        raise ValueError(f"Canonical p-SMILES normalization failed: {failed[:5]}")
    frame["canonical_psmiles"] = normalized.map(lambda result: result[0])
    if frame["canonical_psmiles"].duplicated().any():
        duplicated = frame.loc[
            frame["canonical_psmiles"].duplicated(keep=False), "canonical_psmiles"
        ].tolist()
        raise ValueError(
            "Wildcard normalization created duplicate structures; reaggregation is required: "
            f"{duplicated[:5]}"
        )
    frame.insert(1, "legacy_sample_index", frame["sample_index"].to_numpy(dtype=int))
    frame["sample_index"] = np.arange(len(frame), dtype=int)
    frame["bigsmiles"] = frame["canonical_psmiles"].map(_psmiles_to_linear_bigsmiles)

    # Recreate the five folders only after the exclusion rule has been applied.
    permutation = np.random.RandomState(SPLIT_SEED).permutation(len(frame))
    folds = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(permutation, 5)]
    folder = np.empty(len(frame), dtype=np.int8)
    role = np.empty(len(frame), dtype=object)
    for fold_number, indices in enumerate(folds, start=1):
        folder[indices] = fold_number
        role[indices] = "heldout_test" if fold_number == 1 else "cv_pool"
    frame["folder"] = folder
    frame["split"] = role
    validate_input(frame)
    n_samples = len(frame)

    # Fail fast on conversion/parser errors before allocating or filling any
    # fingerprint matrix.
    preflight_started = time.time()
    with open(os.devnull, "w") as devnull:
        for checked, row in enumerate(frame.itertuples(index=False), start=1):
            try:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    _bigfp_search_tools.generate_NetworkX_graphs(
                        str(row.bigsmiles), is_bigsmarts=False
                    )
            except Exception as exc:
                raise ValueError(
                    f"BigSMILES parser preflight failed for sample {row.sample_index}: "
                    f"p-SMILES={row.canonical_psmiles}; BigSMILES={row.bigsmiles}"
                ) from exc
            if checked % 1000 == 0 or checked == n_samples:
                print(f"BigSMILES preflight {checked}/{n_samples}", flush=True)

    morgan = allocate_partial(output_dir, "fingerprint_morgan.npy", (n_samples, FP_SIZE))
    pecfp = allocate_partial(output_dir, "fingerprint_pecfp.npy", (n_samples, FP_SIZE))
    p_tfp = allocate_partial(output_dir, "fingerprint_ptfp.npy", (n_samples, P_TFP_SIZE))
    bigfp = allocate_partial(
        output_dir, "fingerprint_bigfp.npy", (n_samples, FP_SIZE + P_TFP_SIZE)
    )
    morgan_generator = GetMorganGenerator(radius=2, fpSize=FP_SIZE, includeChirality=False)
    started = time.time()
    with open(os.devnull, "w") as devnull:
        for completed, row in enumerate(frame.itertuples(index=False), start=1):
            mol = Chem.MolFromSmiles(str(row.canonical_psmiles))
            if mol is None:
                raise ValueError(f"sample {row.sample_index}: canonical p-SMILES parse failed")
            bit_vector = morgan_generator.GetFingerprint(mol)
            morgan_row = np.zeros(FP_SIZE, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(bit_vector, morgan_row)
            with contextlib.redirect_stdout(devnull):
                pecfp_row, p_tfp_row = _standard_bigsmiles_fingerprints(
                    str(row.bigsmiles), FP_SIZE
                )
            if p_tfp_row[0] != 1:
                raise ValueError(
                    f"sample {row.sample_index}: standard p-TFP did not identify the BigSMILES as a polymer"
                )
            morgan[row.sample_index] = morgan_row
            pecfp[row.sample_index] = pecfp_row
            p_tfp[row.sample_index] = p_tfp_row
            bigfp[row.sample_index, :FP_SIZE] = pecfp_row
            bigfp[row.sample_index, FP_SIZE:] = p_tfp_row
            if completed % 50 == 0 or completed == n_samples:
                for matrix in (morgan, pecfp, p_tfp, bigfp):
                    matrix.flush()
                print(
                    f"fingerprints {completed}/{n_samples}; elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    del morgan, pecfp, p_tfp, bigfp
    matrix_paths = {
        name: finalize_partial(output_dir, name)
        for name in (
            "fingerprint_morgan.npy",
            "fingerprint_pecfp.npy",
            "fingerprint_ptfp.npy",
            "fingerprint_bigfp.npy",
        )
    }
    frame.to_csv(output_dir / "cleaned_tg.csv", index=False)
    frame[
        [
            "sample_index",
            "legacy_sample_index",
            "source_rows",
            "canonical_psmiles",
            "bigsmiles",
            "tg",
            "folder",
            "split",
        ]
    ].to_csv(output_dir / "split_assignments.csv", index=False)
    split_payload = {f"fold{i}": fold for i, fold in enumerate(folds, start=1)}
    split_payload["train_pool"] = np.concatenate(folds[1:])
    split_payload["test"] = folds[0]
    np.savez(output_dir / "split_indices.npz", **split_payload)
    for fold_number, indices in enumerate(folds, start=1):
        np.save(output_dir / f"tg_{n_samples}_indices_folder{fold_number}.npy", indices)

    previous_excluded_path = source_dir / "excluded_rows.csv"
    if previous_excluded_path.exists():
        excluded_frame = pd.read_csv(previous_excluded_path)
    else:
        excluded_frame = pd.DataFrame()
    deuterium_records = pd.DataFrame(
        {
            "source_row": deuterium_excluded["source_rows"],
            "SMILES": deuterium_excluded["canonical_psmiles"],
            "tg": deuterium_excluded["tg"],
            "reason": "explicit_deuterium_[2H]",
            "detail": "excluded before splitting and fingerprint generation",
        }
    )
    pd.concat([excluded_frame, deuterium_records], ignore_index=True).to_csv(
        output_dir / "excluded_rows.csv", index=False
    )
    split_summary = {
        "protocol": "exclude [2H], then ThermalProcess five-folder held-out-test protocol",
        "seed": SPLIT_SEED,
        "source_sample_count": int(len(source_frame)),
        "excluded_explicit_deuterium": int(len(deuterium_excluded)),
        "sample_count": n_samples,
        "folder_sizes": {
            f"fold{i}": int(len(fold)) for i, fold in enumerate(folds, start=1)
        },
        "heldout_test_folder": "fold1",
        "cv_pool_folders": ["fold2", "fold3", "fold4", "fold5"],
        "test_is_excluded_from_all_hyperparameter_search": True,
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    stored_pecfp = np.load(matrix_paths["fingerprint_pecfp.npy"], mmap_mode="r")
    stored_p_tfp = np.load(matrix_paths["fingerprint_ptfp.npy"], mmap_mode="r")
    stored_bigfp = np.load(matrix_paths["fingerprint_bigfp.npy"], mmap_mode="r")
    if not np.array_equal(stored_bigfp[:, :FP_SIZE], stored_pecfp):
        raise AssertionError("BigFingerprint prefix does not equal p-ECFP")
    if not np.array_equal(stored_bigfp[:, FP_SIZE:], stored_p_tfp):
        raise AssertionError("BigFingerprint suffix does not equal p-TFP")
    if not np.all(stored_p_tfp[:, 0] == 1):
        raise AssertionError("Not every BigSMILES has the polymer bit set")

    metadata = {
        "status": "corrected",
        "source_artifacts": str(source_dir.resolve()),
        "source_cleaned_tg_sha256": sha256(source_csv),
        "source_sample_count": int(len(source_frame)),
        "excluded_explicit_deuterium": int(len(deuterium_excluded)),
        "excluded_legacy_sample_indices": deuterium_excluded["sample_index"]
        .astype(int)
        .tolist(),
        "sample_count": n_samples,
        "split_seed": SPLIT_SEED,
        "split_created_after_exclusion": True,
        "morgan_input": "canonical p-SMILES",
        "pecfp_input": "BigSMILES",
        "ptfp_input": "BigSMILES",
        "bigfp_definition": "concatenate(p-ECFP[2048], p-TFP[40])",
        "pecfp_implementation": "polymersearch-2025/polymer_ecfp.py::ECFP",
        "ptfp_implementation": "polymersearch-2025/polymer_topfp.py::bigsmiles_topology_vector",
        "ptfp_unique_vectors": int(len(np.unique(stored_p_tfp, axis=0))),
        "ptfp_nonconstant_dimensions_zero_based": np.flatnonzero(
            np.ptp(stored_p_tfp, axis=0)
        ).tolist(),
        "matrix_sha256": {name: sha256(path) for name, path in matrix_paths.items()},
        "elapsed_seconds": time.time() - started,
        "bigsmiles_preflight_seconds": started - preflight_started,
    }
    with (output_dir / "fingerprint_generation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rebuild(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()

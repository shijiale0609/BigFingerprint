#!/usr/bin/env python3
"""Prepare the corrected 2,700-row EA/IP data bundle for SCC GPR jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ea_ip_svr_rf import (
    FINGERPRINT_FILES,
    TARGET_COLUMNS,
    build_sampled_global_indices,
    make_test_splits,
)

OUTPUT_FINGERPRINT_FILES = {
    "morgan": "fingerprint_morgan.npy",
    "pecfp": "fingerprint_pecfp.npy",
    "bigfp": "fingerprint_bigfp.npy",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(raw_data_dir, output_dir):
    """Subset full EA/IP arrays and save fold-aligned SCC inputs.

    Parameters
    ----------
    raw_data_dir : pathlib.Path
        Directory containing ``dataset.csv`` and the three full arrays.
    output_dir : pathlib.Path
        Destination for the 2,700-row arrays, targets, folds, and manifest.

    Returns
    -------
    dict
        Manifest containing dimensions, seeds, and SHA256 checksums.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    global_indices, sampled_monomer_b = build_sampled_global_indices()
    folds = make_test_splits(len(global_indices))
    fold_assignment = np.empty(len(global_indices), dtype=np.int64)
    for fold_number, positions in enumerate(folds, start=1):
        fold_assignment[positions] = fold_number

    full_targets = pd.read_csv(raw_data_dir / "dataset.csv")
    targets = pd.DataFrame(
        {
            "subset_position": np.arange(len(global_indices)),
            "global_index": global_indices,
            "test_fold": fold_assignment,
            "ea_eV": full_targets[TARGET_COLUMNS["ea"]].to_numpy(dtype=float)[
                global_indices
            ],
            "ip_eV": full_targets[TARGET_COLUMNS["ip"]].to_numpy(dtype=float)[
                global_indices
            ],
        }
    )
    targets.to_csv(output_dir / "targets.csv", index=False)
    np.savez_compressed(
        output_dir / "split_indices.npz",
        **{f"fold{index}": values for index, values in enumerate(folds, start=1)},
    )

    shapes = {}
    dtypes = {}
    for fingerprint, input_filename in FINGERPRINT_FILES.items():
        full_matrix = np.load(raw_data_dir / input_filename, mmap_mode="r")
        selected = np.asarray(full_matrix[global_indices])
        output_filename = OUTPUT_FINGERPRINT_FILES[fingerprint]
        np.save(output_dir / output_filename, selected)
        shapes[output_filename] = list(selected.shape)
        dtypes[output_filename] = str(selected.dtype)

    output_names = [
        "targets.csv",
        "split_indices.npz",
        *OUTPUT_FINGERPRINT_FILES.values(),
    ]
    manifest = {
        "sample_count": 2700,
        "sample_random_seed": 0,
        "sampled_monomer_b": sampled_monomer_b,
        "kfold_random_state": 42,
        "fold_sizes": [len(fold) for fold in folds],
        "fingerprint_shapes": shapes,
        "fingerprint_dtypes": dtypes,
        "files_sha256": {name: sha256(output_dir / name) for name in output_names},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_data(args.raw_data_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

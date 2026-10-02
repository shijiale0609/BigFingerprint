#!/usr/bin/env python3
"""Re-run one historical p-TFP timeout case with the unmodified workflow."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from bigfingerprint.polymer_topfp import bigsmiles_topology_vector

HERE = Path(__file__).resolve().parent
AUDIT_CSV = HERE / "results_489" / "ptfp_recovered_timeout_cases.csv"
OUTPUT_DIR = HERE / "results_489" / "ptfp_standard_rerun"


class RerunTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame):
    raise RerunTimeoutError("standard p-TFP rerun exceeded the requested time limit")


def load_case(polymer_id: str) -> dict[str, str]:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["polymer_id"] == polymer_id:
                return row
    raise ValueError(f"Unknown recovered timeout case: {polymer_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("polymer_id")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    row = load_case(args.polymer_id)
    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout_seconds)
    status = "error"
    vector = None
    error = ""

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with (
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                calculated = bigsmiles_topology_vector(row["BigSMILES"])
        calculated = np.asarray(calculated, dtype=int)
        if calculated.shape != (40,):
            raise ValueError(f"Expected 40 p-TFP dimensions, got {calculated.shape}")
        vector = calculated.tolist()
        status = "completed"
    except RerunTimeoutError as exc:
        status = "timeout"
        error = repr(exc)
    except Exception as exc:
        error = repr(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    elapsed = time.monotonic() - started
    result = {
        "polymer_id": row["polymer_id"],
        "original_target_index": int(row["original_target_index"]),
        "status": status,
        "timeout_seconds": args.timeout_seconds,
        "elapsed_seconds": elapsed,
        "pTFP": vector,
        "pTFP_bitstring": "" if vector is None else "".join(map(str, vector)),
        "error": error,
        "method": "unmodified polymer_topfp.bigsmiles_topology_vector",
        "BigSMILES": row["BigSMILES"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.polymer_id}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

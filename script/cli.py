"""Command-line interface for BigFingerprint."""

import argparse
import json
from pathlib import Path

from . import __version__
from .api import generate_bigfingerprint_vectors
from .bigfingerprint_preprocess import preprocess_bigsmiles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bigfingerprint",
        description="Generate polymer fingerprints from BigSMILES.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="expand explicit or missing [H] end groups"
    )
    preprocess_parser.add_argument("bigsmiles")

    fingerprint_parser = subparsers.add_parser(
        "fingerprint", help="generate BigFingerprint vectors"
    )
    fingerprint_parser.add_argument("bigsmiles")
    fingerprint_parser.add_argument("--radius", type=int, default=2, dest="pecfp_iterations")
    fingerprint_parser.add_argument("--fp-size", type=int, default=2048)
    fingerprint_parser.add_argument(
        "--no-preprocess-h",
        action="store_true",
        help="fingerprint the input verbatim without [H] preprocessing",
    )
    fingerprint_parser.add_argument("--verbose", action="store_true")
    fingerprint_parser.add_argument(
        "-o", "--output", type=Path, help="write JSON to this file instead of stdout"
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "preprocess":
        for candidate in preprocess_bigsmiles(args.bigsmiles):
            print(candidate)
        return 0

    result = generate_bigfingerprint_vectors(
        args.bigsmiles,
        preprocess_h=not args.no_preprocess_h,
        pecfp_iterations=args.pecfp_iterations,
        fp_size=args.fp_size,
        verbose=args.verbose,
    )
    payload = {
        "input_bigsmiles": result.input_bigsmiles,
        "fingerprints": [
            {
                "bigsmiles": candidate,
                "size": int(vector.size),
                "on_bits": [int(index) for index in vector.nonzero()[0]],
                "vector": vector.astype(int).tolist(),
            }
            for candidate, vector in result
        ],
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

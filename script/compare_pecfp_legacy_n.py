"""Compare the legacy and N-iteration pECFP implementations at a fixed radius."""

import argparse

import numpy as np

import polymer_ecfp
import search_tools


DEFAULT_BIGSMILES = "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}"


def calculate(ecfp, bigsmiles, iterations, fp_size):
    # Each implementation gets a fresh graph because the legacy code removes
    # explicit hydrogen nodes in-place.
    graphs = search_tools.generate_NetworkX_graphs(bigsmiles, is_bigsmarts=False)
    _, clusters, _ = search_tools.identify_cycles(graphs)
    return ecfp(
        graphs=graphs,
        individual_clusters=clusters,
        iteration_input=iterations,
        fp_size=fp_size,
        print_all=False,
        print_identifiers=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bigsmiles", default=DEFAULT_BIGSMILES)
    # The legacy implementation only materializes identifiers through round 2.
    parser.add_argument("--iterations", type=int, default=2, choices=[2])
    parser.add_argument("--fp-size", type=int, default=2048)
    args = parser.parse_args()

    legacy = calculate(polymer_ecfp._legacy_ECFP_two_iterations,
                       args.bigsmiles, args.iterations, args.fp_size)
    general = calculate(polymer_ecfp.ECFP, args.bigsmiles, args.iterations, args.fp_size)
    differing_bits = int(np.count_nonzero(legacy != general))
    print({
        "iterations": args.iterations,
        "legacy_on_bits": int(legacy.sum()),
        "n_iteration_on_bits": int(general.sum()),
        "identical": bool(np.array_equal(legacy, general)),
        "differing_bits": differing_bits,
    })


if __name__ == "__main__":
    main()

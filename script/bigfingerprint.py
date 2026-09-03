import numpy as np

from . import polymer_ecfp, polymer_topfp, search_tools
from .bigfingerprint_preprocess import preprocess_bigsmiles


def generate_bigfingerprint_vector(
    bigsmiles,
    pecfp_iterations=2,
    *,
    fp_size=2048,
    print_identifiers=False,
    save_graph=False,
    graph_name="",
    verbose=False,
):
    """Generate one unified BigFingerprint vector from a BigSMILES string.

    Missing or explicit ``[H]`` end groups are always preprocessed.  p-ECFP is
    calculated for every resulting candidate and combined by bitwise OR.  The
    p-TFP vector is calculated from the original BigSMILES because end-group
    expansion does not change the polymer topology.  The return value is
    therefore always one binary NumPy vector with shape ``(fp_size + 40,)``.
    """
    if not isinstance(bigsmiles, str) or not bigsmiles.strip():
        raise ValueError("bigsmiles must be a non-empty string")
    if not isinstance(pecfp_iterations, int) or pecfp_iterations < 0:
        raise ValueError("pecfp_iterations must be a non-negative integer")
    if not isinstance(fp_size, int) or fp_size <= 0:
        raise ValueError("fp_size must be a positive integer")

    candidates = preprocess_bigsmiles(bigsmiles)
    candidate_pecfp_vectors = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_graph_name = graph_name
        if save_graph and len(candidates) > 1:
            candidate_graph_name = f"{graph_name}_candidate_{candidate_index}"
        candidate_pecfp_vectors.append(
            _generate_pecfp_vector(
                candidate,
                pecfp_iterations=pecfp_iterations,
                fp_size=fp_size,
                print_identifiers=print_identifiers,
                save_graph=save_graph,
                graph_name=candidate_graph_name,
                verbose=verbose,
            )
        )

    pecfp_vector = np.bitwise_or.reduce(
        np.asarray(candidate_pecfp_vectors, dtype=np.uint8),
        axis=0,
    )
    ptopfp_vector = np.asarray(
        polymer_topfp.bigsmiles_topology_vector(bigsmiles),
        dtype=np.uint8,
    )
    return np.concatenate((pecfp_vector, ptopfp_vector), axis=0)


def _generate_pecfp_vector(
    bigsmiles,
    *,
    pecfp_iterations,
    fp_size,
    print_identifiers,
    save_graph,
    graph_name,
    verbose,
):
    """Generate p-ECFP for one fully resolved BigSMILES candidate."""
    graphs = search_tools.generate_NetworkX_graphs(bigsmiles, is_bigsmarts=False)
    _, individual_clusters, _ = search_tools.identify_cycles(graphs)
    return polymer_ecfp.ECFP(
        graphs=graphs,
        individual_clusters=individual_clusters,
        iteration_input=pecfp_iterations,
        repeat_unit_property=False,
        backbone_property=False,
        fp_size=fp_size,
        print_identifiers=print_identifiers,
        print_all=verbose,
        graph_name=graph_name,
        save_graph=save_graph,
    )


if __name__ == "__main__":
    bigsmiles_example = "{[][<]C(=O)CCC(=O)[<],[>]OCC(CO[>])(CO[>])CO[>][]}"
    bigfingerprint_example = generate_bigfingerprint_vector(
        bigsmiles_example,
        pecfp_iterations=2,
        fp_size=2048,
    )
    print("BigSMILES:", bigsmiles_example)
    print("BigFingerprint:", bigfingerprint_example)
    print("BigFingerprint Shape:", bigfingerprint_example.shape)
    print("\n\n")

    bigsmiles_example = "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}"
    bigfingerprint_example = generate_bigfingerprint_vector(
        bigsmiles_example,
        pecfp_iterations=2,
        fp_size=2048,
    )
    print("BigSMILES:", bigsmiles_example)
    print("BigFingerprint:", bigfingerprint_example)
    print("BigFingerprint Shape:", bigfingerprint_example.shape)
    print("\n\n")

import numpy as np

from . import polymer_ecfp, polymer_topfp, search_tools


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
    """Generate one BigFingerprint vector from one resolved BigSMILES string.

    This low-level function fingerprints the input verbatim.  Use
    :func:`bigfingerprint.generate_bigfingerprint_vectors` when ``[H]`` end
    groups should be completed and expanded first.

    The vector is the concatenation of a configurable-length pECFP vector and
    the 40-element p-TFP vector.  Legacy diagnostic output and files are off by
    default so normal library calls are side-effect-free.
    """
    if not isinstance(bigsmiles, str) or not bigsmiles.strip():
        raise ValueError("bigsmiles must be a non-empty string")
    if not isinstance(pecfp_iterations, int) or pecfp_iterations < 0:
        raise ValueError("pecfp_iterations must be a non-negative integer")
    if not isinstance(fp_size, int) or fp_size <= 0:
        raise ValueError("fp_size must be a positive integer")

    def calculate():
        graphs = search_tools.generate_NetworkX_graphs(
            bigsmiles, is_bigsmarts=False
        )
        _, individual_clusters, _ = search_tools.identify_cycles(graphs)
        pecfp_vector = polymer_ecfp.ECFP(
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
        ptopfp_vector = polymer_topfp.bigsmiles_topology_vector(bigsmiles)
        return np.concatenate((pecfp_vector, ptopfp_vector), axis=0)

    return calculate()
 


if __name__ == "__main__":
    bigsmiles_example = "{[][<]C(=O)CCC(=O)[<],[>]OCC(CO[>])(CO[>])CO[>][]}"
    bigfingerprint_example = generate_bigfingerprint_vector(bigsmiles_example)
    print("BigSMILES:", bigsmiles_example)
    print("BigFingerprint:", bigfingerprint_example)
    print("BigFingerprint Shape:", bigfingerprint_example.shape)
    print("\n\n")

    bigsmiles_example = "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}"
    bigfingerprint_example = generate_bigfingerprint_vector(bigsmiles_example)
    print("BigSMILES:", bigsmiles_example)
    print("BigFingerprint:", bigfingerprint_example)
    print("BigFingerprint Shape:", bigfingerprint_example.shape)
    print("\n\n")

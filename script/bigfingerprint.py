import numpy as np
import matplotlib.pyplot as plt
import json

import search_tools 
import polymer_ecfp
import polymer_topfp


def generate_bigfingerprint_vector(bigsmiles):

    graphs = search_tools.generate_NetworkX_graphs(bigsmiles, is_bigsmarts = False)
    group_cycles, individual_clusters, treat_as_cycle = search_tools.identify_cycles(graphs)
    pecfp_vector = polymer_ecfp.ECFP(graphs = graphs, individual_clusters = individual_clusters, iteration_input = 2, repeat_unit_property = False, backbone_property = False, fp_size = 2048, print_identifiers = True)

    ptopfp_vector = polymer_topfp.bigsmiles_topology_vector(bigsmiles)

    bigfingerprint_vector = np.concatenate((pecfp_vector, ptopfp_vector),axis=0)
    
    return bigfingerprint_vector
 


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

import numpy as np

import search_tools 
import polymer_ecfp


bigsmarts_query_list = [
"{[][]}",
"{[][]}!{[][]}",
"{[][]}?*{[][]}",
"{[][]}?*{[][]}!{[][]}",
"{[][]}?*{[][]}?*{[][]}",
"{[][<]C(=O)O?*[>][]}",
"{[][<]OC(=O)O?*[>][]}",
"{[][<]NC(=O)O?*[>][]}",
"{[][<]C=C?*[>][]}",
"{[][<]NC(=O)N?*[>][]}",
]

#bigsmiles = "{[][<]COC[>],[<]COC[>][]}"
bigsmiles = "{[][$]CC[$][]}"
#bigsmiles = "C([#Arm])([#Arm])([#Arm])[#Arm].{#Arm=CO{[<][>]CCO[<][]}}"
#bigsmiles = "{[][<]CC(C)O[>][]}"




def polymer_bigsmarts_search_one_hot_encoding(given_bigsmiles, given_bigsmarts_query_list):

    one_hot_encoding = []

    for bigsmarts in given_bigsmarts_query_list:
        return_result = search_tools.graph_traversal_match(bigsmarts, bigsmiles)
        if return_result == True:
            one_hot_encoding.append(1)
        else:
            one_hot_encoding.append(0)
    return np.array(one_hot_encoding)


graphs = search_tools.generate_NetworkX_graphs(bigsmiles, is_bigsmarts = False)
group_cycles, individual_clusters, treat_as_cycle = search_tools.identify_cycles(graphs)
v1 = polymer_ecfp.ECFP(graphs = graphs, individual_clusters = individual_clusters)

v2 = polymer_bigsmarts_search_one_hot_encoding(given_bigsmiles=bigsmiles, given_bigsmarts_query_list=bigsmarts_query_list)

print(type(v1), type(v2))
bigfingerprint = np.concatenate((v1, v2),axis=0)

print(bigfingerprint.shape)

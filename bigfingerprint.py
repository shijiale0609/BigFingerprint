import numpy as np
import matplotlib.pyplot as plt

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

bigsmiles_list = [
"{[][<]COC[>],[<]COC[>][]}",
"{[][$]CC[$][]}",
"C([#Arm])([#Arm])([#Arm])[#Arm].{#Arm=CO{[<][>]CCO[<][]}}",
"{[][<]CC(C)O[>][]}",
]



def polymer_bigsmarts_search_one_hot_encoding(given_bigsmiles, given_bigsmarts_query_list):

    one_hot_encoding = []

    for bigsmarts in given_bigsmarts_query_list:
        return_result = search_tools.graph_traversal_match(bigsmarts, bigsmiles)
        if return_result == True:
            one_hot_encoding.append(1)
        else:
            one_hot_encoding.append(0)
    return np.array(one_hot_encoding)

bigfingerprint_matrix = []
for bigsmiles in bigsmiles_list:
    graphs = search_tools.generate_NetworkX_graphs(bigsmiles, is_bigsmarts = False)
    group_cycles, individual_clusters, treat_as_cycle = search_tools.identify_cycles(graphs)
    v1 = polymer_ecfp.ECFP(graphs = graphs, individual_clusters = individual_clusters)

    v2 = polymer_bigsmarts_search_one_hot_encoding(given_bigsmiles=bigsmiles, given_bigsmarts_query_list=bigsmarts_query_list)

    print(type(v1), type(v2))
    bigfingerprint = np.concatenate((v1, v2),axis=0)
    bigfingerprint_matrix.append(bigfingerprint)
    
#print(bigfingerprint.shape)
bigfingerprint_matrix = np.array(bigfingerprint_matrix)
print(bigfingerprint_matrix.shape)
for i in range(0, 4):
    for j in range(0, 1024):
        if bigfingerprint_matrix[i,j]>0.0001 and bigfingerprint_matrix[i,j]<1:
            print(i,j, bigfingerprint_matrix[i,j])


plt.figure(figsize=(500, 1))
#plt.imshow(bigfingerprint_matrix, cmap='viridis', aspect='auto')  # You can choose other colormaps like 'hot', 'coolwarm', 'jet', etc.
#plt.colorbar(label='Value')
plt.imshow(bigfingerprint_matrix, cmap='gray', aspect='auto')
plt.title('BigFinerprint Matrix')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()




from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np
import itertools
import networkx as nx    
import copy
import xlwt
import pandas as pd
import sys, os
import matplotlib.pyplot as plt

import search_tools

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def in_ring(atomistic, atom):
    symbols = nx.get_node_attributes(atomistic, "symbol")
    for neighbor in atomistic[atom]:
        count = 0
        for path in nx.all_simple_paths(atomistic, atom, neighbor):
            found = False
            for p in path:
                if ">" in symbols[p] or "<" in symbols[p] or "$" in symbols[p]:
                    found = True
                    break
            if not found:
                count += 1
            if count == 2:
                return True
    return False

def in_repeat(graphs, individual_clusters, atom):
    ids = graphs["ids"]
    # get the topological id of the atom
    id = ids[atom]
    # determine if the topological id is in a stochastic object
    for stochastic_object in individual_clusters:
        if id in stochastic_object:
            return True
    return False

def get_all_backbone_atoms(graphs, individual_clusters):
    atomistic = graphs["atomistic"]
    symbols = nx.get_node_attributes(atomistic, "symbol")
    backbone = set()
    # get all atoms adjacent to descriptors in the stochastic objects
    for node in symbols:
        if is_descriptor(symbols[node]):
            neighbors = atomistic[node]
            for n in neighbors:
                if in_repeat(graphs, individual_clusters, n) and n not in backbone:
                    backbone.add(n) 

    # between every unique pair of atoms, get all paths that do not contain descriptors and add to set
    backbone = list(backbone)
    all_backbone = set()
    for i in range(len(backbone)):
        for j in range(i + 1, len(backbone)):
            for path in nx.all_simple_paths(atomistic, backbone[i], backbone[j]):
                found = False
                for p in path:
                    if is_descriptor(symbols[p]):
                        found = True
                        break
                if not found:
                    # path does not contain descriptors
                    for p in path:
                        all_backbone.add(p)

    return all_backbone

def is_descriptor(symbol):
    return ">" in symbol or "<" in symbol or "$" in symbol

def bonds_compat_desc(list_bonds, bonds):
    path = []
    for i in range(len(list_bonds) - 1):
        bond_id = tuple(sorted([list_bonds[i], list_bonds[i + 1]]))
        path.append(bonds[bond_id][:2])

    possible = [
            ["1_", "2_"], 
            ["2_", "1_"], 
            ["1_", "2_", "1_", "2_"], 
            ["1_", "2_", "2_", "1_"], 
            ["2_", "1_", "1_", "2_"],
            ["2_", "1_", "2_", "1_"],
            ["1a", "1b"],
            ["1b", "1a"],  
            ["2a", "2b"], 
            ["2b", "2a"]
        ]
    
    return path in possible

def ECFP(graphs, individual_clusters, iteration_input = 2, fp_size = 1024, print_identifiers = False, print_all = True, graph_name = ""):
    # https://chemicbook.com/2021/03/25/a-beginners-guide-for-understanding-extended-connectivity-fingerprints.html
    # remove [H] from graph

    if not print_all:
        blockPrint()

    if print_identifiers:
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Sheet 1")
        sheet1.write(0, 0, "Node")
        sheet1.write(0, 1, "Neighbors")
        sheet1.write(0, 2, "Valence - Hydrogens")
        sheet1.write(0, 3, "Atomic #")
        sheet1.write(0, 4, "Atom Mass")
        sheet1.write(0, 5, "Atom Charge")
        sheet1.write(0, 6, "Hydrogens")
        sheet1.write(0, 7, "Ring")
        sheet1.write(0, 8, "Repeat")
        sheet1.write(0, 9, "Backbone")
        row = 1

    atomistic = graphs["atomistic"]
    print(type(atomistic), nx.is_directed(atomistic),isinstance(atomistic, nx.MultiGraph))



    plt.figure(figsize=(8, 6))
    nx.draw(atomistic, with_labels=True, node_color='lightblue', node_size=800, font_size=14, font_weight='bold')

    # Save the visualization to a file
    graph_file_name = "network_graph"+ graph_name +".png" 
    plt.savefig(graph_file_name, format="PNG")    


    symbols = nx.get_node_attributes(atomistic, "symbol")
    for key in symbols:
        if symbols[key] == "H":
            atomistic.remove_node(key)
    symbols = nx.get_node_attributes(atomistic, "symbol")
    print("########################")
    print(symbols)
    print("########################")
    formal_charge = nx.get_node_attributes(atomistic, "formal_charge")
    valence = nx.get_node_attributes(atomistic, "valence")
    atomic_num = nx.get_node_attributes(atomistic, "atomic_num")
    mass = nx.get_node_attributes(atomistic, "mass")
    num_hs = nx.get_node_attributes(atomistic, "num_hs") 
    bonds = nx.get_edge_attributes(atomistic, "bond_type")

    backbone_atoms = get_all_backbone_atoms(graphs, individual_clusters)

    all_identifiers = dict()
    feature_list = [] 
    print("node, atomic invariants, identifier after hashing")
    for node in symbols:
        if not is_descriptor(symbols[node]) and symbols[node] != "":
            neighbors = len(atomistic[node]) 
            valence_atoms = valence[node]
            hydrogens_atom = num_hs[node]
            atomic_atom = atomic_num[node]
            mass_atom = mass[node]
            charge_atom = formal_charge[node] 
            ring = int(in_ring(atomistic, node))
            repeat = int(in_repeat(graphs, individual_clusters, node))
            backbone = int(node in backbone_atoms)
            
            invar = (neighbors, valence_atoms - hydrogens_atom, atomic_atom, mass_atom, charge_atom, hydrogens_atom, ring, repeat, backbone)
            identifier = hash(invar)
            if print_identifiers:
                col = 0
                sheet1.write(row, col, node)
                col += 1
                for i in invar:
                    sheet1.write(row, col, i)
                    col += 1
                row += 1
            print(str(node) + ", " + str(invar) + ", " + str(identifier))
            all_identifiers[(node, 0)] = identifier
            feature_list.append(identifier)

    if print_identifiers:
        book.save("Atom_Invariants.xls")

    iteration = 1
    while iteration <= 2:
        print("\nIteration:", iteration)
        for node in symbols:
            if is_descriptor(symbols[node]) or symbols[node] == "":
                continue   
            
            print("\nnode:", node)
            atom_hash = all_identifiers[(node, iteration - 1)]
            print("identifier(s) from the previous iteration:", atom_hash)

            neighbor_hash = dict()
            for neighbor_1 in atomistic[node]:
                # the neighbor_1 all need to check whether it is blank node
                if is_descriptor(symbols[neighbor_1]) or symbols[neighbor_1] == "":
                    continue

                for bond in bonds:
                    if bond == tuple(sorted([node, neighbor_1])):
                        if "SINGLE" in bonds[bond]:
                            order = 1
                        elif "DOUBLE" in bonds[bond]:
                            order = 2
                        elif "TRIPLE" in bonds[bond]:
                            order = 3 
                        else:
                            order = 4
                if iteration == 1:
                    neighbor_hash[neighbor_1] = [(order, all_identifiers[(neighbor_1, iteration - 1)])]
                    print("neighbor information",neighbor_1, neighbor_hash[neighbor_1])

                elif iteration == 2:
                    n = []
                    ident = all_identifiers[(neighbor_1, iteration - 1)]
                    print("ident information",neighbor_1, iteration, ident)
                    if type(ident) == dict:
                        for key in ident:
                            n.append((order, ident[key]))
                        print(n)    
                    else:
                        n = [(order, ident)]
                    neighbor_hash[neighbor_1] = n
                    print("neighbor information iteration 2", neighbor_1, neighbor_hash[neighbor_1] )

            print("neighbor (order, prev. identifier):", neighbor_hash)
            
            # populate across hash, which maps atom across descriptor to hash of that atom from previous iteration
            across_hash = dict()
            for neighbor_1 in atomistic[node]:
                if not is_descriptor(symbols[neighbor_1]) :
                    continue
                for neighbor_2 in atomistic[neighbor_1]:
                    if symbols[neighbor_2] == "":
                        for neighbor_3 in atomistic[neighbor_2]:
                            for neighbor_4 in atomistic[neighbor_3]:
                                for bond in bonds:
                                    if bond == tuple(sorted([neighbor_3, neighbor_4])):
                                        if "SINGLE" in bonds[bond]:
                                            order = 1
                                        elif "DOUBLE" in bonds[bond]:
                                            order = 2
                                        elif "TRIPLE" in bonds[bond]:
                                            order = 3 
                                        else:
                                            order = 4
                                list_bonds = [node, neighbor_1, neighbor_2, neighbor_3, neighbor_4]
                                a = symbols[node] != "" and not is_descriptor(symbols[node])
                                b = is_descriptor(symbols[neighbor_1])
                                c = symbols[neighbor_2] == ""
                                d = is_descriptor(symbols[neighbor_3])
                                e = symbols[neighbor_4] != "" and not is_descriptor(symbols[neighbor_4])
                                
                                compatible = bonds_compat_desc(list_bonds, bonds)
                                repeat_nodes = len(set(list_bonds)) == 5
                                if not(a and b and c and d and e and compatible and repeat_nodes): 
                                    continue
                                across_hash[neighbor_4] = (order, all_identifiers[(neighbor_4, iteration - 1)])
                    else:
                        for bond in bonds:
                            if bond == tuple(sorted([neighbor_1, neighbor_2])):
                                if "SINGLE" in bonds[bond]:
                                    order = 1
                                elif "DOUBLE" in bonds[bond]:
                                    order = 2
                                elif "TRIPLE" in bonds[bond]:
                                    order = 3 
                                else:
                                    order = 4
                        list_bonds = [node, neighbor_1, neighbor_2]
                        if not bonds_compat_desc(list_bonds, bonds):
                            continue
                        across_hash[neighbor_2] = (order, all_identifiers[(neighbor_2, iteration - 1)])
            print("across (order, prev. identifier):", across_hash)   

            # if the node is connected to both a distinct descriptor and a blank node
            check_neighbor_empty_node = False
            check_neighbor_descriptor = False
            neighbor_empty_node_index = 0
            neighbor_descriptor_index = 0
            neighbor_empty_node_bond_order = 1
            neighbor_descriptor_bond_order = 1

            for neighbor_1 in atomistic[node]:
                if symbols[neighbor_1] == "":
                    check_neighbor_empty_node = True
                    neighbor_empty_node_index = neighbor_1
                    neighbor_empty_node_bond_type =  atomistic[node][neighbor_empty_node_index]["bond_type"]
                    if "SINGLE" in neighbor_empty_node_bond_type:
                        neighbor_empty_node_bond_order = 1
                    elif "DOUBLE" in neighbor_empty_node_bond_type:
                        neighbor_empty_node_bond_order = 2
                    elif "TRIPLE" in neighbor_empty_node_bond_type:
                        neighbor_empty_node_bond_order = 3 
                    else:
                        neighbor_empty_node_bond_order = 4
                    
                if is_descriptor(symbols[neighbor_1]):
                    check_neighbor_descriptor = True
                    neighbor_descriptor_index = neighbor_1
                    descriptor_bond_type =  atomistic[node][neighbor_descriptor_index]["bond_type"]
                    if "SINGLE" in descriptor_bond_type:
                        neighbor_descriptor_bond_order = 1
                    elif "DOUBLE" in descriptor_bond_type:
                        neighbor_descriptor_bond_order = 2
                    elif "TRIPLE" in descriptor_bond_type:
                        neighbor_descriptor_bond_order = 3
                    else:  
                        neighbor_descriptor_bond_order = 4


            if check_neighbor_empty_node and check_neighbor_descriptor:
                across_hash[node] = (neighbor_empty_node_bond_order, all_identifiers[(node, iteration - 1)])
                across_hash[node] = (neighbor_descriptor_bond_order, all_identifiers[(node, iteration - 1)])
                print("========================================")
                print("update across (order, prev. identifier):", across_hash)

                            
            if iteration == 1:
                if not across_hash:
                    print("\nno across atoms")
                    neighbors = []
                    for key in neighbor_hash:
                        for order_ident in neighbor_hash[key]:
                            neighbors.append(order_ident)
                    print("neighbors combined:", neighbors)                       
                    neighbors = sorted(neighbors)
                    print("sorted neighbors:", neighbors)

                    proper_list = [1, atom_hash]
                    for pair in neighbors:
                        for item in pair:
                            proper_list.append(item)

                    print("iter, atom hash + sorted:", proper_list)
                    identifier = hash(tuple(proper_list)) 
                    print("identifier after hashing:", identifier)

                    all_identifiers[(node, iteration)] = identifier
                    feature_list.append(identifier) 
                else:
                    print("\nacross atoms (adjacent to descriptor)")
                    new_identifiers = dict()
                    for atom_across in across_hash:
                        print("next across:", atom_across)
                        neighbors = []
                        for key in neighbor_hash:
                            for order_ident in neighbor_hash[key]:
                                neighbors.append(order_ident)
                        print("neighbors combined:", neighbors)
                        neighbors.append(across_hash[atom_across]) 
                        print("neighbors + across:", neighbors)

                        if check_neighbor_empty_node and check_neighbor_descriptor:
                            neighbors.append(across_hash[node]) 

                        neighbors = sorted(neighbors)
                        print("sorted:", neighbors)

                        proper_list = [1, atom_hash]
                        for pair in neighbors:
                            for item in pair:
                                proper_list.append(item)

                        print("iter, atom hash + sorted:", proper_list)
                        identifier = hash(tuple(proper_list)) 
                        print("identifier after hashing atom", str(atom_across) + ":", identifier)

                        new_identifiers[atom_across] = identifier 
                        feature_list.append(identifier) 
                    
                    all_identifiers[(node, iteration)] = new_identifiers
            
            elif iteration == 2:
                list_length = []  
                for key in neighbor_hash:    
                    list_length.append(list(range(len(neighbor_hash[key]))))
                print("\nlength of neighbors:", list_length)
                neighbor_combos = [p for p in itertools.product(*list_length)] 
                print("potential combinations:", neighbor_combos)

                if not across_hash: # interior
                    print("\nno across atoms")
                    for n_c in neighbor_combos: 
                        neighbors = []
                        i = 0
                        for key in neighbor_hash:
                            order_ident = neighbor_hash[key]
                            neighbors.append(order_ident[n_c[i]])
                            i += 1
                        print("neighbors combined:", neighbors)                       
                        neighbors = sorted(neighbors)
                        print("sorted neighbors:", neighbors)

                        proper_list = [2, atom_hash]
                        for pair in neighbors:
                            for item in pair:
                                proper_list.append(item)
                        print("iter, atom hash + sorted:", proper_list)
                        identifier = hash(tuple(proper_list)) 
                        print("identifier after hashing:", identifier)

                        feature_list.append(identifier) 
                else:
                    print("\nacross atoms (adjacent to descriptor)")
                    for atom_across in across_hash:
                        print("next across:", atom_across)
                        for n_c in neighbor_combos: 
                            neighbors = []
                            i = 0
                            for key in neighbor_hash:
                                order_ident = neighbor_hash[key]
                                neighbors.append(order_ident[n_c[i]]) 
                                i += 1
                            print("neighbors combined:", neighbors)
                            neighbors.append((across_hash[atom_across][0], across_hash[atom_across][1][node]))
                            print("neighbors + across:", neighbors)

                            if check_neighbor_empty_node and check_neighbor_descriptor:
                               neighbors.append((across_hash[node][0], across_hash[node][1][node])) 

                            neighbors = sorted(neighbors)


                            print("sorted:", neighbors)

                            proper_list = [2, atom_hash[atom_across]]
                            for pair in neighbors:
                                for item in pair:
                                    proper_list.append(item)



                            print("iter, atom hash + sorted:", proper_list)
                            identifier = hash(tuple(proper_list)) 
                            print("identifier after hashing atom", str(atom_across) + ":", identifier)

                            feature_list.append(identifier) 

        iteration += 1
    
    print("\nidentifier, remainder (index = 1)")
    # convert to an array
    fp = np.zeros(fp_size)
    for f in feature_list:
        r = f % fp_size
        fp[r] = 1
        print(str(f) + ", " + str(r))
    
    if not print_all:
        enablePrint()

    
    return fp    


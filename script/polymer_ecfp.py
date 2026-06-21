from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np
import itertools
import networkx as nx    
import copy
try:
    import xlwt
except ImportError:
    xlwt = None
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
    #print(graphs["topology"])
    #symbols = nx.get_node_attributes(graphs["topology"], "symbol")
    #print(symbols)
    #for node in symbols:
    #    print(node, symbols[node])
    #print(ids)
    # get the topological id of the atom
    id = ids[atom]
    #print(individual_clusters)
    # determine if the topological id is in a stochastic object
    for stochastic_object in individual_clusters:
        #print(atom, id, stochastic_object)
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

    for i in range(0, len(backbone)):
        neighbors = atomistic[backbone[i]]
        amount_of_neighbor_descriptors = 0
        for n in neighbors:
            if is_descriptor(symbols[n]):
                amount_of_neighbor_descriptors += 1
                all_backbone.add(n)
        if amount_of_neighbor_descriptors >= 2:
            all_backbone.add(backbone[i])
            
    #print("all_backbone result:",all_backbone)
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

def _legacy_ECFP_two_iterations(graphs, individual_clusters, iteration_input = 2, repeat_unit_property = False, backbone_property = False, fp_size = 2048, print_identifiers = False, print_all = True, graph_name = ""):
    # https://chemicbook.com/2021/03/25/a-beginners-guide-for-understanding-extended-connectivity-fingerprints.html
    # remove [H] from graph

    if not print_all:
        blockPrint()

    if print_identifiers:
        if xlwt is None:
            raise ImportError("print_identifiers=True requires the optional xlwt package")
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

        if repeat_unit_property == True:
            sheet1.write(0, 8, "Repeat")
        if backbone_property == True:
            sheet1.write(0, 9, "Backbone")
        row = 1

    atomistic = graphs["atomistic"]
    #print(type(atomistic), nx.is_directed(atomistic),isinstance(atomistic, nx.MultiGraph))



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

    if backbone_property == True:
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
            if repeat_unit_property == True:
                repeat = int(in_repeat(graphs, individual_clusters, node))
            if backbone_property == True:
                backbone = int(node in backbone_atoms)
            
            if repeat_unit_property == True and backbone_property == True:
                invar = (neighbors, valence_atoms - hydrogens_atom, atomic_atom, mass_atom, charge_atom, hydrogens_atom, ring, repeat, backbone)
            elif repeat_unit_property == True and backbone_property == False:
                invar = (neighbors, valence_atoms - hydrogens_atom, atomic_atom, mass_atom, charge_atom, hydrogens_atom, ring, repeat)
            elif repeat_unit_property == False and backbone_property == True:
                invar = (neighbors, valence_atoms - hydrogens_atom, atomic_atom, mass_atom, charge_atom, hydrogens_atom, ring, backbone)
            else:
                invar = (neighbors, valence_atoms - hydrogens_atom, atomic_atom, mass_atom, charge_atom, hydrogens_atom, ring)

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
    while iteration <= iteration_input:
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
            neighbor_descriptor_extra_neighbor_index_list = []
            neighbor_descriptor_extra_neighbor_bond_list = []

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
                    # check whether the descriptor is connected with other atoms
                    amount_of_neighbors_for_descriptor = len(atomistic[neighbor_1])
                    if amount_of_neighbors_for_descriptor > 2:
                        for neighbor_2 in atomistic[neighbor_1]:
                            if neighbor_2 != node and symbols[neighbor_2] != "":
                                descriptor_neighbor_index = neighbor_2
                                descriptor_neighbor_bond_type =  atomistic[neighbor_1][descriptor_neighbor_index]["bond_type"]
                                if "SINGLE" in descriptor_neighbor_bond_type:
                                    descriptor_neighbor_bond_order = 1
                                elif "DOUBLE" in descriptor_neighbor_bond_type:
                                    descriptor_neighbor_bond_order = 2
                                elif "TRIPLE" in descriptor_neighbor_bond_type:
                                    descriptor_neighbor_bond_order = 3 
                                else:
                                    descriptor_neighbor_bond_order = 4
                                neighbor_descriptor_extra_neighbor_index_list.append(descriptor_neighbor_index)
                                neighbor_descriptor_extra_neighbor_bond_list.append(descriptor_neighbor_bond_order)
                                
            


            if check_neighbor_empty_node and check_neighbor_descriptor:
                across_hash[node] = (neighbor_empty_node_bond_order, all_identifiers[(node, iteration - 1)])
                across_hash[node] = (neighbor_descriptor_bond_order, all_identifiers[(node, iteration - 1)])
                #if len(neighbor_descriptor_extra_neighbor_index_list)>0:
                #    print(len(neighbor_descriptor_extra_neighbor_index_list))
                #    print(node, neighbor_descriptor_extra_neighbor_index_list, neighbor_descriptor_extra_neighbor_bond_list)
                #    for index_i in range(0, len(neighbor_descriptor_extra_neighbor_index_list)):
                #        across_hash[neighbor_descriptor_extra_neighbor_index_list[index_i]] = (neighbor_descriptor_extra_neighbor_bond_list[index_i], all_identifiers[(neighbor_descriptor_extra_neighbor_index_list[index_i], iteration - 1)])
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
                    print(across_hash)
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


# Generic N-iteration implementation.  It intentionally redefines ECFP below
# so existing callers keep their public API while gaining arbitrary radii.
def _pecfp_bond_order(atomistic, first, second):
    bond_type = atomistic[first][second]["bond_type"]
    if "SINGLE" in bond_type:
        return 1
    if "DOUBLE" in bond_type:
        return 2
    if "TRIPLE" in bond_type:
        return 3
    return 4


def _pecfp_is_atom(symbol):
    return symbol != "" and not is_descriptor(symbol)


def _pecfp_all_identifiers(context_identifiers):
    return set().union(*context_identifiers.values())


def _pecfp_across_atoms(atomistic, symbols, node):
    """Return compatible atoms reachable across descriptors and their bond order."""
    bonds = nx.get_edge_attributes(atomistic, "bond_type")
    across = {}
    for descriptor in atomistic[node]:
        if not is_descriptor(symbols[descriptor]):
            continue
        for next_node in atomistic[descriptor]:
            if next_node == node:
                continue
            if symbols[next_node] == "":
                for next_descriptor in atomistic[next_node]:
                    if not is_descriptor(symbols[next_descriptor]):
                        continue
                    for target in atomistic[next_descriptor]:
                        path = [node, descriptor, next_node, next_descriptor, target]
                        if (target != node and _pecfp_is_atom(symbols[target])
                                and len(set(path)) == len(path)
                                and bonds_compat_desc(path, bonds)):
                            across[target] = _pecfp_bond_order(atomistic, next_descriptor, target)
            elif _pecfp_is_atom(symbols[next_node]):
                path = [node, descriptor, next_node]
                if bonds_compat_desc(path, bonds):
                    across[next_node] = _pecfp_bond_order(atomistic, descriptor, next_node)
    return across


def ECFP(graphs, individual_clusters, iteration_input=2, repeat_unit_property=False,
         backbone_property=False, fp_size=2048, print_identifiers=False,
         print_all=True, graph_name=""):
    """Generate pECFP using atom invariants plus ``iteration_input`` expansions."""
    if not isinstance(iteration_input, int) or iteration_input < 0:
        raise ValueError("iteration_input must be a non-negative integer")
    if not isinstance(fp_size, int) or fp_size <= 0:
        raise ValueError("fp_size must be a positive integer")

    def log(*args):
        if print_all:
            print(*args)

    atomistic = graphs["atomistic"].copy()
    atomistic.remove_nodes_from(
        node for node, attrs in atomistic.nodes(data=True) if attrs.get("symbol") == "H"
    )
    working_graphs = dict(graphs)
    working_graphs["atomistic"] = atomistic
    symbols = nx.get_node_attributes(atomistic, "symbol")

    plt.figure(figsize=(8, 6))
    nx.draw(atomistic, with_labels=True, node_color="lightblue", node_size=800,
            font_size=14, font_weight="bold")
    plt.savefig("network_graph" + graph_name + ".png", format="PNG")
    plt.close()

    if print_identifiers:
        if xlwt is None:
            raise ImportError("print_identifiers=True requires the optional xlwt package")
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Sheet 1")
        headers = ["Node", "Neighbors", "Valence - Hydrogens", "Atomic #",
                   "Atom Mass", "Atom Charge", "Hydrogens", "Ring"]
        if repeat_unit_property:
            headers.append("Repeat")
        if backbone_property:
            headers.append("Backbone")
        for column, header in enumerate(headers):
            sheet1.write(0, column, header)
        row = 1

    attributes = {
        "formal_charge": nx.get_node_attributes(atomistic, "formal_charge"),
        "valence": nx.get_node_attributes(atomistic, "valence"),
        "atomic_num": nx.get_node_attributes(atomistic, "atomic_num"),
        "mass": nx.get_node_attributes(atomistic, "mass"),
        "num_hs": nx.get_node_attributes(atomistic, "num_hs"),
    }
    backbone_atoms = (get_all_backbone_atoms(working_graphs, individual_clusters)
                      if backbone_property else set())
    state, feature_list = {}, []

    for node, symbol in symbols.items():
        if not _pecfp_is_atom(symbol):
            continue
        invariants = [len(atomistic[node]),
                      attributes["valence"].get(node, 0) - attributes["num_hs"].get(node, 0),
                      attributes["atomic_num"].get(node, 0), attributes["mass"].get(node, 0),
                      attributes["formal_charge"].get(node, 0), attributes["num_hs"].get(node, 0),
                      int(bool(in_ring(atomistic, node)))]
        if repeat_unit_property:
            invariants.append(int(in_repeat(working_graphs, individual_clusters, node)))
        if backbone_property:
            invariants.append(int(node in backbone_atoms))
        identifier = hash(tuple(invariants))
        state[node] = {node: {identifier}}
        feature_list.append(identifier)
        if print_identifiers:
            sheet1.write(row, 0, node)
            for column, value in enumerate(invariants, start=1):
                sheet1.write(row, column, value)
            row += 1

    if print_identifiers:
        book.save("Atom_Invariants.xls")

    for iteration in range(1, iteration_input + 1):
        log("Iteration:", iteration)
        next_state = {}
        for node in state:
            local_messages = []
            for neighbor in atomistic[node]:
                if _pecfp_is_atom(symbols[neighbor]):
                    order = _pecfp_bond_order(atomistic, node, neighbor)
                    local_messages.append([(order, identifier) for identifier in
                                           sorted(_pecfp_all_identifiers(state[neighbor]))])
            local_combinations = list(itertools.product(*local_messages))
            across = _pecfp_across_atoms(atomistic, symbols, node)
            next_state[node] = {}
            for context, across_order in (across or {node: None}).items():
                own = state[node].get(context, _pecfp_all_identifiers(state[node]))
                crossed = (state[context].get(node, _pecfp_all_identifiers(state[context]))
                           if across_order is not None else {None})
                produced = set()
                for own_identifier in own:
                    for local in local_combinations:
                        for crossed_identifier in crossed:
                            neighbors = list(local)
                            if across_order is not None:
                                neighbors.append((across_order, crossed_identifier))
                            hashed = [iteration, own_identifier]
                            for order, identifier in sorted(neighbors):
                                hashed.extend((order, identifier))
                            identifier = hash(tuple(hashed))
                            produced.add(identifier)
                            feature_list.append(identifier)
                next_state[node][context] = produced
        state = next_state

    fp = np.zeros(fp_size)
    for identifier in feature_list:
        fp[identifier % fp_size] = 1
    return fp

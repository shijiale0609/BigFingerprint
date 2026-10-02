"""Fast filters and RDKit fragment helpers for BigSMARTS searches."""

import re

import networkx as nx
from rdkit import Chem

from .graphs import desc_regex
from .search_tools import generate_NetworkX_graphs, identify_cycles


def generate_filters(bigsmiles):
    """Calculate inexpensive structural filters for a BigSMILES string.

    Parameters
    ----------
    bigsmiles : str
        BigSMILES representation to summarize.

    Returns
    -------
    dict
        Nested-object, stochastic-object, descriptor, fragment, and graph
        information used to prefilter similarity-search candidates.
    """
    graphs = generate_NetworkX_graphs(bigsmiles, is_bigsmarts=False)
    group_cycles, _, _ = identify_cycles(graphs)

    level = nx.get_node_attributes(graphs["atomistic"], "level")
    contains_nested = any(value > 0 for value in level.values())

    max_descriptors = 0
    symbols = nx.get_node_attributes(graphs["multidigraph"], "symbol")
    edges = graphs["multidigraph"].edges
    for node, symbol in symbols.items():
        if not list(re.finditer(desc_regex, symbol)):
            descriptor_count = sum(node in edge for edge in edges)
            max_descriptors = max(max_descriptors, descriptor_count)

    return {
        "bigsmiles": bigsmiles,
        "nested": contains_nested,
        "num_objects": len(group_cycles),
        "max_desc": max_descriptors,
        "ring_smiles": to_smiles(graphs),
        "atomistic": graphs["atomistic"],
        "topology": graphs["topology"],
        "top_undir": graphs["top_undir"],
        "multidigraph": graphs["multidigraph"],
    }


def cycle_to_ring_smiles(graphs, cycle):
    """Convert one atomistic graph cycle to an RDKit SMILES fragment."""
    symbols = nx.get_node_attributes(graphs["atomistic"], "symbol")
    formal_charge = nx.get_node_attributes(graphs["atomistic"], "formal_charge")
    is_aromatic = nx.get_node_attributes(graphs["atomistic"], "is_aromatic")
    bonds = nx.get_edge_attributes(graphs["atomistic"], "bond_type_object")
    ids = nx.get_node_attributes(graphs["atomistic"], "ids")

    node_to_index = {}
    molecule = Chem.RWMol()
    for node in graphs["atomistic"].nodes():
        if (
            ids[node] in cycle
            and node not in graphs["descriptors"]
            and symbols[node] != ""
        ):
            atom = Chem.Atom(symbols[node])
            atom.SetFormalCharge(formal_charge[node])
            atom.SetIsAromatic(is_aromatic[node])
            node_to_index[node] = molecule.AddAtom(atom)

    already_added = set()
    for first, second in graphs["atomistic"].edges():
        if ids[first] not in cycle or ids[second] not in cycle:
            continue
        if first in graphs["descriptors"] or second in graphs["descriptors"]:
            continue
        first_index = node_to_index[first]
        second_index = node_to_index[second]
        atom_pair = tuple(sorted((first_index, second_index)))
        if atom_pair not in already_added:
            molecule.AddBond(first_index, second_index, bonds[first, second])
            already_added.add(atom_pair)

    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule)


# Preserve the original public spelling for downstream notebooks.
cycle_to_ring_SMILES = cycle_to_ring_smiles


def to_smiles(graphs):
    """Convert stochastic-object cycles and end groups to SMILES fragments."""
    cycles, clusters, _ = identify_cycles(graphs)

    end_groups = []
    topological_ids = set(nx.get_node_attributes(graphs["atomistic"], "ids").values())
    for topological_id in topological_ids:
        if not any(topological_id in cluster for cluster in clusters):
            end_groups.append(cycle_to_ring_smiles(graphs, [topological_id]))

    repeat_units = []
    for stochastic_object in cycles:
        repeat_units.append(
            [cycle_to_ring_smiles(graphs, cycle) for cycle in stochastic_object]
        )

    return repeat_units, end_groups


def search_small_molecules(graphs_q, graphs_t):
    """Search RDKit fragments extracted from query and target polymer graphs."""
    atomistic = graphs_q["atomistic"]
    symbols = nx.get_node_attributes(atomistic, "symbol")
    for node, symbol in symbols.items():
        if symbol == "?*":
            symbols[node] = "Fm"
    nx.set_node_attributes(atomistic, symbols, "symbol")
    graphs_q["atomistic"] = atomistic

    repeats_q, _ = to_smiles(graphs_q)
    repeats_t, _ = to_smiles(graphs_t)

    def localization():
        local_elements = nx.get_node_attributes(atomistic, "local_el")
        for local_element in local_elements.values():
            for smarts in local_element["ru_local_el"]:
                query = Chem.MolFromSmarts(smarts)
                if query is None:
                    return False
                if not any(
                    Chem.MolFromSmiles(target).HasSubstructMatch(query)
                    for target_object in repeats_t
                    for target in target_object
                ):
                    return False
        return True

    for query_object in repeats_q:
        for query_smiles in query_object:
            if query_smiles == "*[Fm]":
                if not localization():
                    return False
                continue

            query = Chem.MolFromSmarts(query_smiles.replace("[Fm]", "*"))
            if query is None:
                return False
            found = any(
                Chem.MolFromSmiles(target).HasSubstructMatch(query)
                for target_object in repeats_t
                for target in target_object
            )
            if not found:
                return False

    return True

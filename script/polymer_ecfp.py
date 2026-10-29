"""Polymer extended-connectivity fingerprint (p-ECFP) generation."""

import itertools

import networkx as nx
import numpy as np

try:
    import xlwt
except ImportError:  # Optional diagnostic dependency.
    xlwt = None


def in_ring(atomistic, atom):
    symbols = nx.get_node_attributes(atomistic, "symbol")
    for neighbor in atomistic[atom]:
        count = 0
        for path in nx.all_simple_paths(atomistic, atom, neighbor):
            if not any(is_descriptor(symbols[node]) for node in path):
                count += 1
            if count == 2:
                return True
    return False


def in_repeat(graphs, individual_clusters, atom):
    topological_id = graphs["ids"][atom]
    return any(topological_id in cluster for cluster in individual_clusters)


def get_all_backbone_atoms(graphs, individual_clusters):
    atomistic = graphs["atomistic"]
    symbols = nx.get_node_attributes(atomistic, "symbol")

    backbone = set()
    for node, symbol in symbols.items():
        if is_descriptor(symbol):
            for neighbor in atomistic[node]:
                if in_repeat(graphs, individual_clusters, neighbor):
                    backbone.add(neighbor)

    backbone = list(backbone)
    all_backbone = set()
    for first_index, first in enumerate(backbone):
        for second in backbone[first_index + 1 :]:
            for path in nx.all_simple_paths(atomistic, first, second):
                if not any(is_descriptor(symbols[node]) for node in path):
                    all_backbone.update(path)

    for atom in backbone:
        descriptor_neighbors = [
            neighbor for neighbor in atomistic[atom] if is_descriptor(symbols[neighbor])
        ]
        all_backbone.update(descriptor_neighbors)
        if len(descriptor_neighbors) >= 2:
            all_backbone.add(atom)

    return all_backbone


def is_descriptor(symbol):
    return any(marker in symbol for marker in (">", "<", "$"))


def bonds_compat_desc(path_nodes, bonds):
    path = []
    for first, second in zip(path_nodes, path_nodes[1:]):
        bond_id = tuple(sorted((first, second)))
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
        ["2b", "2a"],
    ]
    return path in possible


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
                        if (
                            target != node
                            and _pecfp_is_atom(symbols[target])
                            and len(set(path)) == len(path)
                            and bonds_compat_desc(path, bonds)
                        ):
                            across[target] = _pecfp_bond_order(
                                atomistic, next_descriptor, target
                            )
            elif _pecfp_is_atom(symbols[next_node]):
                path = [node, descriptor, next_node]
                if bonds_compat_desc(path, bonds):
                    across[next_node] = _pecfp_bond_order(
                        atomistic, descriptor, next_node
                    )
    return across


def ECFP(
    graphs,
    individual_clusters,
    iteration_input=2,
    repeat_unit_property=False,
    backbone_property=False,
    fp_size=2048,
    print_identifiers=False,
    print_all=True,
    graph_name="",
    save_graph=False,
):
    """Generate a p-ECFP vector at radius 0, 1, or 2.

    Parameters
    ----------
    graphs : dict
        Graph collection returned by :func:`generate_NetworkX_graphs`.
    individual_clusters : list[set]
        Stochastic-object clusters returned by :func:`identify_cycles`.
    iteration_input : int, default=2
        p-ECFP radius: 0, 1, or 2. As in RDKit's Morgan fingerprints, radius
        counts neighborhood-update iterations after atom initialization.
        Radius 0 uses only the initial atom identifiers.
    repeat_unit_property : bool, default=False
        Include repeat-unit membership in the initial atom invariants.
    backbone_property : bool, default=False
        Include backbone membership in the initial atom invariants.
    fp_size : int, default=2048
        Number of bits in the folded fingerprint.
    print_identifiers : bool, default=False
        Export initial atom invariants to ``Atom_Invariants.xls``.
    print_all : bool, default=True
        Print progress for each update round.
    graph_name : str, default=""
        Suffix for the optional network-graph filename.
    save_graph : bool, default=False
        Save the atomistic NetworkX graph as a PNG image.

    Returns
    -------
    numpy.ndarray
        Binary p-ECFP vector with shape ``(fp_size,)``.
    """
    if not isinstance(iteration_input, int) or iteration_input < 0:
        raise ValueError("iteration_input must be a non-negative integer")
    if iteration_input > 2:
        raise ValueError(
            "iteration_input must be at most 2; iterations greater than 2 "
            "are not currently supported"
        )
    if not isinstance(fp_size, int) or fp_size <= 0:
        raise ValueError("fp_size must be a positive integer")

    def log(*args):
        if print_all:
            print(*args)

    atomistic = graphs["atomistic"].copy()
    atomistic.remove_nodes_from(
        [
            node
            for node, attributes in atomistic.nodes(data=True)
            if attributes.get("symbol") == "H"
        ]
    )
    working_graphs = dict(graphs)
    working_graphs["atomistic"] = atomistic
    symbols = nx.get_node_attributes(atomistic, "symbol")

    if save_graph:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "save_graph=True requires the optional 'debug' dependencies. "
                "Install them with 'pip install BigFingerprint[debug]'."
            ) from exc
        plt.figure(figsize=(8, 6))
        nx.draw(
            atomistic,
            with_labels=True,
            node_color="lightblue",
            node_size=800,
            font_size=14,
            font_weight="bold",
        )
        plt.savefig("network_graph" + graph_name + ".png", format="PNG")
        plt.close()

    if print_identifiers:
        if xlwt is None:
            raise ImportError(
                "print_identifiers=True requires the optional xlwt package"
            )
        book = xlwt.Workbook(encoding="utf-8")
        sheet = book.add_sheet("Sheet 1")
        headers = [
            "Node",
            "Neighbors",
            "Valence - Hydrogens",
            "Atomic #",
            "Atom Mass",
            "Atom Charge",
            "Hydrogens",
            "Ring",
        ]
        if repeat_unit_property:
            headers.append("Repeat")
        if backbone_property:
            headers.append("Backbone")
        for column, header in enumerate(headers):
            sheet.write(0, column, header)
        row = 1

    attributes = {
        "formal_charge": nx.get_node_attributes(atomistic, "formal_charge"),
        "valence": nx.get_node_attributes(atomistic, "valence"),
        "atomic_num": nx.get_node_attributes(atomistic, "atomic_num"),
        "mass": nx.get_node_attributes(atomistic, "mass"),
        "num_hs": nx.get_node_attributes(atomistic, "num_hs"),
    }
    backbone_atoms = (
        get_all_backbone_atoms(working_graphs, individual_clusters)
        if backbone_property
        else set()
    )
    state = {}
    feature_list = []

    for node, symbol in symbols.items():
        if not _pecfp_is_atom(symbol):
            continue
        invariants = [
            len(atomistic[node]),
            attributes["valence"].get(node, 0) - attributes["num_hs"].get(node, 0),
            attributes["atomic_num"].get(node, 0),
            attributes["mass"].get(node, 0),
            attributes["formal_charge"].get(node, 0),
            attributes["num_hs"].get(node, 0),
            int(bool(in_ring(atomistic, node))),
        ]
        if repeat_unit_property:
            invariants.append(int(in_repeat(working_graphs, individual_clusters, node)))
        if backbone_property:
            invariants.append(int(node in backbone_atoms))
        identifier = hash(tuple(invariants))
        state[node] = {node: {identifier}}
        feature_list.append(identifier)
        if print_identifiers:
            sheet.write(row, 0, node)
            for column, value in enumerate(invariants, start=1):
                sheet.write(row, column, value)
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
                    local_messages.append(
                        [
                            (order, identifier)
                            for identifier in sorted(
                                _pecfp_all_identifiers(state[neighbor])
                            )
                        ]
                    )
            local_combinations = list(itertools.product(*local_messages))
            across = _pecfp_across_atoms(atomistic, symbols, node)
            next_state[node] = {}
            for context, across_order in (across or {node: None}).items():
                own = state[node].get(context, _pecfp_all_identifiers(state[node]))
                crossed = (
                    state[context].get(node, _pecfp_all_identifiers(state[context]))
                    if across_order is not None
                    else {None}
                )
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

    fingerprint = np.zeros(fp_size)
    for identifier in feature_list:
        fingerprint[identifier % fp_size] = 1
    return fingerprint

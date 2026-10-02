import re
from dataclasses import dataclass
from typing import Optional

try:
    from rdkit import Chem
except ImportError:  # Keep this module importable in non-RDKit environments.
    Chem = None


DESCRIPTOR_RE = re.compile(r"\[(?:\$|<|>|\$=|<=|>=)\d*\]")
_DESCRIPTOR_MAP_BASE = 900000


@dataclass(frozen=True)
class OrientedRepeatFragment:
    smiles: str
    mapped_smiles: str
    repeat_descriptor: str
    repeat_descriptor_side: str
    insert_side: str
    anchor_atom_map: int
    attachment_index: Optional[int] = None


def terminal_descriptors(repeat_unit_smiles):
    """Return the first and last BigSMILES bonding descriptors in a repeat unit."""
    matches = list(DESCRIPTOR_RE.finditer(repeat_unit_smiles))
    if not matches:
        raise ValueError(f"No BigSMILES bonding descriptor found: {repeat_unit_smiles}")
    return matches[0].group(0), matches[-1].group(0)


def orient_repeat_unit_fragment(
    repeat_unit_smiles, repeat_descriptor_side, insert_side
):
    """
    Convert a BigSMILES repeat unit into a SMILES fragment oriented for attachment.

    Args:
        repeat_unit_smiles: Repeat unit containing BigSMILES descriptors, e.g. "[<]CCO[>]".
        repeat_descriptor_side: Which descriptor of this repeat unit is used for attachment:
            "left" or "right".
        insert_side: Where this fragment will be inserted relative to the stochastic object:
            "right" means object + fragment, so the attachment atom must be first.
            "left" means fragment + object, so the attachment atom must be last.

    Returns:
        OrientedRepeatFragment. The `smiles` field has all BigSMILES descriptors removed
        and is oriented so concatenating it on the requested side attaches through the
        selected repeat-unit descriptor.
    """
    if repeat_descriptor_side not in {"left", "right"}:
        raise ValueError("repeat_descriptor_side must be 'left' or 'right'")
    if insert_side not in {"left", "right"}:
        raise ValueError("insert_side must be 'left' or 'right'")

    _require_rdkit()

    marked_smiles, descriptor_maps, descriptor_tokens = _mark_descriptors(
        repeat_unit_smiles
    )
    mol = Chem.MolFromSmiles(marked_smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse repeat unit after descriptor marking: {marked_smiles}"
        )

    descriptor_map = descriptor_maps[repeat_descriptor_side]
    anchor_atom_map = _find_anchor_atom_map(mol, descriptor_map)
    fragment = _keep_selected_dummy_descriptor(mol, descriptor_map)

    # Keep the selected descriptor as a temporary terminal dummy.  This also
    # handles branch endings such as C(=O), where the attachment atom is not the
    # final lexical atom in the SMILES text.
    dummy_position = "first" if insert_side == "right" else "last"
    smiles, mapped_smiles = _oriented_smiles_by_selected_dummy(
        fragment,
        descriptor_map,
        dummy_position,
    )

    return OrientedRepeatFragment(
        smiles=smiles,
        mapped_smiles=mapped_smiles,
        repeat_descriptor=descriptor_tokens[repeat_descriptor_side],
        repeat_descriptor_side=repeat_descriptor_side,
        insert_side=insert_side,
        anchor_atom_map=anchor_atom_map,
    )


def orient_symmetric_descriptor_fragments(repeat_unit_smiles, insert_side):
    """
    Return both possible orientations for a repeat unit attached through a symmetric [$] end.

    This is the helper to use when a [$] terminal can match either side of the repeat unit.
    Duplicate SMILES are removed while preserving order.
    """
    fragments = [
        orient_repeat_unit_fragment(repeat_unit_smiles, "left", insert_side),
        orient_repeat_unit_fragment(repeat_unit_smiles, "right", insert_side),
    ]
    unique = []
    seen = set()
    for fragment in fragments:
        if fragment.smiles in seen:
            continue
        seen.add(fragment.smiles)
        unique.append(fragment)
    return unique


def bigsmiles_repeat_to_wildcard_smiles(repeat_unit_smiles):
    """Replace all BigSMILES bonding descriptors in one repeat unit with bare wildcards."""
    return DESCRIPTOR_RE.sub("*", repeat_unit_smiles)


def orient_wildcard_repeat_fragment(wildcard_smiles, wildcard_index, insert_side):
    """
    Orient a repeat-unit fragment by treating one wildcard as the attachment site.

    Args:
        wildcard_smiles: Repeat-unit SMILES containing wildcard atoms, e.g. "*CCC(C)*".
        wildcard_index: 0-based index of the wildcard atom used as the attachment site.
        insert_side: "right" for object + fragment, "left" for fragment + object.

    Returns:
        OrientedRepeatFragment whose `smiles` has all wildcard atoms removed and is
        oriented so the attachment atom is first (`insert_side="right"`) or last
        (`insert_side="left"`).
    """
    if insert_side not in {"left", "right"}:
        raise ValueError("insert_side must be 'left' or 'right'")

    _require_rdkit()

    marked_smiles, wildcard_maps = _mark_wildcards(wildcard_smiles)
    if wildcard_index < 0 or wildcard_index >= len(wildcard_maps):
        raise IndexError(
            f"wildcard_index {wildcard_index} is out of range for "
            f"{len(wildcard_maps)} wildcard atoms"
        )

    mol = Chem.MolFromSmiles(marked_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse wildcard repeat unit: {marked_smiles}")

    descriptor_map = wildcard_maps[wildcard_index]
    anchor_atom_map = _find_anchor_atom_map(mol, descriptor_map)
    fragment = _keep_selected_dummy_descriptor(mol, descriptor_map)

    dummy_position = "first" if insert_side == "right" else "last"
    smiles, mapped_smiles = _oriented_smiles_by_selected_dummy(
        fragment,
        descriptor_map,
        dummy_position,
    )

    return OrientedRepeatFragment(
        smiles=smiles,
        mapped_smiles=mapped_smiles,
        repeat_descriptor="*",
        repeat_descriptor_side=str(wildcard_index),
        insert_side=insert_side,
        anchor_atom_map=anchor_atom_map,
        attachment_index=wildcard_index,
    )


def orient_all_wildcard_repeat_fragments(wildcard_smiles, insert_side):
    """
    Orient one wildcard repeat unit using each wildcard as a possible attachment site.

    Example:
        "*CCC(C)*", insert_side="right" -> fragments corresponding to
        "-CCC(C)" and "-C(C)CC".
    """
    _, wildcard_maps = _mark_wildcards(wildcard_smiles)
    fragments = [
        orient_wildcard_repeat_fragment(wildcard_smiles, idx, insert_side)
        for idx in range(len(wildcard_maps))
    ]
    return _unique_fragments(fragments)


def orient_bigsmiles_repeat_unit_fragments(
    repeat_unit_smiles, insert_side, descriptor="[$]"
):
    """
    Orient all possible terminal fragments from a BigSMILES repeat unit.

    This is the direct helper for the [$] ambiguity:
        "[$]CCC(C)[$]" -> "*CCC(C)*" -> both wildcard attachment sites.

    Args:
        repeat_unit_smiles: Repeat unit containing BigSMILES descriptors.
        insert_side: "right" for object + fragment, "left" for fragment + object.
        descriptor: Which descriptor token is eligible as the attachment site.
            Defaults to "[$]". Numbered descriptors such as "[$2]" are also supported.
    """
    descriptor_occurrences = [
        idx
        for idx, match in enumerate(DESCRIPTOR_RE.finditer(repeat_unit_smiles))
        if match.group(0) == descriptor
    ]
    if not descriptor_occurrences:
        return []

    wildcard_smiles = bigsmiles_repeat_to_wildcard_smiles(repeat_unit_smiles)
    fragments = [
        orient_wildcard_repeat_fragment(wildcard_smiles, idx, insert_side)
        for idx in descriptor_occurrences
    ]
    return _unique_fragments(fragments)


def _require_rdkit():
    if Chem is None:
        raise ImportError(
            "RDKit is required for repeat-unit SMILES orientation. "
            "Install RDKit in the Python environment used to run this module."
        )


def _mark_descriptors(repeat_unit_smiles):
    matches = list(DESCRIPTOR_RE.finditer(repeat_unit_smiles))
    if not matches:
        raise ValueError(f"No BigSMILES bonding descriptor found: {repeat_unit_smiles}")

    descriptor_maps = {
        "left": _DESCRIPTOR_MAP_BASE + 1,
        "right": _DESCRIPTOR_MAP_BASE + len(matches),
    }
    descriptor_tokens = {
        "left": matches[0].group(0),
        "right": matches[-1].group(0),
    }

    pieces = []
    cursor = 0
    for idx, match in enumerate(matches, start=1):
        pieces.append(repeat_unit_smiles[cursor : match.start()])
        pieces.append(f"[*:{_DESCRIPTOR_MAP_BASE + idx}]")
        cursor = match.end()
    pieces.append(repeat_unit_smiles[cursor:])

    return "".join(pieces), descriptor_maps, descriptor_tokens


def _mark_wildcards(wildcard_smiles):
    pieces = []
    wildcard_maps = []
    cursor = 0
    wildcard_index = 0

    for match in re.finditer(r"\[\*\]|\*", wildcard_smiles):
        pieces.append(wildcard_smiles[cursor : match.start()])
        wildcard_index += 1
        atom_map = _DESCRIPTOR_MAP_BASE + wildcard_index
        wildcard_maps.append(atom_map)
        pieces.append(f"[*:{atom_map}]")
        cursor = match.end()

    pieces.append(wildcard_smiles[cursor:])

    if not wildcard_maps:
        raise ValueError(f"No wildcard atom found: {wildcard_smiles}")

    return "".join(pieces), wildcard_maps


def _unique_fragments(fragments):
    unique = []
    seen = set()
    for fragment in fragments:
        if fragment.smiles in seen:
            continue
        seen.add(fragment.smiles)
        unique.append(fragment)
    return unique


def _find_anchor_atom_map(mol, descriptor_map):
    descriptor_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == descriptor_map
    ]
    if len(descriptor_atoms) != 1:
        raise ValueError(
            f"Expected exactly one descriptor dummy with map {descriptor_map}"
        )

    neighbors = [
        atom for atom in descriptor_atoms[0].GetNeighbors() if atom.GetAtomicNum() != 0
    ]
    if len(neighbors) != 1:
        raise ValueError(
            "Expected the selected descriptor to be attached to exactly one molecular atom"
        )

    return neighbors[0].GetIdx() + 1


def _remove_dummy_descriptors(mol):
    rw_mol = Chem.RWMol(mol)

    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetAtomMapNum(atom.GetIdx() + 1)

    dummy_indices = [
        atom.GetIdx()
        for atom in rw_mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() >= _DESCRIPTOR_MAP_BASE
    ]
    for atom_idx in sorted(dummy_indices, reverse=True):
        rw_mol.RemoveAtom(atom_idx)

    fragment = rw_mol.GetMol()
    Chem.SanitizeMol(fragment)
    return fragment


def _keep_selected_dummy_descriptor(mol, selected_descriptor_map):
    rw_mol = Chem.RWMol(mol)

    dummy_indices_to_remove = []
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() >= _DESCRIPTOR_MAP_BASE:
            if atom.GetAtomMapNum() != selected_descriptor_map:
                dummy_indices_to_remove.append(atom.GetIdx())
        elif atom.GetAtomicNum() != 0:
            atom.SetAtomMapNum(0)

    for atom_idx in sorted(dummy_indices_to_remove, reverse=True):
        rw_mol.RemoveAtom(atom_idx)

    fragment = rw_mol.GetMol()
    Chem.SanitizeMol(fragment)
    return fragment


def _oriented_smiles_by_selected_dummy(
    fragment, selected_descriptor_map, dummy_position
):
    dummy_token = f"[*:{selected_descriptor_map}]"
    candidates = []

    if dummy_position == "first":
        dummy_atom = _dummy_atom_with_map(fragment, selected_descriptor_map)
        mapped_smiles = Chem.MolToSmiles(
            fragment, rootedAtAtom=dummy_atom.GetIdx(), canonical=False
        )
        if not mapped_smiles.startswith(dummy_token):
            raise ValueError(
                f"Could not orient selected wildcard first: {mapped_smiles}"
            )
        smiles = mapped_smiles[len(dummy_token) :]
        return smiles, mapped_smiles

    atom_orders = [list(range(fragment.GetNumAtoms()))]
    if fragment.GetNumAtoms() > 1:
        atom_orders.append(list(reversed(atom_orders[0])))

    for atom_order in atom_orders:
        ordered_fragment = (
            fragment
            if atom_order == atom_orders[0]
            else Chem.RenumberAtoms(fragment, atom_order)
        )
        for atom in ordered_fragment.GetAtoms():
            mapped_smiles = Chem.MolToSmiles(
                ordered_fragment,
                rootedAtAtom=atom.GetIdx(),
                canonical=False,
            )
            if mapped_smiles.endswith(dummy_token):
                candidates.append((len(mapped_smiles), mapped_smiles))

    if not candidates:
        raise ValueError(
            "Could not orient selected wildcard at the end of the SMILES string"
        )

    _, mapped_smiles = sorted(candidates)[0]
    smiles = mapped_smiles[: -len(dummy_token)]
    return smiles, mapped_smiles


def _dummy_atom_with_map(mol, descriptor_map):
    atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == descriptor_map
    ]
    if len(atoms) != 1:
        raise ValueError(
            f"Expected exactly one selected wildcard with map {descriptor_map}"
        )
    return atoms[0]


def _oriented_smiles(fragment, anchor_atom_map, wanted_anchor_position):
    candidates = []
    atom_orders = [list(range(fragment.GetNumAtoms()))]
    if fragment.GetNumAtoms() > 1:
        atom_orders.append(list(reversed(atom_orders[0])))

    for atom_ordering in atom_orders:
        ordered_fragment = (
            fragment
            if atom_ordering == atom_orders[0]
            else Chem.RenumberAtoms(fragment, atom_ordering)
        )
        for atom in ordered_fragment.GetAtoms():
            root = atom.GetIdx()
            mapped_smiles = Chem.MolToSmiles(
                ordered_fragment,
                rootedAtAtom=root,
                canonical=False,
            )
            atom_order = _atom_map_order(mapped_smiles)
            if not atom_order:
                continue
            if wanted_anchor_position == "first" and atom_order[0] == anchor_atom_map:
                candidates.append(
                    (len(mapped_smiles), mapped_smiles, root, ordered_fragment)
                )
            elif wanted_anchor_position == "last" and atom_order[-1] == anchor_atom_map:
                candidates.append(
                    (len(mapped_smiles), mapped_smiles, root, ordered_fragment)
                )

    if not candidates:
        raise ValueError(
            f"Could not orient fragment with anchor atom {anchor_atom_map} as "
            f"{wanted_anchor_position}. The selected descriptor may not be terminal."
        )

    _, mapped_smiles, root, selected_fragment = sorted(
        candidates,
        key=lambda candidate: candidate[:3],
    )[0]

    unmapped = Chem.Mol(selected_fragment)
    for atom in unmapped.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(unmapped, rootedAtAtom=root, canonical=False)

    return smiles, mapped_smiles


def _atom_map_order(smiles):
    return [int(value) for value in re.findall(r":(\d+)\]", smiles)]


if __name__ == "__main__":
    examples = [
        ("[<]CCO[>]", "left", "right"),
        ("[<]CCO[>]", "right", "right"),
        ("[<]CCO[>]", "left", "left"),
        ("[<]CCO[>]", "right", "left"),
        ("[<]CC(C)O[>]", "left", "right"),
        ("[<]CC(C)O[>]", "right", "right"),
        ("[$]CCO[$]", "left", "right"),
        ("[$]CCO[$]", "right", "right"),
    ]
    for repeat, repeat_side, insert_side in examples:
        fragment = orient_repeat_unit_fragment(repeat, repeat_side, insert_side)
        print(
            f"{repeat:18s} repeat_side={repeat_side:5s} insert={insert_side:5s} -> {fragment.smiles}"
        )

    print("\nWildcard attachment-site expansion")
    for insert_side in ["right", "left"]:
        fragments = orient_bigsmiles_repeat_unit_fragments(
            "[$]CCC(C)[$]", insert_side=insert_side
        )
        print(insert_side, [fragment.smiles for fragment in fragments])

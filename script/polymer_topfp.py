import numpy as np
import re
from collections import Counter

from rdkit import Chem

from . import search_tools

from .graphs import get_objects, get_repeats


def reorder_list_list(list_list):
    lengths = [len(sublist) for sublist in list_list]

    # Sort the sublists by decreasing length
    sorted_list_list = sorted(list_list, key=len, reverse=True)
    return sorted_list_list


def replace_bigsmiles_connectors(s: str) -> str:
    """
    Replace all BigSMILES connector tokens ([<], [>], [$], [<1], [>2], [$3]) with '*'
    """
    return re.sub(r"\[(?:<|>|\$)[0-9]*\]", "*", s)


def bigsmiles_to_molecular_fragment_smiles_list(BigSMILES):
    """Extract molecular-fragment SMILES from BigSMILES repeat units.

    Parameters
    ----------
    BigSMILES : str
        Polymer representation to parse.

    Returns
    -------
    list[str]
        Repeat-unit fragments used for chirality and stereochemistry bits.
    """
    objects = get_objects(BigSMILES)

    repeat_unit_string_list = []

    for object in objects[0]:

        # get all repeat units in the object
        repeats = get_repeats(object)
        for i in range(0, len(repeats)):
            if repeats[i] == []:
                continue

            for j in range(0, len(repeats[i])):
                repeat_unit_string = repeats[i][j]

                # Extract content inside curly braces
                inside_braces = re.findall(r"\{(.*?)\}", repeat_unit_string)

                if inside_braces != []:
                    # Remove the curly brace parts (including braces)
                    cleaned_repeat_unit_string = re.sub(
                        r"\{.*?\}", "", repeat_unit_string
                    )

                    repeat_unit_string_list.append(cleaned_repeat_unit_string)

                    for k in range(0, len(inside_braces)):
                        inside_string = remove_outer_square_brackets_tokens(
                            inside_braces[k]
                        )

                        if "," in inside_string:
                            # 把 [$]CCS[$],[$]CC([$])[$] 转换成list
                            # 对list中的每一个item进行处理
                            inside_string_repeat_unit_list = [
                                part for part in inside_string.split(",") if part
                            ]

                            for (
                                inside_string_repeat_unit_text
                            ) in inside_string_repeat_unit_list:

                                repeat_unit_string_list.append(
                                    inside_string_repeat_unit_text
                                )

                elif inside_braces == []:

                    repeat_unit_string_list.append(repeat_unit_string)

    return repeat_unit_string_list


def polymer_bigsmarts_search_one_hot_encoding(bigsmiles):

    bigsmarts_query_list = [
        "{[][]}",  # polymer
        "{[][]}?*{[][]}",  # block copolymer, at least 2 blocks
        "{[][]}?*{[][]}?*{[][]}",  # block copolymer, at least 3 blocks
        "{[][]}?*{[][]}?*{[][]}?*{[][]}",  # block copolymer, at least 4 blocks
        "{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}",  # block copolymer, at least 5 blocks
        "{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}",  # block copolymer, at least 6 blocks
        "{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}",  # block copolymer, at least 7 blocks
        "{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}?*{[][]}",  # block copolymer, at least 8 blocks
        "{[][<]?*{[>][<]?*[>][<]}?*[>][]}",  # segmented topology (nested object along the backbone)
        "{[][<]?*(?*{[>][<]?*[>][]})?*[>][]}",  # graft topology (nested object on the sidechain)
        "{[][]}?*(?*{[][]})?*{[][]}?*(?*{[][]})?*{[][]}",  # H polymer
        "{[][]}?*(?*{[][]})?*{[][]}",  # star polymer, at least 3 arms
        "{[][]}?*(?*{[][]})(?*{[][]})?*{[][]}",  # star polymer, at least 4 arms
        "{[][]}?*(?*{[][]})(?*{[][]})(?*{[][]})?*{[][]}",  # star polymer, at least 5 arms
        "{[][]}?*(?*{[][]})(?*{[][]})(?*{[][]})(?*{[][]})?*{[][]}",  # star polymer, at least 6 arms
        "{[][]}?*(?*{[][]})(?*{[][]})(?*{[][]})(?*{[][]})(?*{[][]})?*{[][]}",  # star polymer, at least 7 arms
        "{[][]}?*(?*{[][]})(?*{[][]})(?*{[][]})(?*{[][]})(?*{[][]})(?*{[][]})?*{[][]}",  # star polymer, at least 8 arms
    ]

    bigsmarts_search_one_hot_encoding = []

    for bigsmarts in bigsmarts_query_list:
        return_result = search_tools.graph_traversal_match(bigsmarts, bigsmiles)

        if return_result == True:
            bigsmarts_search_one_hot_encoding.append(1)

        else:
            bigsmarts_search_one_hot_encoding.append(0)

    return bigsmarts_search_one_hot_encoding


def build_chirality_vector(smiles_list):

    chiralisty_R = 0
    chiralisty_S = 0

    for smiles in smiles_list:
        update_smiles = replace_bigsmiles_connectors(smiles)
        mol = Chem.MolFromSmiles(update_smiles)
        if mol is not None:
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            for idx, chirality in chiral_centers:
                if chirality in ("R"):
                    chiralisty_R = True

                elif chirality in ("S"):
                    chiralisty_S = True
                else:
                    pass

    return [chiralisty_R, chiralisty_S]


def build_stereochemistry_vector(smiles_list):

    stereochemistry_trans = 0
    stereochemistry_cis = 0

    for smiles in smiles_list:
        update_smiles = replace_bigsmiles_connectors(smiles)
        mol = Chem.MolFromSmiles(update_smiles)
        if mol is None:
            continue

        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                stereo = bond.GetStereo()
                if stereo == Chem.rdchem.BondStereo.STEREOE:
                    stereochemistry_trans = 1
                elif stereo == Chem.rdchem.BondStereo.STEREOZ:
                    stereochemistry_cis = 1
                else:
                    pass

    return [stereochemistry_trans, stereochemistry_cis]


def is_left_to_left_pattern(s):
    pattern = r"^\[<\].*\[<\]$"
    return bool(re.match(pattern, s))


def is_right_to_right_pattern(s):
    pattern = r"^\[>\].*\[>\]$"
    return bool(re.match(pattern, s))


def remove_outer_square_brackets_tokens(text):
    # Remove the first and last [anything] tokens
    # Remove first [anything]
    text = re.sub(r"^\[[^\]]+\]", "", text, count=1)
    # Remove last [anything]
    text = re.sub(r"\[[^\]]+\]$", "", text, count=1)
    return text


def check_list_purity(symbols):
    has_numbered = False
    pure_pattern = re.compile(r"^\[\$]$|^\[>]$|^\[<]$")
    numbered_pattern = re.compile(r"^\[\$[0-9]+]$|^\[>[0-9]+]$|^\[<[0-9]+]$")

    for item in symbols:
        if pure_pattern.fullmatch(item):
            continue
        elif numbered_pattern.fullmatch(item):
            has_numbered = True
        else:
            print(f"Warning: unexpected item '{item}'")
            has_numbered = True  # treat unknown as impurity

    if has_numbered:
        return False

    else:
        return True


def check_cases_with_number_specified(symbols):
    dollar_pattern = re.compile(r"^\[\$([0-9]+)]$")
    left_pattern = re.compile(r"^\[<([0-9]+)]$")
    right_pattern = re.compile(r"^\[>([0-9]+)]$")

    dollar_numbers = []
    left_numbers = set()
    right_numbers = set()

    # Collect numbers
    for item in symbols:
        if m := dollar_pattern.fullmatch(item):
            dollar_numbers.append(m.group(1))
        elif m := left_pattern.fullmatch(item):
            left_numbers.add(m.group(1))
        elif m := right_pattern.fullmatch(item):
            right_numbers.add(m.group(1))

    # Case 1: at least two same [$number]
    dollar_counts = Counter(dollar_numbers)
    for num, count in dollar_counts.items():
        if count >= 2:
            pass
            return True

    # Case 2: at least one [<number] and one [>number], same number
    shared_numbers = left_numbers & right_numbers
    if shared_numbers:
        for num in shared_numbers:
            pass
        return True

    return False


def check_self_reaction_network(symbols_list=None):

    if check_list_purity(symbols_list) == True:
        num_dollar = symbols_list.count("[$]")
        num_left = symbols_list.count("[<]")
        num_right = symbols_list.count("[>]")

        if num_dollar >= 2:
            return True

        elif num_left >= 1 and num_right >= 1:
            return True

        else:
            return False
    else:
        num_dollar = symbols_list.count("[$]")
        num_left = symbols_list.count("[<]")
        num_right = symbols_list.count("[>]")

        if num_dollar >= 2:
            return True

        elif num_left >= 1 and num_right >= 1:
            return True

        elif check_cases_with_number_specified(symbols_list):
            return True

        else:
            return False


def process_list(list_input=None):
    list_processed = []
    for i in range(0, len(list_input)):
        if (
            ("[<" in list_input[i])
            or ("[>" in list_input[i])
            or ("[$" in list_input[i])
        ):
            list_processed.append(list_input[i])
    return list_processed


def build_network_vector(BigSMILES):
    """Convert a BigSMILES string to its network-topology subvector.

    Parameters
    ----------
    BigSMILES : str
        Polymer representation to process.

    Returns
    -------
    numpy.ndarray
        Eighteen-element network-topology vector.
    """
    objects = get_objects(BigSMILES)

    network_vector = np.zeros(18, dtype=int)

    for object in objects[0]:

        # get all repeat units in the object
        repeats = get_repeats(object)
        for i in range(0, len(repeats)):
            if repeats[i] == []:
                continue

            connectting_symbol_list_in_one_block = []

            for j in range(0, len(repeats[i])):

                repeat_unit_string = repeats[i][j]

                # Extract content inside curly braces
                inside_braces = re.findall(r"\{(.*?)\}", repeat_unit_string)

                if inside_braces != []:
                    # Remove the curly brace parts (including braces)
                    cleaned_repeat_unit_string = re.sub(
                        r"\{.*?\}", "", repeat_unit_string
                    )

                    connectting_symbol_list_in_one_repeat_unit = re.findall(
                        r"\[[^\]]+\]", cleaned_repeat_unit_string
                    )

                    connectting_symbol_list_in_one_block.append(
                        process_list(connectting_symbol_list_in_one_repeat_unit)
                    )

                    for k in range(0, len(inside_braces)):

                        inside_string = remove_outer_square_brackets_tokens(
                            inside_braces[k]
                        )

                        if "," in inside_string:

                            inside_string_repeat_unit_list = [
                                part for part in inside_string.split(",") if part
                            ]

                            inside_braces_connectting_symbol_list_in_one_block = []

                            for (
                                inside_string_repeat_unit_text
                            ) in inside_string_repeat_unit_list:

                                inside_braces_connectting_symbol_list_in_one_repeat_unit = re.findall(
                                    r"\[[^\]]+\]", inside_string_repeat_unit_text
                                )
                                inside_braces_connectting_symbol_list_in_one_block.append(
                                    inside_braces_connectting_symbol_list_in_one_repeat_unit
                                )

                elif inside_braces == []:

                    connectting_symbol_list_in_one_repeat_unit = re.findall(
                        r"\[[^\]]+\]", repeat_unit_string
                    )

                    connectting_symbol_list_in_one_block.append(
                        process_list(connectting_symbol_list_in_one_repeat_unit)
                    )

            if len(connectting_symbol_list_in_one_block) == 1:

                if len(connectting_symbol_list_in_one_block[0]) >= 3:

                    network_vector[0] = 1

                    only_list = connectting_symbol_list_in_one_block[0]

                    if len(only_list) == 3:
                        network_vector[3] = 1

                    elif len(only_list) == 4:
                        network_vector[4] = 1

                    elif len(only_list) == 5:
                        network_vector[5] = 1

                    elif len(only_list) == 6:
                        network_vector[6] = 1

                    elif len(only_list) == 7:
                        network_vector[7] = 1

                    elif len(only_list) == 8:
                        network_vector[8] = 1

                    else:
                        pass

                    network_vector[16] = 1

            elif len(connectting_symbol_list_in_one_block) >= 2:

                sorted_lconnectting_symbol_list_in_one_block = reorder_list_list(
                    connectting_symbol_list_in_one_block
                )

                first_maximum_list = sorted_lconnectting_symbol_list_in_one_block[0]

                second_maximum_list = sorted_lconnectting_symbol_list_in_one_block[1]

                if len(first_maximum_list) >= 3:
                    network_vector[0] = 1
                    network_vector[1] = 1

                    # first
                    if len(first_maximum_list) == 3:
                        network_vector[3] = 1
                        if check_self_reaction_network(first_maximum_list):
                            network_vector[16] = 1

                    elif len(first_maximum_list) == 4:
                        network_vector[4] = 1
                        if check_self_reaction_network(first_maximum_list):
                            network_vector[16] = 1

                    elif len(first_maximum_list) == 5:
                        network_vector[5] = 1
                        if check_self_reaction_network(first_maximum_list):
                            network_vector[16] = 1

                    elif len(first_maximum_list) == 6:
                        network_vector[6] = 1
                        if check_self_reaction_network(first_maximum_list):
                            network_vector[16] = 1

                    elif len(first_maximum_list) == 7:
                        network_vector[7] = 1
                        if check_self_reaction_network(first_maximum_list):
                            network_vector[16] = 1

                    elif len(first_maximum_list) == 8:
                        network_vector[8] = 1
                        if check_self_reaction_network(first_maximum_list):
                            network_vector[16] = 1

                    else:
                        pass

                    # second
                    if len(second_maximum_list) == 2:
                        network_vector[9] = 1

                    elif len(second_maximum_list) == 3:
                        network_vector[10] = 1
                        if check_self_reaction_network(second_maximum_list):
                            network_vector[17] = 1

                    elif len(second_maximum_list) == 4:
                        network_vector[11] = 1
                        if check_self_reaction_network(second_maximum_list):
                            network_vector[17] = 1

                    elif len(second_maximum_list) == 5:
                        network_vector[12] = 1
                        if check_self_reaction_network(second_maximum_list):
                            network_vector[17] = 1

                    elif len(second_maximum_list) == 6:
                        network_vector[13] = 1
                        if check_self_reaction_network(second_maximum_list):
                            network_vector[17] = 1

                    elif len(second_maximum_list) == 7:
                        network_vector[14] = 1
                        if check_self_reaction_network(second_maximum_list):
                            network_vector[17] = 1

                    elif len(second_maximum_list) == 8:
                        network_vector[15] = 1
                        if check_self_reaction_network(second_maximum_list):
                            network_vector[17] = 1

                    else:
                        pass

                if (
                    len(connectting_symbol_list_in_one_block) >= 3
                    and len(first_maximum_list) >= 3
                ):
                    network_vector[0] = 1
                    network_vector[1] = 1
                    network_vector[2] = 1

    network_vector_list = network_vector.tolist()

    return network_vector_list


def build_linear_statistical_copolymer_vector(BigSMILES):

    objects = get_objects(BigSMILES)

    linear_stochastic_copolymer_vector = 0

    for object in objects[0]:

        repeats = get_repeats(object)

        for i in range(0, len(repeats)):
            if repeats[i] == []:

                continue

            connectting_symbol_list_in_one_block = []

            for j in range(0, len(repeats[i])):

                repeat_unit_string = repeats[i][j]

                # Extract content inside curly braces
                inside_braces = re.findall(r"\{(.*?)\}", repeat_unit_string)
                if inside_braces != []:
                    # Remove the curly brace parts (including braces)
                    cleaned_repeat_unit_string = re.sub(
                        r"\{.*?\}", "", repeat_unit_string
                    )

                    connectting_symbol_list_in_one_repeat_unit = re.findall(
                        r"\[[^\]]+\]", cleaned_repeat_unit_string
                    )

                    connectting_symbol_list_in_one_block.append(
                        connectting_symbol_list_in_one_repeat_unit
                    )

                    for k in range(0, len(inside_braces)):
                        inside_string = remove_outer_square_brackets_tokens(
                            inside_braces[k]
                        )

                        if "," in inside_string:

                            inside_string_repeat_unit_list = [
                                part for part in inside_string.split(",") if part
                            ]

                            inside_braces_connectting_symbol_list_in_one_block = []
                            for (
                                inside_string_repeat_unit_text
                            ) in inside_string_repeat_unit_list:

                                inside_braces_connectting_symbol_list_in_one_repeat_unit = re.findall(
                                    r"\[[^\]]+\]", inside_string_repeat_unit_text
                                )
                                inside_braces_connectting_symbol_list_in_one_block.append(
                                    inside_braces_connectting_symbol_list_in_one_repeat_unit
                                )

                            alternative_copolymer_exist_inside_braces = (
                                ["[<]", "[<]"]
                                in inside_braces_connectting_symbol_list_in_one_block
                                and ["[<]", "[<]"]
                                in inside_braces_connectting_symbol_list_in_one_block
                            )

                            if (
                                len(inside_braces_connectting_symbol_list_in_one_block)
                                == 2
                                and alternative_copolymer_exist_inside_braces
                            ):
                                pass

                            else:
                                linear_stochastic_copolymer_vector = 1

                elif inside_braces == []:

                    connectting_symbol_list_in_one_repeat_unit = re.findall(
                        r"\[[^\]]+\]", repeat_unit_string
                    )

                    connectting_symbol_list_in_one_block.append(
                        connectting_symbol_list_in_one_repeat_unit
                    )

            if len(connectting_symbol_list_in_one_block) >= 2:

                sorted_lconnectting_symbol_list_in_one_block = reorder_list_list(
                    connectting_symbol_list_in_one_block
                )

                first_maximum_list = sorted_lconnectting_symbol_list_in_one_block[0]

                if len(first_maximum_list) == 2:

                    alternative_copolymer_exist = [
                        "[<]",
                        "[<]",
                    ] in connectting_symbol_list_in_one_block and [
                        "[<]",
                        "[<]",
                    ] in connectting_symbol_list_in_one_block

                    if (
                        len(connectting_symbol_list_in_one_block) == 2
                        and alternative_copolymer_exist
                    ):
                        pass

                    else:
                        linear_stochastic_copolymer_vector = 1

    return [linear_stochastic_copolymer_vector]


def bigsmiles_topology_vector(bigsmiles):
    """Generate the 40-dimensional polymer topology fingerprint (p-TFP).

    Parameters
    ----------
    bigsmiles : str
        BigSMILES representation whose global topology is encoded.

    Returns
    -------
    numpy.ndarray
        Binary p-TFP vector with shape ``(40,)``.
    """
    list1 = polymer_bigsmarts_search_one_hot_encoding(bigsmiles)
    list2 = build_chirality_vector(
        bigsmiles_to_molecular_fragment_smiles_list(bigsmiles)
    )
    list3 = build_stereochemistry_vector(
        bigsmiles_to_molecular_fragment_smiles_list(bigsmiles)
    )
    list4 = build_linear_statistical_copolymer_vector(bigsmiles)
    list5 = build_network_vector(bigsmiles)

    topology_vector_list = list1 + list2 + list3 + list4 + list5
    topology_vector = np.array(topology_vector_list)
    return topology_vector

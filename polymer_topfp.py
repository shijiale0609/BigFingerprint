import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx
import re
import pandas as pd 
import os
import sys
import copy

import search_tools 
import polymer_ecfp

from graphs import build_atomistic, build_topology, get_objects, get_repeats, desc_regex
from adjacency import feed_to_bonds_n

from atomistic import Atom_Graph_Matcher
from adjacency import *


def reorder_list_list(list_list):
    lengths = [len(sublist) for sublist in list_list]

    # Sort the sublists by decreasing length
    sorted_list_list = sorted(list_list, key=len, reverse=True)
    return sorted_list_list


def polymer_bigsmarts_search_one_hot_encoding(bigsmiles):

    bigsmarts_query_list = [
        "{[][]}", #polymer
        "{[][]}?*{[][]}", # block copolymer, at least 2 blocks
        "{[][<]?*{[>][<]?*[>][<]}?*[>][]}", #segmented topology (nested object along the backbone)
        "{[][<]?*(?*{[>][<]?*[>][]})?*[>][]}", # graft topology (nested object on the sidechain)
        "{[][]}?*(?*{[][]})?*{[][]}", # star polymer, at least 3 arms
        "{[][]}?*(?*{[][]})?*{[][]}?*(?*{[][]})?*{[][]}", # H polymer 
        ]

    bigsmarts_search_one_hot_encoding = []

    for bigsmarts in bigsmarts_query_list:
        return_result = search_tools.graph_traversal_match(bigsmarts, bigsmiles)

        if return_result == True:
            bigsmarts_search_one_hot_encoding.append(1)

        else:
            bigsmarts_search_one_hot_encoding.append(0)

    return bigsmarts_search_one_hot_encoding


def number_of_brackets(bigsmiles):
    # Count the number of objects in the given BIGSMILES string
    count = 0
    for char in bigsmiles:
        if char == '{':
            count += 1
    return count


def number_of_objects(bigsmiles):
    # Count the number of objects in the given BIGSMILES string
    objects = get_objects(bigsmiles)
    count = len(objects[0])
    return count


def extract_valid_chirality_tokens(s):
    """
    Detect atoms with exactly one @ or exactly two @@ in square brackets.
    Reject any atom with more than two @.
    """
    brackets = re.findall(r'\[[^\]]+\]', s)
    at_tokens = []
    at_count = 0
    att_count = 0

    for b in brackets:
        # Count how many "@" exist in this atom
        at_matches = re.findall(r'@+', b)
        for match in at_matches:
            if match == '@':
                at_tokens.append('@')
                at_count += 1
            elif match == '@@':
                at_tokens.append('@@')
                att_count += 1
            else:
                # More than two @ is invalid — skip or warn
                continue

    chirality_list = []
    if bool(at_tokens) == True:
        chirality_list.append(1)
    else:
        chirality_list.append(0)
    
    chirality_list.append(at_count)
    chirality_list.append(att_count)
    
    return chirality_list


def check_trans_cis(bigsmiles):
    # trans
    trans_pattern  = r'(/C=C/|\\C=C\\)'
    # cis
    cis_pattern= r'(/C=C\\|\\C=C/)'
    
    has_trans = re.search(trans_pattern, bigsmiles) is not None
    
    has_cis = re.search(cis_pattern, bigsmiles) is not None
    
    trans_cis_list = []
    
    if has_trans == True:
        trans_cis_list.append(1)
    else:
        trans_cis_list.append(0)

    if has_cis == True:
        trans_cis_list.append(1)
    else:
        trans_cis_list.append(0)

    return trans_cis_list

def is_left_to_left_pattern(s):
    pattern = r'^\[<\].*\[<\]$'
    return bool(re.match(pattern, s))

def is_right_to_right_pattern(s):
    pattern = r'^\[>\].*\[>\]$'
    return bool(re.match(pattern, s))

def remove_outer_square_brackets_tokens(text):
    # Remove the first and last [anything] tokens
    # Remove first [anything]
    text = re.sub(r'^\[[^\]]+\]', '', text, count=1)
    # Remove last [anything]
    text = re.sub(r'\[[^\]]+\]$', '', text, count=1)
    return text


def bigsmiles_to_network_vector_linear_statistical_copolymer_vector(BigSMILES):
    """
    Converts a bigsmarts string to a network vector.
    
    Args:
        bigsmarts (str): The bigsmarts string to process
        
    Returns:
        numpy.ndarray: A 25-element network vector
    """
    objects = get_objects(BigSMILES)
    
    # Uncomment for debugging
    # print("objects", objects, len(objects[0]))
    
    number_of_objects = len(objects[0])
    
    network_vector = np.zeros(25, dtype=int)
    linear_stochastic_copolymer_vector = 0
    
    for object in objects[0]:
        
        # get all repeat units in the object
        repeats = get_repeats(object)
        # print("repeats", repeats, len(repeats), repeats[0])
        
        for i in range(0, len(repeats)):
            if repeats[i] == []:
                # print("skip")
                continue
            # print("i", i, repeats[i], len(repeats[i]))
                
            connectting_symbol_list_in_one_block = []
            
            for j in range(0, len(repeats[i])):
                
                # print(i, j, repeats[i][j])
                
                repeat_unit_string = repeats[i][j]
                
                # Extract content inside curly braces
                inside_braces = re.findall(r'\{(.*?)\}', repeat_unit_string)
                # print(inside_braces)
                
                if inside_braces != []:
                    # Remove the curly brace parts (including braces)
                    cleaned_repeat_unit_string = re.sub(r'\{.*?\}', '', repeat_unit_string)
                    
                    connectting_symbol_list_in_one_repeat_unit = re.findall(r'\[[^\]]+\]', cleaned_repeat_unit_string)
                    
                    connectting_symbol_list_in_one_block.append(connectting_symbol_list_in_one_repeat_unit)
                    
                    for k in range(0, len(inside_braces)):
                    #print(k,inside_braces[k])
                        inside_string = remove_outer_square_brackets_tokens(inside_braces[k])
                    #print(k, inside_string)
                    
                        if "," in inside_string:
                        #把 [$]CCS[$],[$]CC([$])[$] 转换成list
                        #对list中的每一个item进行处理
                            inside_string_repeat_unit_list = [part for part in inside_string.split(',') if part]
                            #print("inside_string_repeat_unit_list",inside_string_repeat_unit_list)
                        
                            inside_braces_connectting_symbol_list_in_one_block = []
                            for inside_string_repeat_unit_text in inside_string_repeat_unit_list:
                            
                                inside_braces_connectting_symbol_list_in_one_repeat_unit = re.findall(r'\[[^\]]+\]', inside_string_repeat_unit_text)
                                inside_braces_connectting_symbol_list_in_one_block.append(inside_braces_connectting_symbol_list_in_one_repeat_unit)
                        
                            alternative_copolymer_exist_inside_braces = ["[<]","[<]"] in inside_braces_connectting_symbol_list_in_one_block and ["[<]","[<]"] in inside_braces_connectting_symbol_list_in_one_block
                        
                            if  len(inside_braces_connectting_symbol_list_in_one_block) == 2 and  alternative_copolymer_exist_inside_braces:
                                pass
                        
                            else:
                                linear_stochastic_copolymer_vector = 1
                    
                            
                elif inside_braces == []:
                    
                    connectting_symbol_list_in_one_repeat_unit = re.findall(r'\[[^\]]+\]', repeat_unit_string)
            
                    connectting_symbol_list_in_one_block.append(connectting_symbol_list_in_one_repeat_unit)
                    
            # print(connectting_symbol_list_in_one_block)
            # print(len(connectting_symbol_list_in_one_block))
            
            if len(connectting_symbol_list_in_one_block) == 1:
                
                if len(connectting_symbol_list_in_one_block[0]) > 2:
                    
                    network_vector[0] = 1
                    only_list = connectting_symbol_list_in_one_block[0]
                    
                    network_vector[1] = len(only_list)
                    
                    only_list_unique_set = list(set(only_list))
                
                    unique_length_only_list = len(only_list_unique_set)
                
                    only_list_dollar_exist = "[$]" in only_list_unique_set
                    only_list_left_arrow_exist = "[<]" in only_list_unique_set
                    only_list_right_arrow_exist = "[>]" in only_list_unique_set
                    
                    unique_length = len(set(connectting_symbol_list_in_one_block[0]))
                    
                    if unique_length_only_list == 1 and only_list_dollar_exist:
                        # print("pure [$] branched polymers")
                        network_vector[2] = 1
                        
                    elif (unique_length_only_list == 2) and only_list_left_arrow_exist and only_list_right_arrow_exist:
                        # print("[<] and [>] branched polymers")
                        network_vector[3] = 1
                        
                    elif unique_length_only_list == 3:
                        # print("[$], [<] and [>] branched polymers")
                        network_vector[4] = 1
                    else:
                        # print("error. BigSMILES writing is not correct. Please check the BigSMILES.")
                        pass
                        
                    # Count the number of '[$]' elements
                    count_dollar_symbols = connectting_symbol_list_in_one_block[0].count('[$]')
                    network_vector[5] = count_dollar_symbols
                    
                    # Count the number of '[<]' and '[>]' elements
                    network_vector[6] = len(connectting_symbol_list_in_one_block[0]) - count_dollar_symbols
                    
            
            elif len(connectting_symbol_list_in_one_block) >= 2:
                

            
                sorted_lconnectting_symbol_list_in_one_block = reorder_list_list(connectting_symbol_list_in_one_block)
                
                first_maximum_list = sorted_lconnectting_symbol_list_in_one_block[0]
                
                if len(first_maximum_list) >2:

                    network_vector[7] = 1
                    
                    network_vector[8] = len(connectting_symbol_list_in_one_block)
                    
                    # print(first_maximum_list)
                    # the number of [] of the first maximum
                    network_vector[9] = len(first_maximum_list)
                
                    first_maximum_list_unique_set = list(set(first_maximum_list))
                
                    unique_length_first_maximum_list = len(first_maximum_list_unique_set)
                
                    first_list_dollar_exist = "[$]" in first_maximum_list_unique_set
                    first_list_left_arrow_exist = "[<]" in first_maximum_list_unique_set
                    first_list_right_arrow_exist = "[>]" in first_maximum_list_unique_set
                
                    if unique_length_first_maximum_list == 1 and first_list_dollar_exist:
                        network_vector[10] = 1
                    
                    elif (unique_length_first_maximum_list == 1) and (first_list_left_arrow_exist or first_list_right_arrow_exist):
                        network_vector[11] = 1
                    
                    elif (unique_length_first_maximum_list == 2) and (first_list_left_arrow_exist and first_list_right_arrow_exist):
                        network_vector[12] = 1
                    
                    elif (unique_length_first_maximum_list == 2) and first_list_dollar_exist and (first_list_left_arrow_exist or first_list_right_arrow_exist):
                        network_vector[13] = 1   
                
                    elif unique_length_first_maximum_list == 3:
                        network_vector[14] = 1 
                    
                    else:
                    # print("error, please check the format of BigSMILES")
                        pass
                    
                    first_list_count_dollar_symbols = first_maximum_list.count('[$]')
                    network_vector[15] = first_list_count_dollar_symbols
                    network_vector[16] = len(first_maximum_list) - first_list_count_dollar_symbols
                
     
                    second_maximum_list = sorted_lconnectting_symbol_list_in_one_block[1]
                    # print(first_maximum_list)
                    # the number of [] of the first maximum
                    network_vector[17] = len(second_maximum_list)
                
                    second_maximum_list_unique_set = list(set(second_maximum_list))
                
                    unique_length_second_maximum_list = len(second_maximum_list_unique_set)
                
                    second_list_dollar_exist = "[$]" in second_maximum_list_unique_set
                    second_list_left_arrow_exist = "[<]" in second_maximum_list_unique_set
                    second_list_right_arrow_exist = "[>]" in second_maximum_list_unique_set
                
                    if unique_length_second_maximum_list == 1 and second_list_dollar_exist:
                        network_vector[18] = 1
                    
                    elif (unique_length_second_maximum_list == 1) and (second_list_left_arrow_exist or second_list_right_arrow_exist):
                        network_vector[19] = 1
                    
                    elif (unique_length_second_maximum_list == 2) and (second_list_left_arrow_exist and second_list_right_arrow_exist):
                        network_vector[20] = 1
                    
                    elif (unique_length_second_maximum_list == 2) and second_list_dollar_exist and (second_list_left_arrow_exist or second_list_right_arrow_exist):
                        network_vector[21] = 1   
                
                    elif unique_length_second_maximum_list == 3:
                        network_vector[22] = 1 
                    
                    else:
                        # print("error, please check the format of BigSMILES")
                        pass
                    
                    second_list_count_dollar_symbols = second_maximum_list.count('[$]')
                    network_vector[23] = second_list_count_dollar_symbols
                    network_vector[24] = len(second_maximum_list) - second_list_count_dollar_symbols
                
                elif len(first_maximum_list) == 2:
                    #print("linear copolymer")
                    alternative_copolymer_exist = ["[<]","[<]"] in connectting_symbol_list_in_one_block and ["[<]","[<]"] in connectting_symbol_list_in_one_block
                    
                    if len(connectting_symbol_list_in_one_block)==2 and alternative_copolymer_exist:
                        pass
                    else:
                        linear_stochastic_copolymer_vector = 1
                        
                
                else:
                    print("error")
                    
                    pass
            else:
                # print("no repeat units")
                pass
            
    network_vector_linear_stochastic_copolymer_vector_list = network_vector.tolist()
    #print(network_vector_list)
    
    network_vector_linear_stochastic_copolymer_vector_list.append(linear_stochastic_copolymer_vector) 
    
    return network_vector_linear_stochastic_copolymer_vector_list



def bigsmiles_topology_vector(bigsmiles):
    list1 = polymer_bigsmarts_search_one_hot_encoding(bigsmiles)
    list2 = [number_of_brackets(bigsmiles)]
    list3 = [number_of_objects(bigsmiles)]
    list4 = extract_valid_chirality_tokens(bigsmiles)
    list5 = check_trans_cis(bigsmiles)
    list6 = bigsmiles_to_network_vector_linear_statistical_copolymer_vector(bigsmiles)

    topology_vector_list = list1 + list2 + list3 + list4 + list5 + list6
    # print("topology_vector_list", topology_vector_list)
    topology_vector = np.array(topology_vector_list)
    # print("topology_vector", topology_vector)
    return topology_vector

bigsmiles_example = "{[][<]CCCC[<],[>]CC([>])C([>])C[>][]}"
print(bigsmiles_topology_vector(bigsmiles_example), bigsmiles_topology_vector(bigsmiles_example).shape)
